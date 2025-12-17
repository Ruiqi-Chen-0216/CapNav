#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CapNav Orchestrator (Vertex Gemini Batch Version, Clean Output)
---------------------------------------------------------------
批量运行多场景视频推理任务，使用 Vertex AI + GCS 视频 + Service Account 凭据。

✅ 特性：
- 所有参数集中定义在主入口
- 自动检测 graph 文件夹
- 自动推断输出文件夹后缀（rawvideo / 64frames / 32frames / 16frames）
- 每题单独保存结果（断点续跑）
- 自动记录 failed.json
- 自动生成场景级汇总 results.json
"""

import os
import json
import time
from google import genai
from src.vlm_clients import gemini_client_credential as gemini_client


# ======================================================
# 1️⃣ 工具函数
# ======================================================

def detect_all_scenes(graph_dir: str):
    """自动检测 graph 文件夹下的所有场景"""
    scenes = [f.split("-graph.json")[0] for f in os.listdir(graph_dir) if f.endswith("-graph.json")]
    scenes.sort()
    print(f"📦 Detected {len(scenes)} scenes in {graph_dir}")
    return scenes


def load_prompts(scene_name: str, prompt_root: str):
    """加载某个场景的全部 prompt"""
    prompt_dir = os.path.join(prompt_root, scene_name)
    if not os.path.exists(prompt_dir):
        raise FileNotFoundError(f"No prompt folder found: {prompt_dir}")
    prompt_files = sorted([f for f in os.listdir(prompt_dir) if f.endswith(".txt")])
    prompts = []
    for fname in prompt_files:
        with open(os.path.join(prompt_dir, fname), "r", encoding="utf-8") as f:
            prompts.append((fname, f.read().strip()))
    return prompts


def run_single_prompt(client, prompt_text: str, video_uri: str, model: str, fps: float = 1.0):
    """执行单个 prompt 的 Gemini 调用"""
    return gemini_client.query_gemini_credential(
        client,
        prompt=prompt_text,
        video_uri=video_uri,
        model=model,
        fps=fps,  # ✅ 传入采样帧率
    )


# ======================================================
# 2️⃣ 主函数：运行单场景
# ======================================================

def run_scene(scene_name: str, client, model_name: str, bucket_name: str,
              prompt_root: str, result_root: str, video_subdir: str,
              rate_limit_delay: int, fps: float = 1.0):
    """运行一个场景下的所有 prompts"""

    # === 输出命名逻辑 ===
    if "64frames_1fps" in video_subdir:
        frames = int(64 * fps)
        frames = max(frames, 1)
        model_tag = f"{model_name}_{frames}frames"
    else:
        model_tag = f"{model_name}_rawvideo"

    # === 输出路径 ===
    out_dir = os.path.join(result_root, model_tag, scene_name)
    os.makedirs(out_dir, exist_ok=True)

    # === 视频路径 ===
    if video_subdir:
        video_uri = f"gs://{bucket_name}/{video_subdir}/{scene_name}.mp4"
    else:
        video_uri = f"gs://{bucket_name}/{scene_name}.mp4"

    prompts = load_prompts(scene_name, prompt_root)

    print(f"\n🎬 Scene: {scene_name}")
    print(f"📹 Video: {video_uri}")
    print(f"🧩 Prompts: {len(prompts)}")
    print(f"📂 Output directory: {out_dir}")
    print(f"🎞️ Sampling rate: {fps} fps → Output tag: {model_tag}")

    summary = []
    failed_list = []
    start_time = time.time()

    for idx, (prompt_file, prompt_text) in enumerate(prompts, start=1):
        prompt_name = os.path.splitext(prompt_file)[0]
        out_file = os.path.join(out_dir, f"{prompt_name}.json")

        if os.path.exists(out_file):
            print(f"⏭️ Skipping {prompt_name} (already done).")
            with open(out_file, "r", encoding="utf-8") as f:
                summary.append(json.load(f))
            continue

        print(f"\n🚀 Prompt {idx}/{len(prompts)} → {prompt_file}")
        t0 = time.time()
        entry = {"scene": scene_name, "prompt_file": prompt_file}

        try:
            result = run_single_prompt(client, prompt_text, video_uri, model_name, fps=fps)
            raw_text = result.get("text", "").strip()

            try:
                parsed = json.loads(raw_text)
                entry["result"] = parsed
                print("✅ Parsed JSON successfully.")
            except json.JSONDecodeError:
                print("⚠️ JSON parse failed, saving raw text instead.")
                entry["result"] = {"raw_text": raw_text}
                failed_list.append({
                    "scene": scene_name,
                    "prompt_file": prompt_file,
                    "note": "JSON parse failed"
                })

            entry["usage"] = result.get("usage")
            entry["time_sec"] = round(time.time() - t0, 2)

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved → {out_file}")

            summary.append(entry)

        except Exception as e:
            err_msg = str(e)
            print(f"❌ Exception: {err_msg}")
            failed_list.append({
                "scene": scene_name,
                "prompt_file": prompt_file,
                "error": err_msg
            })
            if "service agents are being provisioned" in err_msg.lower():
                print("🕐 Service agents provisioning... waiting 120s before retry.")
                time.sleep(120)

        if idx < len(prompts):
            elapsed = time.time() - start_time
            if elapsed < rate_limit_delay:
                wait_t = rate_limit_delay - elapsed
                print(f"⏳ Waiting {wait_t:.1f}s ...")
                time.sleep(wait_t)
            start_time = time.time()

    merged = os.path.join(out_dir, f"{scene_name}_results.json")
    with open(merged, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Scene completed → {merged}")

    if failed_list:
        failed_path = os.path.join(out_dir, f"{scene_name}_failed.json")
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed_list, f, indent=2, ensure_ascii=False)
        print(f"⚠️ {len(failed_list)} prompts failed → {failed_path}")


# ======================================================
# 3️⃣ 主入口：参数集中定义
# ======================================================

if __name__ == "__main__":
    # === 参数配置区 ===
    GEMINI_KEY_PATH = r"C:\Users\pendr\Desktop\CapNav\configs\gemini-research-key.json"
    GRAPH_DIR = r"C:\Users\pendr\Desktop\CapNav\graph"
    PROMPT_DIR = r"C:\Users\pendr\Desktop\CapNav\generated_prompts"
    RESULT_ROOT = r"C:\Users\pendr\Desktop\CapNav\results"
    BUCKET_NAME = "capnav-videos"

    # 🧩 选择输入类型（raw 或 processed）
    #VIDEO_SUBDIR = ""  # ✅ 原始视频 (rawvideo)
    VIDEO_SUBDIR = "videos_64frames_1fps"  # ✅ 下采样视频模式

    MODEL_NAME = "gemini-2.5-pro"
    LOCATION = "us-central1"
    RATE_LIMIT_DELAY = 5
    FPS = 0.25   # ✅ 自动映射到 32frames
    #FPS = 1.0   # ✅ 自动映射到 64frames

    # === 初始化环境与客户端 ===
    if not os.path.exists(GEMINI_KEY_PATH):
        raise FileNotFoundError(f"❌ Credential not found: {GEMINI_KEY_PATH}")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GEMINI_KEY_PATH
    print(f"🔑 GOOGLE_APPLICATION_CREDENTIALS set to: {GEMINI_KEY_PATH}")

    with open(GEMINI_KEY_PATH, "r", encoding="utf-8") as f:
        project_id = json.load(f)["project_id"]
    client = gemini_client.init_gemini_credential(project_id=project_id, location=LOCATION)

    # === 自动遍历场景 ===
    scene_list = detect_all_scenes(GRAPH_DIR)
    for scene in scene_list:
        run_scene(
            scene_name=scene,
            client=client,
            model_name=MODEL_NAME,
            bucket_name=BUCKET_NAME,
            prompt_root=PROMPT_DIR,
            result_root=RESULT_ROOT,
            video_subdir=VIDEO_SUBDIR,
            rate_limit_delay=RATE_LIMIT_DELAY,
            fps=FPS,
        )
