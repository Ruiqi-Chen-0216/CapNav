#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CapNav Orchestrator (Doubao Seed Batch Version — MP3D subset)
-------------------------------------------------------------
批量运行 Doubao Seed 视频推理任务，仅针对场景名以 'MP3D' 开头的场景。
"""

import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError  # ⬅️ 新增
from src.utils.config_utils import load_api_keys
from src.vlm_clients import init_seed_client, query_seed_video_structured


# ======================================================
# 工具函数
# ======================================================
def detect_all_scenes(graph_dir: str, prefix_filter: str = "MP3D"):
    """自动检测 graph 文件夹下所有符合前缀的场景"""
    all_scenes = [f.split("-graph.json")[0] for f in os.listdir(graph_dir) if f.endswith("-graph.json")]
    scenes = [s for s in all_scenes if s.startswith(prefix_filter)]
    scenes.sort()
    print(f"📦 Detected {len(scenes)} scenes starting with '{prefix_filter}' in {graph_dir}")
    return scenes


# ===============================
# 0️⃣ 通用：超时调用包装器（Windows 兼容）
# ===============================
def call_with_timeout(func, timeout_seconds: int, **kwargs):
    """在单线程池中以超时方式调用 func(**kwargs)。抛出 FuturesTimeoutError 表示超时。"""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(func, **kwargs)
        return fut.result(timeout=timeout_seconds)



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


# ======================================================
# 2️⃣ 主函数：运行单场景
# ======================================================
def run_scene(scene_name: str, client, model_name: str,
              prompt_root: str, result_root: str,
              video_root: str, rate_limit_delay: int, fps: int = 2,
              per_prompt_timeout: int = 600):
    """运行一个场景下的所有 prompts"""

    # === 视频路径 ===
    video_path = os.path.join(video_root, f"{scene_name}.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No video found for scene {scene_name}: {video_path}")

    # === 输出命名逻辑 ===
    if "64frames_1fps" in video_root:
        frames = int(64 * fps)
        frames = max(frames, 1)
        model_tag = f"{model_name}_{frames}frames"
    elif "videos_seed" in video_root:
        model_tag = f"{model_name}_rawvideo"
    else:
        model_tag = f"{model_name}_unknownsrc"

    # === 输出路径 ===
    out_dir = os.path.join(result_root, model_tag, scene_name)
    os.makedirs(out_dir, exist_ok=True)

    # === 加载 prompts ===
    prompts = load_prompts(scene_name, prompt_root)

    print(f"\n🎬 Scene: {scene_name}")
    print(f"📹 Video: {video_path}")
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
            # === 带 600s 超时的调用 ===
            result = call_with_timeout(
                query_seed_video_structured,
                timeout_seconds=per_prompt_timeout,
                client=client,
                video_path=video_path,
                prompt=prompt_text,
                model=model_name,
                thinking_type="enabled",  # 保持与原脚本一致
                fps=fps,
            )

            # === 返回解析 ===
            try:
                parsed = json.loads(result["json"])
                entry["result"] = parsed
                print("✅ Parsed JSON successfully.")
            except json.JSONDecodeError:
                print("⚠️ JSON parse failed, saving raw text instead.")
                entry["result"] = {"raw_text": result["json"]}
                failed_list.append({
                    "scene": scene_name,
                    "prompt_file": prompt_file,
                    "note": "JSON parse failed"
                })

            entry["time_sec"] = round(time.time() - t0, 2)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved → {out_file}")

            summary.append(entry)

        except FuturesTimeoutError:
            # === 超时兜底：写入失败占位结果，并记录 failed_list ===
            print(f"⏰ Timeout (> {per_prompt_timeout}s). Marking as failed.")
            entry["result"] = {
                "answer": "failed to answer the question",
                "path": [],
                "reasons": []
            }
            entry["time_sec"] = round(time.time() - t0, 2)
            failed_list.append({
                "scene": scene_name,
                "prompt_file": prompt_file,
                "error": f"timeout > {per_prompt_timeout}s"
            })
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved (timeout fallback) → {out_file}")
            summary.append(entry)

        except Exception as e:
            # === 其它异常也写入失败占位结果，避免丢条目 ===
            err_msg = str(e)
            print(f"❌ Exception: {err_msg}")
            entry["result"] = {
                "answer": "failed to answer the question",
                "path": [],
                "reasons": []
            }
            entry["time_sec"] = round(time.time() - t0, 2)
            failed_list.append({
                "scene": scene_name,
                "prompt_file": prompt_file,
                "error": err_msg
            })
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved (exception fallback) → {out_file}")
            summary.append(entry)

        # Rate limit delay
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
# 3️⃣ 主入口
# ======================================================
if __name__ == "__main__":
    # === 参数集中定义 ===
    GRAPH_DIR = r"C:\Users\pendr\Desktop\CapNav\graph"
    PROMPT_DIR = r"C:\Users\pendr\Desktop\CapNav\generated_prompts"
    RESULT_ROOT = r"C:\Users\pendr\Desktop\CapNav\results"

    # === 视频输入模式 ===
    VIDEO_ROOT = r"C:\Users\pendr\Desktop\CapNav\videos_seed"             # ✅ 原始视频
    # VIDEO_ROOT = r"C:\Users\pendr\Desktop\CapNav\videos_64frames_1fps"  # ✅ 下采样视频模式

    MODEL_NAME = "doubao-seed-1-6-251015"
    RATE_LIMIT_DELAY = 0
    FPS = 1.0  # ✅ 控制帧率计算 frames

    api_keys = load_api_keys()
    if "seed" not in api_keys or "api_key" not in api_keys["seed"]:
        raise KeyError("Missing Doubao Seed API key in configs/api_keys.yaml")

    api_key = api_keys["seed"]["api_key"]
    client = init_seed_client(api_key)

    # === 自动检测场景 ===
    scene_list = detect_all_scenes(GRAPH_DIR, prefix_filter="MP3D")
    for scene in scene_list:
        run_scene(
            scene_name=scene,
            client=client,
            model_name=MODEL_NAME,
            prompt_root=PROMPT_DIR,
            result_root=RESULT_ROOT,
            video_root=VIDEO_ROOT,
            rate_limit_delay=RATE_LIMIT_DELAY,
            fps=FPS,
            per_prompt_timeout=600,  # ⬅️ 600s 超时
        )


