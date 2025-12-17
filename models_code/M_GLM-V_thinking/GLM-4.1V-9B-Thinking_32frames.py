#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CapNav Orchestrator (GLM-4.1V, uniform sampling & output cleaning)
-----------------------------------------------------------------
批量对场景视频执行多轮多模态推理任务，兼容 GLM-4.1V 官方接口。

✅ 特性：
- 自动检测 graph 文件夹下所有场景
- 可自定义采样帧率 / 帧数（支持均匀采样）
- 跳过已完成 prompt（断点续跑）
- 自动清理 <think> 与 <answer> 标签
- 自动提取 JSON 数组并解析
- 生成 failed.json 与场景级汇总
"""

import os
import re
import json
import time
import torch
from transformers import AutoProcessor, Glm4vForConditionalGeneration


# ======================================================
# 1️⃣ 模型加载
# ======================================================
def init_glm4v_model(model_path="/scr/models/GLM-4.1V-9B-Thinking",
                     base_fps=1, total_frames=64, num_frames=64):
    """加载 GLM-4.1V 模型与 Processor，并智能配置视频采样参数"""
    print(f"🚀 Loading GLM-4.1V model from: {model_path}")
    model = Glm4vForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

    # ⚙️ 均匀采样逻辑：若视频64帧，则 fps = base_fps * (64 / num_frames)
    if num_frames >= total_frames:
        fps = base_fps
    else:
        fps = base_fps * (total_frames / num_frames)

    if hasattr(processor, "video_processor"):
        processor.video_processor.fps = fps
        processor.video_processor.num_frames = num_frames
        print(f"✅ Uniform sampling: {num_frames} frames evenly from {total_frames} @ fps={fps:.2f}")
    else:
        print("⚠️ Warning: processor has no video_processor (might be image-only).")

    print("✅ Model and processor loaded successfully.\n")
    return model, processor, fps


# ======================================================
# 2️⃣ 场景与 prompt 读取逻辑
# ======================================================
def detect_all_scenes(graph_dir="/gscratch/makelab/ruiqi/graph"):
    scenes = [f.split("-graph.json")[0] for f in os.listdir(graph_dir) if f.endswith("-graph.json")]
    scenes.sort()
    print(f"📦 Detected {len(scenes)} scenes in {graph_dir}")
    return scenes


def load_prompts(scene_name):
    base = "/gscratch/makelab/ruiqi/generated_prompts"
    prompt_dir = os.path.join(base, scene_name)
    files = sorted([f for f in os.listdir(prompt_dir) if f.endswith(".txt")])
    prompts = []
    for fname in files:
        with open(os.path.join(prompt_dir, fname), "r", encoding="utf-8") as f:
            prompts.append((fname, f.read().strip()))
    return prompts


def get_video(scene_name):
    vpath = f"/gscratch/makelab/ruiqi/videos_64frames_1fps/{scene_name}.mp4"
    if not os.path.exists(vpath):
        raise FileNotFoundError(f"❌ No video found for {scene_name}: {vpath}")
    return vpath


# ======================================================
# 3️⃣ 单个 prompt 推理 + 输出清理
# ======================================================
def run_single_prompt(model, processor, video_path, prompt_text,
                      fps=1.0, max_new_tokens=2048):
    """对单个 prompt 执行视频推理"""
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video_url": os.path.abspath(video_path)},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        conversation,
        fps=fps,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.3,
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    raw_text = processor.batch_decode(trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0].strip()

    # === Step 1. 移除 <think> ... </think> 段落 ===
    if "<think>" in raw_text:
        raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

    # === Step 2. 提取 <answer> ... </answer> 内部内容 ===
    match = re.search(r"<answer>(.*?)</answer>", raw_text, flags=re.DOTALL)
    if match:
        answer_block = match.group(1).strip()
    else:
        answer_block = raw_text  # fallback

    # === Step 3. 从 answer 中提取 JSON 数组 ===
    json_start = answer_block.find("[")
    json_end = answer_block.rfind("]")
    if json_start != -1 and json_end != -1:
        json_str = answer_block[json_start:json_end + 1].strip()
    else:
        json_str = answer_block

    return raw_text, json_str


# ======================================================
# 4️⃣ 批量运行场景（带断点续跑 + failed 记录 + 汇总）
# ======================================================
def run_scene(scene_name, model, processor, model_name="GLM-4.1V-9B-Thinking",
              fps=1, num_frames=64, max_new_tokens=2048, delay=0):
    video_path = get_video(scene_name)
    prompts = load_prompts(scene_name)

    out_dir = f"/gscratch/makelab/ruiqi/results/{model_name}_{num_frames}frames/{scene_name}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n🎬 Scene: {scene_name}")
    print(f"📹 Video: {os.path.basename(video_path)}")
    print(f"🧩 Total prompts: {len(prompts)}")
    print(f"📂 Output directory: {out_dir}\n")

    summary_results = []
    start_time = time.time()

    for i, (prompt_file, prompt_text) in enumerate(prompts, 1):
        prompt_name = os.path.splitext(prompt_file)[0]
        out_file = os.path.join(out_dir, f"{prompt_name}.json")

        if os.path.exists(out_file):
            print(f"⏭️ Skipping {prompt_name} (already exists).")
            with open(out_file, "r", encoding="utf-8") as f:
                summary_results.append(json.load(f))
            continue

        print(f"\n🚀 Running {i}/{len(prompts)} → {prompt_name}")
        entry = {"scene": scene_name, "prompt_file": prompt_file}
        t0 = time.time()

        try:
            raw_text, json_str = run_single_prompt(model, processor, video_path, prompt_text, fps, max_new_tokens)

            # === Step 4. 尝试解析 JSON ===
            try:
                entry["result"] = json.loads(json_str)
                print("✅ Cleaned and parsed JSON successfully.")
            except json.JSONDecodeError:
                print("⚠️ JSON parsing failed — logging this prompt to failed list.")
                failed_log = os.path.join(out_dir, f"{scene_name}_failed.json")
                failed_entry = {
                    "scene": scene_name,
                    "prompt_file": prompt_file,
                    "time_sec": round(time.time() - t0, 2),
                    "note": "JSON parse failed (malformed output)"
                }
                if os.path.exists(failed_log):
                    with open(failed_log, "r", encoding="utf-8") as f:
                        failed_list = json.load(f)
                else:
                    failed_list = []
                failed_list.append(failed_entry)
                with open(failed_log, "w", encoding="utf-8") as f:
                    json.dump(failed_list, f, indent=2, ensure_ascii=False)
                continue  # 跳过保存

            entry["result_text"] = raw_text
            entry["time_sec"] = round(time.time() - t0, 2)

            # 💾 保存单题结果
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved → {out_file}")

            summary_results.append(entry)

        except Exception as e:
            entry["error"] = str(e)
            print(f"❌ Exception during generation: {e}")

        # 限速控制
        if delay > 0 and i < len(prompts):
            elapsed = time.time() - start_time
            if elapsed < delay:
                wait_t = delay - elapsed
                print(f"⏳ Waiting {wait_t:.1f}s ...")
                time.sleep(wait_t)
            start_time = time.time()

    # ✅ 保存场景级汇总
    merged = os.path.join(out_dir, f"{scene_name}_results.json")
    with open(merged, "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Scene completed → {merged}")


# ======================================================
# 5️⃣ 主入口
# ======================================================
if __name__ == "__main__":
    MODEL_PATH = "/scr/models/GLM-4.1V-9B-Thinking"
    MODEL_NAME = "GLM-4.1V-9B-Thinking"
    GRAPH_DIR = "/gscratch/makelab/ruiqi/graph"

    BASE_FPS = 1
    TOTAL_FRAMES = 64
    NUM_FRAMES = 32  # 改成 16/32/64

    model, processor, FPS = init_glm4v_model(MODEL_PATH,
                                             base_fps=BASE_FPS,
                                             total_frames=TOTAL_FRAMES,
                                             num_frames=NUM_FRAMES)

    SCENE_LIST = detect_all_scenes(GRAPH_DIR)
    for scene in SCENE_LIST:
        run_scene(scene, model, processor,
                  model_name=MODEL_NAME,
                  fps=FPS,
                  num_frames=NUM_FRAMES,
                  max_new_tokens=8192,
                  delay=0)
