#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python /gscratch/makelab/ruiqi/M_MimoVL/MiMo-VL-7B-RL-2508_64frames_no_think.py
"""
CapNav Orchestrator (MiMo-VL-7B-RL, No-Think Mode)
--------------------------------------------------
自动遍历 graph 文件夹下所有场景，对每个场景执行多轮视频推理任务（禁用思维模式）。

✅ 特性：
- 自动检测 graph 文件夹下所有场景
- 启用 /no_think 模式（强制禁用 reasoning）
- 跳过已完成的 prompt（断点续跑）
- 每题独立保存结果
- 自动生成场景级汇总
"""

import os
import json
import time
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


# ======================================================
# 1️⃣ 模型加载
# ======================================================
def init_local_model(model_dir="/scr/models/MiMo-VL-7B-RL-2508"):
    """初始化本地 MiMo-VL-7B-RL 模型"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🧠 Using GPU: {gpu_name} ({total_mem:.1f} GB)")
    else:
        print("⚠️ No GPU detected, using CPU. Expect slow performance.")

    print(f"🚀 Loading model from local path: {model_dir}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    print("✅ Model and processor loaded successfully.\n")
    return model, processor


# ======================================================
# 2️⃣ 场景与 prompt 加载
# ======================================================
def detect_all_scenes(graph_dir="/gscratch/makelab/ruiqi/graph"):
    """自动扫描 graph 文件夹获取所有场景名"""
    scenes = []
    for f in os.listdir(graph_dir):
        if f.endswith("-graph.json"):
            scenes.append(f.split("-graph.json")[0])
    scenes.sort()
    print(f"📦 Detected {len(scenes)} scenes in {graph_dir}")
    return scenes


def load_prompts(scene_name: str):
    """读取该场景下所有 prompt 文件"""
    base = "/gscratch/makelab/ruiqi/generated_prompts"
    prompt_dir = os.path.join(base, scene_name)
    if not os.path.exists(prompt_dir):
        raise FileNotFoundError(f"No prompt folder found: {prompt_dir}")

    files = sorted([f for f in os.listdir(prompt_dir) if f.endswith(".txt")])
    if not files:
        raise FileNotFoundError(f"No prompt files in {prompt_dir}")

    prompts = []
    for fname in files:
        fpath = os.path.join(prompt_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            prompts.append((fname, f.read().strip()))
    return prompts


def get_video(scene_name: str):
    vpath = f"/gscratch/makelab/ruiqi/videos_64frames_1fps/{scene_name}.mp4"
    if not os.path.exists(vpath):
        raise FileNotFoundError(f"❌ No video found for {scene_name}: {vpath}")
    return vpath


# ======================================================
# 3️⃣ 单个 prompt 推理（No-Think 模式）
# ======================================================
def run_single_prompt(model, processor, video_path, prompt_text, fps=1.0, max_new_tokens=1024):
    """对单个 prompt 执行视频推理（强制 /no_think 模式）"""

    # ✅ 保证 /no_think 是最后一部分
    if not prompt_text.strip().endswith("/no_think"):
        prompt_text = prompt_text.strip() + " /no_think"

    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "video", "path": os.path.abspath(video_path)},
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
            top_p=0.95,
            top_k=20,
            temperature=0.3,
            repetition_penalty=1.0,
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return text


# ======================================================
# 4️⃣ 批量运行场景（带断点续跑）
# ======================================================
def run_scene(scene_name, model, processor, model_name="MiMo-VL-7B-RL-2508",
              fps=1, max_new_tokens=1024, delay=0):
    video_path = get_video(scene_name)
    prompts = load_prompts(scene_name)

    out_dir = f"/gscratch/makelab/ruiqi/results/{model_name}_nothink_16frames/{scene_name}"
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

        # ✅ 跳过已存在结果
        if os.path.exists(out_file):
            print(f"⏭️ Skipping {prompt_name} (already exists).")
            with open(out_file, "r", encoding="utf-8") as f:
                summary_results.append(json.load(f))
            continue

        print(f"\n🚀 Running {i}/{len(prompts)} → {prompt_name}")
        entry = {"scene": scene_name, "prompt_file": prompt_file}
        t0 = time.time()

        try:
            output_text = run_single_prompt(model, processor, video_path, prompt_text, fps, max_new_tokens)

            # === Step 1. 自动截取 JSON 片段（防止前后文本干扰）===
            json_start = output_text.find("[")
            json_end = output_text.rfind("]")
            if json_start != -1 and json_end != -1:
                json_str = output_text[json_start:json_end+1].strip()
            else:
                json_str = output_text

            # === Step 2. 尝试解析 ===
            try:
                entry["result"] = json.loads(json_str)
                print("✅ Parsed JSON successfully.")
            except json.JSONDecodeError:
                print("⚠️ JSON parsing failed — logging to failed list.")
                failed_log = os.path.join(out_dir, f"{scene_name}_failed.json")
                failed_entry = {
                    "scene": scene_name,
                    "prompt_file": prompt_file,
                    "time_sec": round(time.time() - t0, 2),
                    "note": "JSON parse failed in /no_think mode"
                }
                if os.path.exists(failed_log):
                    with open(failed_log, "r", encoding="utf-8") as f:
                        failed_list = json.load(f)
                else:
                    failed_list = []
                failed_list.append(failed_entry)
                with open(failed_log, "w", encoding="utf-8") as f:
                    json.dump(failed_list, f, indent=2, ensure_ascii=False)
                continue

        except Exception as e:
            entry["error"] = str(e)
            print(f"❌ Exception during generation: {e}")

        entry["time_sec"] = round(time.time() - t0, 2)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved → {out_file}")

        summary_results.append(entry)
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
    MODEL_NAME = "MiMo-VL-7B-RL-2508"
    SCENE_LIST = detect_all_scenes("/gscratch/makelab/ruiqi/graph")

    model, processor = init_local_model("/scr/models/MiMo-VL-7B-RL-2508")

    for scene in SCENE_LIST:
        run_scene(scene, model, processor, model_name=MODEL_NAME, fps=0.25, max_new_tokens=8192, delay=0)
