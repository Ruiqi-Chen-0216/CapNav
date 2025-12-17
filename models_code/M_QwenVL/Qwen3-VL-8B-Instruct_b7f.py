#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CapNav Orchestrator (Local Qwen3-VL Batch Version — MP3D subset)
---------------------------------------------------------------
批量运行 Qwen3-VL 视频推理任务，仅针对场景名以 'MP3D' 开头的场景。
模仿 Doubao Seed Batch 版本结构，支持断点续跑与动态 frames 命名。
"""

import os
import json
import time
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


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


def load_prompts(scene_name: str, prompt_root: str):
    """读取指定场景下所有 prompt 文件"""
    prompt_dir = os.path.join(prompt_root, scene_name)
    if not os.path.exists(prompt_dir):
        raise FileNotFoundError(f"No prompt folder found: {prompt_dir}")
    prompt_files = sorted([f for f in os.listdir(prompt_dir) if f.endswith(".txt")])
    prompts = []
    for fname in prompt_files:
        with open(os.path.join(prompt_dir, fname), "r", encoding="utf-8") as f:
            prompts.append((fname, f.read().strip()))
    return prompts


def init_local_model(model_dir="/scr/models/Qwen3-VL-8B-Instruct"):
    """初始化本地 Qwen3-VL 模型"""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🧠 Using GPU: {gpu_name} ({total_mem:.1f} GB)")
    else:
        print("⚠️ No GPU detected, using CPU. Expect slow performance.")

    print(f"🚀 Loading model from: {model_dir}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_dir,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    print("✅ Model and processor loaded.\n")
    return model, processor


# ======================================================
# 单条 prompt 运行
# ======================================================
def run_single_prompt(model, processor, video_path, prompt_text, fps=1.0, max_new_tokens=1024):
    """执行单条视频+文本推理"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": os.path.abspath(video_path), "fps": fps,
                 "min_pixels": 4 * 32 * 32, "max_pixels": 256 * 32 * 32},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            top_k=20,
            temperature=1.0,
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
    text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return text


# ======================================================
# 主逻辑
# ======================================================
def run_scene(scene_name: str, model, processor,
              model_name: str, prompt_root: str, result_root: str,
              video_root: str, rate_limit_delay: int, fps: float = 1.0,
              base_frames: int = 64, max_new_tokens: int = 1024):
    """运行单个场景的所有 prompts"""

    video_path = os.path.join(video_root, f"{scene_name}.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No video found for scene {scene_name}: {video_path}")

    # ✅ 动态计算输出目录帧数后缀
    if "64frames_1fps" in video_root:
        frames = int(base_frames * fps)
        frames = max(frames, 1)
        model_tag = f"{model_name}_{frames}frames"
    elif "videos_seed" in video_root or "videos" in video_root:
        model_tag = f"{model_name}_rawvideo"
    else:
        model_tag = f"{model_name}_unknownsrc"

    out_dir = os.path.join(result_root, model_tag, scene_name)
    os.makedirs(out_dir, exist_ok=True)

    prompts = load_prompts(scene_name, prompt_root)

    print(f"\n🎬 Scene: {scene_name}")
    print(f"📹 Video: {video_path}")
    print(f"🧩 Prompts: {len(prompts)}")
    print(f"📂 Output directory: {out_dir}\n")

    summary, failed_list = [], []
    start_time = time.time()

    for idx, (prompt_file, prompt_text) in enumerate(prompts, start=1):
        out_file = os.path.join(out_dir, f"{os.path.splitext(prompt_file)[0]}.json")

        if os.path.exists(out_file):
            print(f"⏭️ Skipping {prompt_file} (already done).")
            with open(out_file, "r", encoding="utf-8") as f:
                summary.append(json.load(f))
            continue

        print(f"\n🚀 Prompt {idx}/{len(prompts)} → {prompt_file}")
        t0 = time.time()
        entry = {"scene": scene_name, "prompt_file": prompt_file}

        try:
            output_text = run_single_prompt(model, processor, video_path, prompt_text, fps=fps, max_new_tokens=max_new_tokens)
            try:
                entry["result"] = json.loads(output_text)
                print("✅ Parsed JSON successfully.")
            except json.JSONDecodeError:
                entry["result"] = {"raw_text": output_text}
                failed_list.append({"scene": scene_name, "prompt_file": prompt_file, "note": "JSON parse failed"})
                print("⚠️ JSON parse failed, saved raw text.")
        except Exception as e:
            err_msg = str(e)
            print(f"❌ Exception: {err_msg}")
            entry["error"] = err_msg
            failed_list.append({"scene": scene_name, "prompt_file": prompt_file, "error": err_msg})

        entry["time_sec"] = round(time.time() - t0, 2)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)
        summary.append(entry)

        # 限速
        if idx < len(prompts):
            elapsed = time.time() - start_time
            if elapsed < rate_limit_delay:
                time.sleep(rate_limit_delay - elapsed)
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
# 主入口
# ======================================================
if __name__ == "__main__":
    GRAPH_DIR = "/gscratch/makelab/ruiqi/graph"
    PROMPT_DIR = "/gscratch/makelab/ruiqi/generated_prompts"
    RESULT_ROOT = "/gscratch/makelab/ruiqi/results"
    VIDEO_ROOT = "/gscratch/makelab/ruiqi/videos_64frames_1fps"

    MODEL_NAME = "Qwen3-VL-8B-Instruct"
    RATE_LIMIT_DELAY = 0
    FPS = 0.5  # ✅ 自动换算为 32frames
    BASE_FRAMES = 64

    model, processor = init_local_model()
    scene_list =  [
        "MP3D00013",
        "MP3D00014",
        "MP3D00018",
        "MP3D00020",
        "MP3D00025",
    ]

    for scene in scene_list:
        run_scene(
            scene_name=scene,
            model=model,
            processor=processor,
            model_name=MODEL_NAME,
            prompt_root=PROMPT_DIR,
            result_root=RESULT_ROOT,
            video_root=VIDEO_ROOT,
            rate_limit_delay=RATE_LIMIT_DELAY,
            fps=FPS,
            base_frames=BASE_FRAMES,
        )
