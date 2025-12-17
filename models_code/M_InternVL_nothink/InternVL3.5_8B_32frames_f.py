#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python /gscratch/makelab/ruiqi/M_InternVL/InternVL3.5_8B_batch_resumable.py

"""
InternVL3.5 Orchestrator (Batch + Resume + Fail Log)
----------------------------------------------------
✅ 自动遍历所有场景（/graph）
✅ 跳过已完成的 prompt
✅ 无效 JSON 自动写入 fail_log.jsonl，不保存结果
✅ 每场景生成单题 JSON + 汇总 _results.json
"""

import os
import json
import time
import math
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


# ======================================================
# 1️⃣ 基础图像预处理
# ======================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
            best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        processed.append(resized_img.crop(box))
    if use_thumbnail and len(processed) != 1:
        processed.append(image.resize((image_size, image_size)))
    return processed

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=16):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())
    frame_indices = np.linspace(0, max_frame, num_segments, dtype=int)

    transform = build_transform(input_size=input_size)
    pixel_values_list, num_patches_list = [], []

    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pv = torch.stack([transform(t) for t in tiles])
        num_patches_list.append(pv.shape[0])
        pixel_values_list.append(pv)
    return torch.cat(pixel_values_list), num_patches_list


# ======================================================
# 2️⃣ 模型加载
# ======================================================
def init_model():
    print("🚀 Loading InternVL3.5 model...")
    path = '/scr/models/InternVL3_5-8B'
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        device_map="auto"
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token
    generation_config = dict(max_new_tokens=1024, do_sample=True)
    print("✅ Model ready.\n")
    return model, tokenizer, generation_config


# ======================================================
# 3️⃣ Prompt 与 Scene 管理
# ======================================================
def detect_all_scenes(graph_dir="/gscratch/makelab/ruiqi/graph"):
    scenes = [f.split("-graph.json")[0] for f in os.listdir(graph_dir) if f.endswith("-graph.json")]
    scenes.sort()
    print(f"📦 Detected {len(scenes)} scenes in {graph_dir}")
    return scenes

def load_prompts(scene_name):
    base = "/gscratch/makelab/ruiqi/generated_prompts"
    prompt_dir = os.path.join(base, scene_name)
    if not os.path.exists(prompt_dir):
        raise FileNotFoundError(f"No prompt folder found: {prompt_dir}")
    files = sorted([f for f in os.listdir(prompt_dir) if f.endswith(".txt")])
    return [(f, open(os.path.join(prompt_dir, f), "r", encoding="utf-8").read().strip()) for f in files]


# ======================================================
# 4️⃣ Fail Log 机制
# ======================================================
def log_failure(scene_name, prompt_name, error_message, elapsed):
    fail_dir = f"/gscratch/makelab/ruiqi/results/InternVL3.5/{scene_name}"
    os.makedirs(fail_dir, exist_ok=True)
    fail_path = os.path.join(fail_dir, "failed_prompts.jsonl")

    record = {
        "prompt": prompt_name,
        "error": error_message,
        "time_sec": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(fail_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"🧯 Logged FAIL → {fail_path}")


# ======================================================
# 5️⃣ 主推理逻辑
# ======================================================
def run_scene(scene_name, model, tokenizer, generation_config, num_segments=16):
    video_path = f"/gscratch/makelab/ruiqi/videos_64frames_1fps/{scene_name}.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Missing video: {video_path}")
        return

    prompts = load_prompts(scene_name)
    out_root = f"/gscratch/makelab/ruiqi/results/InternVL3.5-8B-32frames/{scene_name}"
    os.makedirs(out_root, exist_ok=True)

    print(f"\n🎬 Scene: {scene_name} | Prompts: {len(prompts)}")
    pixel_values, num_patches_list = load_video(video_path, num_segments=num_segments, max_num=1)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])

    summary = []
    for i, (fname, ptext) in enumerate(prompts, 1):
        prompt_name = os.path.splitext(fname)[0]
        out_path = os.path.join(out_root, f"{prompt_name}.json")
        if os.path.exists(out_path):
            print(f"⏭️ Skipping {fname} (already exists).")
            with open(out_path, "r", encoding="utf-8") as f:
                summary.append(json.load(f))
            continue

        print(f"🚀 Running {i}/{len(prompts)}: {fname}")
        t0 = time.time()
        entry = {"scene": scene_name, "prompt_file": fname}

        try:
            question = video_prefix + "\n" + ptext
            response, _ = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True
            )

            text = response.strip()
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:].strip()
            start_idx = min([i for i in [text.find("["), text.find("{")] if i != -1], default=-1)
            if start_idx > 0:
                text = text[start_idx:].strip()

            # ✅ 尝试解析为 JSON
            try:
                parsed = json.loads(text)
                entry["result"] = parsed
                entry["time_sec"] = round(time.time() - t0, 2)
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(entry, f, indent=2, ensure_ascii=False)
                summary.append(entry)
                print("✅ Parsed JSON successfully and saved.")

            except json.JSONDecodeError:
                elapsed = time.time() - t0
                print("⚠️ Output not valid JSON.")
                log_failure(scene_name, prompt_name, "JSONDecodeError", elapsed)

        except Exception as e:
            elapsed = time.time() - t0
            print(f"❌ Error in {fname}: {e}")
            log_failure(scene_name, prompt_name, repr(e), elapsed)

    # ✅ 保存汇总
    merged = os.path.join(out_root, f"{scene_name}_results.json")
    with open(merged, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Scene completed → {merged}\n")


# ======================================================
# 6️⃣ 主入口：遍历所有场景
# ======================================================
if __name__ == "__main__":
    model, tokenizer, generation_config = init_model()
    all_scenes = detect_all_scenes("/gscratch/makelab/ruiqi/graph")

    for scene in all_scenes:
        run_scene(scene, model, tokenizer, generation_config, num_segments=32)
