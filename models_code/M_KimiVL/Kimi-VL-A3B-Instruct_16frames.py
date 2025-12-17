#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python /gscratch/makelab/ruiqi/M_KimiVL/Kimi-VL-A3B-Instruct_32frames.py

"""
Kimi-VL A3B Instruct Orchestrator (Fixed Frames + Resume + Fail Log)
--------------------------------------------------------------------
✅ 固定帧数（不自动降级）
✅ 自动遍历所有场景（/graph）
✅ 跳过已完成的 prompt
✅ 无效 JSON 自动写入 fail_log.jsonl
✅ 每场景生成单题 JSON + 汇总 _results.json
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = "/scr/cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/scr/cache/huggingface/transformers"

import json
import time
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModelForCausalLM, AutoProcessor


# ======================================================
# 1️⃣ 工具函数
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


def get_video_path(scene_name):
    path = f"/gscratch/makelab/ruiqi/videos_64frames_1fps/{scene_name}.mp4"
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Missing video: {path}")
    return path


def sample_video_frames(video_path, num_frames=32):
    """均匀采样固定数量的视频帧为 PIL.Image 列表"""
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_idx = len(vr) - 1
    idxs = np.linspace(0, max_idx, num=num_frames, dtype=int)
    return [Image.fromarray(vr[i].asnumpy()).convert("RGB") for i in idxs]


def log_failure(model_name, scene_name, prompt_name, error_message, elapsed):
    fail_dir = f"/gscratch/makelab/ruiqi/results/{model_name}-16frames/{scene_name}"
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


def clean_to_json(text):
    """去除 ```json ``` 等围栏，只保留 JSON 主体"""
    s = text.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("json"):
            s = s[4:].strip()
    first = len(s)
    for ch in ["[", "{"]:
        pos = s.find(ch)
        if pos != -1:
            first = min(first, pos)
    if first != len(s):
        s = s[first:].strip()
    return s


# ======================================================
# 2️⃣ 模型加载
# ======================================================
# ======================================================
# 2️⃣ 模型加载
# ======================================================
def init_kimi(model_id="moonshotai/Kimi-VL-A3B-Instruct"):
    print(f"🚀 Loading Kimi-VL model from Hugging Face Hub: {model_id}")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
        local_files_only=False,
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        local_files_only=False,
    )

    # ✅ 添加 chat_template，仿照官方 Kimi 对话结构
    processor.chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "{{ bos_token + 'User: ' + message['content'][1]['text'] + '\\n' }}"
        "{% elif message['role'] == 'assistant' %}"
        "{{ 'Assistant: ' + message['content'][0]['text'] + '\\n' }}"
        "{% endif %}"
        "{% endfor %}"
        "{{ eos_token }}"
    )

    model_name = model_id.split("/")[-1]
    print(f"✅ Model ready → {model_name}\n")
    return model, processor, model_name



# ======================================================
# 3️⃣ 单题推理
# ======================================================
def run_single_prompt(model, processor, frames, prompt_text,
                      max_new_tokens=1024, temperature=0.2):
    """单题固定帧推理"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "<VIDEO_FRAMES>"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = processor(
        images=frames,
        text=text,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
        )

    trimmed = [out[len(in_ids):] for in_ids, out in zip(inputs.input_ids, output_ids)]
    response = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response.strip()


# ======================================================
# 4️⃣ 主推理逻辑
# ======================================================
def run_scene(scene_name, model, processor, model_name,
              num_frames=32, max_new_tokens=1024):
    video_path = get_video_path(scene_name)
    out_root = f"/gscratch/makelab/ruiqi/results/{model_name}-16frames/{scene_name}"
    os.makedirs(out_root, exist_ok=True)

    prompts = load_prompts(scene_name)
    print(f"\n🎬 Scene: {scene_name} | Prompts: {len(prompts)} | Frames: {num_frames}")

    try:
        frames = sample_video_frames(video_path, num_frames=num_frames)
    except Exception as e:
        print(f"❌ Failed to sample frames: {e}")
        return

    summary = []
    for i, (fname, ptext) in enumerate(prompts, 1):
        prompt_name = os.path.splitext(fname)[0]
        out_path = os.path.join(out_root, f"{prompt_name}.json")

        if os.path.exists(out_path):
            print(f"⏭️ Skipping {fname} (already exists).")
            with open(out_path, "r", encoding="utf-8") as f:
                try:
                    summary.append(json.load(f))
                except Exception:
                    pass
            continue

        print(f"🚀 Running {i}/{len(prompts)}: {fname}")
        t0 = time.time()
        entry = {"scene": scene_name, "prompt_file": fname, "model": model_name, "num_frames": num_frames}

        try:
            raw = run_single_prompt(model, processor, frames, ptext, max_new_tokens=max_new_tokens)
            text = clean_to_json(raw)
            parsed = json.loads(text)
            entry["result"] = parsed
            entry["time_sec"] = round(time.time() - t0, 2)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)
            summary.append(entry)
            print("✅ Parsed JSON successfully and saved.")
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as e:
            elapsed = time.time() - t0
            print(f"❌ CUDA OOM in {fname}: {e}")
            log_failure(model_name, scene_name, prompt_name, "CUDA_OOM", elapsed)
            torch.cuda.empty_cache()

        except json.JSONDecodeError:
            elapsed = time.time() - t0
            print("⚠️ Output not valid JSON.")
            log_failure(model_name, scene_name, prompt_name, "JSONDecodeError", elapsed)

        except Exception as e:
            elapsed = time.time() - t0
            print(f"❌ Error in {fname}: {e}")
            log_failure(model_name, scene_name, prompt_name, repr(e), elapsed)

    merged = os.path.join(out_root, f"{scene_name}_results.json")
    with open(merged, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Scene completed → {merged}\n")


# ======================================================
# 5️⃣ 主入口
# ======================================================
if __name__ == "__main__":
    model, processor, model_name = init_kimi("moonshotai/Kimi-VL-A3B-Instruct")
    scenes = detect_all_scenes("/gscratch/makelab/ruiqi/graph")

    FIXED_FRAMES = 16
    for scene in scenes:
        run_scene(
            scene,
            model,
            processor,
            model_name=model_name,
            num_frames=FIXED_FRAMES,
            max_new_tokens=1024,
        )
