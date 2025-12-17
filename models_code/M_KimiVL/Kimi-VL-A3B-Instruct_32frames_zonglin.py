#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Kimi-VL A3B Instruct Orchestrator with Accelerate (Balanced Multi-GPU)
--------------------------------------------------------------------
✅ 均匀分配模型到多 GPU，解决显存不均问题
✅ 固定帧数 + 自动遍历场景
✅ 跳过已完成的 prompt
✅ 自动记录错误日志
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import time
import torch
import numpy as np
from PIL import Image
from decord import VideoReader, cpu
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.utils import infer_auto_device_map
from huggingface_hub import snapshot_download


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
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_idx = len(vr) - 1
    idxs = np.linspace(0, max_idx, num=num_frames, dtype=int)
    return [Image.fromarray(vr[i].asnumpy()).convert("RGB") for i in idxs]


def log_failure(model_name, scene_name, prompt_name, error_message, elapsed):
    fail_dir = f"/gscratch/makelab/ruiqi/results/{model_name}-32frames/{scene_name}"
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


def _log_max_memory(max_memory_dict):
    if not max_memory_dict:
        return
    readable = []
    for idx, budget in sorted(max_memory_dict.items()):
        gi_b = budget / (1024 ** 3)
        readable.append(f"cuda:{idx} <= {gi_b:.1f}GiB")
    print("🧮 Max memory budget per GPU: " + ", ".join(readable))


def build_device_plan(model, dtype=torch.float16, no_split_classes=None):
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return "cpu", None
    if num_gpus == 1:
        return "auto", None

    try:
        target_ratio = float(os.environ.get("KIMI_VL_DEVICE_MEMORY_RATIO", "0.92"))
    except ValueError:
        target_ratio = 0.92
    target_ratio = max(0.1, min(target_ratio, 0.99))

    desired_gpu_use = num_gpus
    env_min_gpu = os.environ.get("KIMI_VL_MIN_GPU_COUNT")
    if env_min_gpu:
        try:
            desired_gpu_use = min(num_gpus, max(1, int(env_min_gpu)))
        except ValueError:
            print(f"⚠️ Invalid KIMI_VL_MIN_GPU_COUNT={env_min_gpu}, using all GPUs.")

    last_max_memory = None
    last_exception = None
    attempt_ratio = target_ratio

    for attempt in range(min(num_gpus, 6)):
        max_memory = {}
        for idx in range(num_gpus):
            props = torch.cuda.get_device_properties(idx)
            budget = int(props.total_memory * attempt_ratio)
            max_memory[idx] = budget
        last_max_memory = max_memory

        try:
            device_map = infer_auto_device_map(
                model,
                dtype=dtype,
                max_memory=max_memory,
                no_split_module_classes=no_split_classes,
            )
            used_gpus = sorted(
                {
                    int(loc.split(":")[-1])
                    for loc in device_map.values()
                    if isinstance(loc, str) and loc.startswith("cuda")
                }
            )
            if len(used_gpus) >= desired_gpu_use:
                print(f"🔀 Inferred balanced device map across {len(used_gpus)} GPUs.")
                _log_max_memory(max_memory)
                return device_map, max_memory

            print(
                "ℹ️ Auto device map used only "
                f"{len(used_gpus)} GPU(s): {used_gpus}. Retrying with tighter budgets."
            )
        except Exception as exc:
            last_exception = exc
            print(f"⚠️ Auto device map attempt failed: {exc}. Retrying.")

        attempt_ratio *= 0.85
        attempt_ratio = max(0.1, attempt_ratio)

    if last_exception:
        print(
            "⚠️ Failed to infer a multi-GPU device map after retries. "
            "Falling back to sequential dispatch."
        )
    else:
        print(
            "ℹ️ Auto device map could not utilise the desired number of GPUs. "
            "Falling back to sequential dispatch."
        )
    if last_max_memory:
        _log_max_memory(last_max_memory)
    return "sequential", last_max_memory


def resolve_checkpoint_path(model_id):
    override = os.environ.get("KIMI_VL_CHECKPOINT_PATH")
    if override:
        if os.path.exists(override):
            print(f"📁 Using checkpoint override: {override}")
            return override
        print(f"⚠️ Provided checkpoint override not found: {override}. Ignoring.")

    try:
        path = snapshot_download(model_id, resume_download=True)
        print(f"☁️ Using checkpoint snapshot: {path}")
        return path
    except Exception as exc:
        raise RuntimeError(f"Failed to download checkpoint for {model_id}: {exc}") from exc


# ======================================================
# 2️⃣ 模型加载（Accelerate 版）
# ======================================================
def init_kimi_with_accelerate(model_id="moonshotai/Kimi-VL-A3B-Instruct"):
    print(f"🚀 Loading Kimi-VL with Accelerate balanced sharding: {model_id}")

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

    # Step 1: 创建空模型
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)

    no_split_module_classes = [
        "OPTDecoderLayer",
        "LlamaDecoderLayer",
        "MistralDecoderLayer",
    ]

    device_map, max_memory = build_device_plan(
        model,
        dtype=torch.float16,
        no_split_classes=no_split_module_classes,
    )

    checkpoint_path = resolve_checkpoint_path(model_id)

    # Step 2: 使用 Accelerate 均匀加载权重
    model = load_checkpoint_and_dispatch(
        model,
        checkpoint=checkpoint_path,
        device_map=device_map,  # ✅ 自动平衡显存负载
        max_memory=max_memory,
        dtype=torch.float16,
        no_split_module_classes=no_split_module_classes,  # 可根据模型结构修改
    )

    # Step 3: 添加 chat_template
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

    print("✅ Model distributed evenly across GPUs.")
    return model, processor, model_id.split("/")[-1]


# ======================================================
# 3️⃣ 单题推理
# ======================================================
def run_single_prompt(model, processor, frames, prompt_text,
                      max_new_tokens=1024, temperature=0.2):
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

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
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
    out_root = f"/gscratch/makelab/ruiqi/results/{model_name}-32frames/{scene_name}"
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
    torch.cuda.empty_cache()


# ======================================================
# 5️⃣ 主入口
# ======================================================
if __name__ == "__main__":
    model, processor, model_name = init_kimi_with_accelerate("moonshotai/Kimi-VL-A3B-Instruct")
    scenes = detect_all_scenes("/gscratch/makelab/ruiqi/graph")

    FIXED_FRAMES = 32
    for scene in scenes:
        run_scene(
            scene,
            model,
            processor,
            model_name=model_name,
            num_frames=FIXED_FRAMES,
            max_new_tokens=1024,
        )
