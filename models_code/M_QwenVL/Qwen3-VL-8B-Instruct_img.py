#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#python /gscratch/makelab/ruiqi/QwenVL/Qwen3-VL-8B-Instruct_img.py


"""
Qwen3-VL Local Orchestrator (Frame-Only + Resume + Fail Log)
------------------------------------------------------------
Batch-run Qwen3-VL-8B-Instruct using pre-extracted image frames instead of videos:
    /gscratch/makelab/ruiqi/frames/<scene_name>/*.jpg

Features:
✅ Loads the model only once
✅ Automatically detects all scenes
✅ Resumes from previously completed prompts
✅ Saves each prompt result separately
✅ Logs failures to fail_log.jsonl (no result file saved)
✅ Merges per-scene results into one _all.json file
✅ Uses sample_fps to simulate frame timestamps
"""

import os
import json
import time
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor


# ======================================================
# 1️⃣  Model loading
# ======================================================
def load_model(local_model_path: str):
    """Load Qwen3-VL-8B-Instruct locally with automatic GPU/CPU detection."""
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"🧠 Using GPU: {gpu_name} ({total_mem:.1f} GB)")
    else:
        print("⚠️ No GPU detected, using CPU (expect slow performance).")

    print(f"🚀 Loading model from: {local_model_path}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        local_model_path,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)
    print("✅ Model and processor loaded.\n")
    return model, processor


# ======================================================
# 2️⃣  Scene & prompt loading
# ======================================================
def detect_all_scenes(graph_root="/gscratch/makelab/ruiqi/graph"):
    """
    Detect all scene names by scanning the graph directory.

    Each scene is identified by files ending with "-graph.json",
    e.g., 'HM3D00000-graph.json' → scene name = 'HM3D00000'.
    """
    scenes = []
    for f in os.listdir(graph_root):
        if f.endswith("-graph.json"):
            scenes.append(f.replace("-graph.json", ""))

    scenes.sort()
    print(f"📦 Detected {len(scenes)} scenes in {graph_root}")
    return scenes



def load_prompts(scene_name: str):
    """Load all text prompt files for a specific scene."""
    prompt_dir = f"/gscratch/makelab/ruiqi/generated_prompts/{scene_name}"
    if not os.path.exists(prompt_dir):
        raise FileNotFoundError(f"No prompt folder found: {prompt_dir}")
    files = sorted([os.path.join(prompt_dir, f)
                    for f in os.listdir(prompt_dir) if f.endswith(".txt")])
    if not files:
        raise FileNotFoundError(f"No prompt files in {prompt_dir}")
    return files


# ======================================================
# 3️⃣  Failure logging
# ======================================================
def log_failure(scene_name, prompt_name, error_message, elapsed):
    """Write a failed-prompt record to fail_log.jsonl (append mode)."""
    fail_dir = f"/gscratch/makelab/ruiqi/results/Qwen3-VL-8B-Instruct/{scene_name}"
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
# 4️⃣  Build frame-based message input
# ======================================================
def build_frame_message(scene_name: str, prompt_text: str, fps: float = 1.0):
    """
    Construct the multimodal message input using a list of frames.
    """
    frame_dir = f"/gscratch/makelab/ruiqi/frames/{scene_name}"
    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"❌ Frame directory not found: {frame_dir}")

    frames = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    if not frames:
        raise FileNotFoundError(f"❌ No frames found in {frame_dir}")

    print(f"🖼️ Loaded {len(frames)} frames from {scene_name}")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frames,
                    "fps": fps,              # ✅ 正确键名
                    "do_sample_frames": False  # ✅ 明确告诉模型不要采样
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    return messages


# ======================================================
# 5️⃣  Run one prompt
# ======================================================
def run_single_prompt(model, processor, scene_name, prompt_path, out_dir, fps: float = 1.0):
    """Run a single prompt and save its result or log failure."""
    prompt_name = os.path.basename(prompt_path)
    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read().strip()

    messages = build_frame_message(scene_name, prompt_text, fps)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        do_sample_frames=False  # ✅ 保留这一行即可，删除 videos_kwargs
    ).to(model.device)



    print(f"\n🚀 Running → {prompt_name}")
    t0 = time.time()

    try:
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=True,
                top_p=0.95,
                top_k=20,
                temperature=1.0,
                repetition_penalty=1.0,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        # try parsing JSON output
        entry = {
            "scene": scene_name,
            "prompt_file": prompt_name,
            "time_sec": round(time.time() - t0, 2),
        }
        try:
            entry["result"] = json.loads(output_text)
            print("✅ Parsed JSON output.")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{prompt_name}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved result → {out_path}")
            return entry

        except json.JSONDecodeError:
            print("⚠️ Output is not valid JSON. Logging failure.")
            log_failure(scene_name, prompt_name, "JSONDecodeError", time.time() - t0)
            return None

    except Exception as e:
        print(f"❌ Exception: {e}")
        log_failure(scene_name, prompt_name, repr(e), time.time() - t0)
        return None


# ======================================================
# 6️⃣  Run all prompts within one scene (with resume)
# ======================================================
def run_scene(scene_name: str, model, processor, fps: float = 1.0):
    """Run all prompts for a given scene with resume and summary merge."""
    prompt_files = load_prompts(scene_name)
    out_dir = f"/gscratch/makelab/ruiqi/results/Qwen3-VL-8B-Instruct/{scene_name}"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n🎮 Scene: {scene_name}")
    print(f"🧩 Total prompts: {len(prompt_files)}")
    print(f"📂 Output directory: {out_dir}")

    all_results = []
    for idx, prompt_path in enumerate(prompt_files, 1):
        prompt_name = os.path.basename(prompt_path).replace(".txt", "")
        out_file = os.path.join(out_dir, f"{prompt_name}.json")

        # skip if result already exists
        if os.path.exists(out_file):
            print(f"⏭️ Skipping {prompt_name} (already exists).")
            with open(out_file, "r", encoding="utf-8") as f:
                all_results.append(json.load(f))
            continue

        print(f"\n===== Prompt {idx}/{len(prompt_files)} =====")
        entry = run_single_prompt(model, processor, scene_name, prompt_path, out_dir, fps)
        if entry:
            all_results.append(entry)

    # merge scene-level summary
    merged_path = os.path.join(out_dir, f"{scene_name}_all.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Scene completed. All results saved → {merged_path}")


# ======================================================
# 7️⃣  Main entry
# ======================================================
if __name__ == "__main__":
    MODEL_PATH = "/scr/models/Qwen3-VL-8B-Instruct"
    model, processor = load_model(MODEL_PATH)

    # ✅ Detect scenes based on graph files, not frame folders
    SCENES = detect_all_scenes("/gscratch/makelab/ruiqi/graph")

    for scene in SCENES:
        run_scene(scene, model, processor, fps=1.0)
