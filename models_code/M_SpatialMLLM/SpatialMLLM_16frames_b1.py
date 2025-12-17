#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial-MLLM Orchestrator (Batch + Raw/JSON Save + Robust Fallback)
-------------------------------------------------------------------
✅ 自动遍历所有场景（/graph）
✅ 批量读取 prompt 并推理
✅ 保存 raw_text 与 JSON 结果
✅ JSON 解析失败 → 写入固定 fallback
✅ 输出到 /gscratch/makelab/ruiqi/results/spatial_mllm
"""

import os
import re
import json
import time
import torch

# -----------------------------------------------------
# 基础路径设置
# -----------------------------------------------------
SPATIAL_MLLM_ROOT = "/scr/Spatial-MLLM"
SRC_DIR = os.path.join(SPATIAL_MLLM_ROOT, "src")
if SRC_DIR not in os.sys.path:
    os.sys.path.insert(0, SRC_DIR)
print(f"✅ Added to sys.path: {SRC_DIR}")

from models import Qwen2_5_VL_VGGTForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info


# =====================================================
# 1️⃣ 初始化模型
# =====================================================
def init_model(model_path="Diankun/Spatial-MLLM-subset-sft", device="cuda"):
    print(f"🚀 Loading model from {model_path}")
    model = Qwen2_5_VL_VGGTForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to(device)
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    return model, processor


# =====================================================
# 2️⃣ 扫描场景与 prompt
# =====================================================
def detect_all_scenes(graph_dir: str):
    scenes = [f.split("-graph.json")[0] for f in os.listdir(graph_dir) if f.endswith("-graph.json")]
    scenes.sort()
    print(f"📦 Detected {len(scenes)} scenes in {graph_dir}")
    return scenes


def load_prompts(prompt_root: str, scene_name: str):
    prompt_dir = os.path.join(prompt_root, scene_name)
    if not os.path.exists(prompt_dir):
        print(f"❌ No prompt folder found for {scene_name}")
        return []
    files = sorted([f for f in os.listdir(prompt_dir) if f.endswith(".txt")])
    return [(f, open(os.path.join(prompt_dir, f), "r", encoding="utf-8").read().strip()) for f in files]


# =====================================================
# 3️⃣ 单题推理
# =====================================================
def run_single_prompt(model, processor, video_path, text, device="cuda"):
    messages = [
        {"role": "user", "content": [
            {"type": "video", "video": video_path, "nframes": 16},
            {"type": "text", "text": text},
        ]}
    ]
    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text_input], videos=video_inputs, padding=True, return_tensors="pt")
    inputs.update({"videos_input": torch.stack(video_inputs) / 255.0})
    inputs = inputs.to(device)

    torch.cuda.empty_cache()
    t0 = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.1,
            top_p=0.001,
            use_cache=True,
        )
    elapsed = time.time() - t0

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_texts[0], elapsed


# =====================================================
# 4️⃣ 场景级运行逻辑
# =====================================================
def run_scene(scene_name, model, processor, cfg):
    video_path = os.path.join(cfg["VIDEO_ROOT"], f"{scene_name}.mp4")
    if not os.path.exists(video_path):
        print(f"❌ Missing video for {scene_name}")
        return

    prompts = load_prompts(cfg["PROMPT_ROOT"], scene_name)
    if not prompts:
        return

    out_dir = os.path.join(cfg["RESULT_ROOT"], scene_name)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n🎬 Scene: {scene_name} | Prompts: {len(prompts)}")

    results = []

    for i, (fname, ptext) in enumerate(prompts, 1):
        prompt_name = os.path.splitext(fname)[0]
        json_out = os.path.join(out_dir, f"{prompt_name}.json")

        if os.path.exists(json_out):
            print(f"⏭️ Skipping {fname} (already exists)")
            with open(json_out, "r", encoding="utf-8") as f:
                results.append(json.load(f))
            continue

        print(f"🚀 [{i}/{len(prompts)}] {fname}")
        t0 = time.time()
        entry = {"scene": scene_name, "prompt_file": fname}

        try:
            raw_text, elapsed = run_single_prompt(model, processor, video_path, ptext, cfg["DEVICE"])

            # 默认 fallback 结构
            result_fallback = {
                "answer": "failed to answer the question",
                "path": []
            }

            # 1️⃣ 尝试找到完整的 <json> ... </json> 块
            match = re.search(r"<json>([\s\S]*?)</json>", raw_text)
            if not match:
                print("⚠️ No <json>...</json> block detected — writing fallback")
                entry["result"] = result_fallback

            else:
                content = match.group(1).strip()

                # --- 提取 answer ---
                ans_match = re.search(r'"answer"\s*:\s*"([^"]+)"', content)
                answer = ans_match.group(1).strip() if ans_match else result_fallback["answer"]

                # --- 提取 reason (可选) ---
                reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', content)
                reason = reason_match.group(1).strip() if reason_match else None

                # --- 提取 path (可选) ---
                path_match = re.search(r'"path"\s*:\s*\[([^\]]*)\]', content)
                if path_match:
                    path_items = [
                        item.strip().strip('"').strip("'")
                        for item in path_match.group(1).split(",")
                        if item.strip()
                    ]
                else:
                    path_items = result_fallback["path"]

                # ✅ 组装结果
                result = {"answer": answer, "path": path_items}
                if reason:
                    result["reason"] = reason

                entry["result"] = result
                print("✅ Extracted result successfully (regex mode).")


            entry["raw_text"] = raw_text
            print(repr(raw_text))
            entry["time_sec"] = round(time.time() - t0, 2)

            with open(json_out, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)

            results.append(entry)
            print("✅ Saved:", json_out)

        except Exception as e:
            print(f"❌ Error in {fname}: {e}")
            entry["result"] = {
                "answer": "failed to answer the question",
                "path": []
            }
            entry["error"] = repr(e)
            entry["raw_text"] = ""
            with open(json_out, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)
            results.append(entry)

    merged_path = os.path.join(out_dir, f"{scene_name}_results.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Scene completed → {merged_path}")


# =====================================================
# 5️⃣ 主入口
# =====================================================
if __name__ == "__main__":
    cfg = {
        "GRAPH_DIR": "/gscratch/makelab/ruiqi/graph",
        "PROMPT_ROOT": "/gscratch/makelab/ruiqi/generated_prompts",
        "VIDEO_ROOT": "/gscratch/makelab/ruiqi/videos_64frames_1fps",
        "RESULT_ROOT": "/gscratch/makelab/ruiqi/results/spatial_mllm",
        "MODEL_PATH": "Diankun/Spatial-MLLM-subset-sft",
        "DEVICE": "cuda",
    }


    model, processor = init_model(cfg["MODEL_PATH"], cfg["DEVICE"])
    scenes = detect_all_scenes(cfg["GRAPH_DIR"])
    SCENE_LIST = [s for s in scenes if s.startswith("MP")]
    print(f"📂 Running {len(SCENE_LIST)} MP scenes:")
    print("   " + ", ".join(SCENE_LIST))
    for s in SCENE_LIST:
        run_scene(s, model, processor, cfg)
