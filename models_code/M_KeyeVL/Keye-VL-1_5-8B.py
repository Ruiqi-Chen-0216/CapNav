#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python /gscratch/makelab/ruiqi/M_KeyeVL/Keye-VL-1.5_Orchestrator.py
"""
Keye-VL-1.5 Orchestrator (Batch + Resume + Fail Log)
----------------------------------------------------
✅ 自动遍历所有场景（/graph）
✅ 读取对应 prompt 与 video
✅ 设置 FPS=1 的视频输入
✅ 跳过已完成 prompt
✅ 输出 JSON + 汇总 results.json
✅ 异常写入 fail_log.jsonl
"""
# --- Full patch: disable flash_attn check completely ---
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Keye-VL-1.5 Orchestrator (Offline + FlashAttn Bypass)
"""

import json
import time
import torch
from transformers import AutoModel, AutoProcessor
from keye_vl_utils import process_vision_info


# ======================================================
# 1️⃣ 模型加载
# ======================================================
def init_model():
    print("🚀 Loading local Keye-VL-1.5-8B ...")
    
    # 你提前下载好的路径
    model_path = "/scr/models/Keye-VL-1_5-8B"

    # 确保目录存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Local model folder not found: {model_path}")

    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype="auto",
        trust_remote_code=True,
    ).eval().to("cuda")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print("✅ Local model and processor ready.\n")
    return model, processor


# ======================================================
# 2️⃣ 管理函数
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


def log_failure(scene_name, prompt_name, error_message, elapsed):
    fail_dir = f"/gscratch/makelab/ruiqi/results/Keye-VL-1.5/{scene_name}"
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
# 3️⃣ 主推理逻辑
# ======================================================
def run_scene(scene_name, model, processor, fps=1.0, max_frames=1024):
    video_path = f"/gscratch/makelab/ruiqi/videos_64frames_1fps/{scene_name}.mp4"
    if not os.path.exists(video_path):
        print(f"❌ Missing video: {video_path}")
        return

    prompts = load_prompts(scene_name)
    out_root = f"/gscratch/makelab/ruiqi/results/Keye-VL-1.5/{scene_name}"
    os.makedirs(out_root, exist_ok=True)

    print(f"\n🎬 Scene: {scene_name} | Prompts: {len(prompts)}")

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
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "fps": fps,
                            "max_frames": max_frames,
                        },
                        {"type": "text", "text": ptext},
                    ],
                }
            ]

            # === 准备输入 ===
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, mm_processor_kwargs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
                **mm_processor_kwargs,
            ).to("cuda")

            # === 模型推理 ===
            generated_ids = model.generate(**inputs, max_new_tokens=1024)
            trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
            output_text = processor.batch_decode(
                trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            # === 尝试解析 JSON ===
            try:
                parsed = json.loads(output_text)
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

    # === 汇总保存 ===
    merged = os.path.join(out_root, f"{scene_name}_results.json")
    with open(merged, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Scene completed → {merged}\n")


# ======================================================
# 4️⃣ 主入口
# ======================================================
if __name__ == "__main__":
    model, processor = init_model()
    all_scenes = detect_all_scenes("/gscratch/makelab/ruiqi/graph")

    for scene in all_scenes:
        run_scene(scene, model, processor, fps=1.0, max_frames=512)
