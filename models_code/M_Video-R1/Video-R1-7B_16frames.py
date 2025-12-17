#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video-R1 Orchestrator (Batch + Resume + Raw Output Save + Robust JSON Fallback)
--------------------------------------------------------
✅ 自动遍历所有场景（/graph）
✅ 跳过已完成的 prompt
✅ 启用 Thinking 模式（<think> 标签）
✅ JSON 解析失败自动写入 fallback result
✅ 每场景生成单题 JSON + 汇总 _results.json
✅ 同时保存带 <think> 的 raw_text
"""

import os
import re
import json
import time
import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info


# ======================================================
# 1️⃣ 初始化模型
# ======================================================
def init_llm(model_name: str):
    print(f"🚀 Loading {model_name} model...")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,
        max_model_len=81920,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
        limit_mm_per_prompt={"video": 1, "image": 1},
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        max_tokens=8196,
    )

    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    processor.tokenizer = tokenizer

    # 修复重复 rescale 警告
    if hasattr(processor, "image_processor"):
        processor.image_processor.do_rescale = False
    if hasattr(processor, "video_processor"):
        processor.video_processor.do_rescale = False

    print("✅ Model ready.\n")
    return llm, processor, sampling_params


# ======================================================
# 2️⃣ Scene & Prompt 管理
# ======================================================
def detect_all_scenes(graph_dir: str):
    scenes = [f.split("-graph.json")[0] for f in os.listdir(graph_dir) if f.endswith("-graph.json")]
    scenes.sort()
    print(f"📦 Detected {len(scenes)} scenes in {graph_dir}")
    return scenes


def load_prompts(prompt_root: str, scene_name: str):
    prompt_dir = os.path.join(prompt_root, scene_name)
    if not os.path.exists(prompt_dir):
        raise FileNotFoundError(f"❌ No prompt folder found: {prompt_dir}")
    files = sorted([f for f in os.listdir(prompt_dir) if f.endswith(".txt")])
    return [(f, open(os.path.join(prompt_dir, f), "r", encoding="utf-8").read().strip()) for f in files]


# ======================================================
# 3️⃣ Fail Log
# ======================================================
def log_failure(result_root, model_tag, scene_name, prompt_name, error_message, elapsed):
    fail_dir = os.path.join(result_root, "thinking", model_tag, scene_name)
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
# 4️⃣ Prompt 模板拼接
# ======================================================
def build_prompt(custom_prompt: str):
    return (
        custom_prompt
        + "\n\n"
        + "Please think about this question as if you were a human pondering deeply. "
          "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc. "
          "Include self-reflection or verification within <think> and </think> tags. "
          "Then give your final JSON answer inside <answer></answer>."
    )


# ======================================================
# 5️⃣ 单场景运行逻辑
# ======================================================
def run_scene(scene_name, llm, processor, sampling_params, cfg):
    """执行单个场景推理"""
    video_path = os.path.join(cfg["VIDEO_ROOT"], f"{scene_name}.mp4")
    if not os.path.exists(video_path):
        print(f"❌ Missing video: {video_path}")
        return

    prompts = load_prompts(cfg["PROMPT_ROOT"], scene_name)
    model_tag = f"{cfg['MODEL_NAME']}_{cfg['NUM_FRAMES']}frames"
    out_root = os.path.join(cfg["RESULT_ROOT"], "thinking", model_tag, scene_name)
    os.makedirs(out_root, exist_ok=True)

    print(f"\n🎬 Scene: {scene_name} | Prompts: {len(prompts)}")

    summary = []

    for i, (fname, ptext) in enumerate(prompts, 1):
        prompt_name = os.path.splitext(fname)[0]
        out_path = os.path.join(out_root, f"{prompt_name}.json")

        if os.path.exists(out_path):
            print(f"⏭️ Skipping {fname} (already exists)")
            with open(out_path, "r", encoding="utf-8") as f:
                summary.append(json.load(f))
            continue

        print(f"🚀 Running {i}/{len(prompts)}: {fname}")
        t0 = time.time()
        entry = {"scene": scene_name, "prompt_file": fname}

        try:
            final_prompt = build_prompt(ptext)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,
                            "max_pixels": 102400,
                            "nframes": cfg["NUM_FRAMES"],
                        },
                        {"type": "text", "text": final_prompt},
                    ],
                }
            ]

            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
            llm_inputs = [{
                "prompt": prompt,
                "multi_modal_data": {"video": video_inputs[0]},
                "mm_processor_kwargs": {k: v[0] for k, v in video_kwargs.items()},
            }]

            outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
            raw_text = outputs[0].outputs[0].text.strip()  # 带 <think> 的原始输出
            match = re.search(r"<answer>([\s\S]*?)</answer>", raw_text)
            if match:
                answer_json_str = match.group(1).strip()
            else:
                answer_json_str = raw_text

            try:
                parsed = json.loads(answer_json_str)
                entry["result"] = parsed
                print("✅ Parsed JSON successfully.")
            except json.JSONDecodeError:
                # ⚠️ JSON 解析失败 → 写 fallback
                print("⚠️ Output not valid JSON → Writing fallback result.")
                entry["result"] = {
                    "answer": "failed to answer the question",
                    "path": []
                }
                # 同时记录 fail log
                elapsed = time.time() - t0
                log_failure(cfg["RESULT_ROOT"], model_tag, scene_name, prompt_name, "JSONDecodeError", elapsed)

            # ✅ 不论是否成功，都保存 raw_text 与耗时
            entry["raw_text"] = raw_text
            entry["time_sec"] = round(time.time() - t0, 2)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)

            summary.append(entry)
            print("✅ Saved JSON & raw_text successfully.")

        except Exception as e:
            elapsed = time.time() - t0
            print(f"❌ Error in {fname}: {e}")
            log_failure(cfg["RESULT_ROOT"], model_tag, scene_name, prompt_name, repr(e), elapsed)

    merged = os.path.join(out_root, f"{scene_name}_results.json")
    with open(merged, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Scene completed → {merged}\n")


# ======================================================
# 6️⃣ 主入口：集中配置
# ======================================================
if __name__ == "__main__":
    # 集中配置所有参数与路径
    cfg = {
        "MODEL_NAME": "Video-R1/Video-R1-7B",
        "NUM_FRAMES": 16,
        "GRAPH_DIR": "/gscratch/makelab/ruiqi/graph",
        "PROMPT_ROOT": "/gscratch/makelab/ruiqi/generated_prompts",
        "VIDEO_ROOT": "/gscratch/makelab/ruiqi/videos_64frames_1fps",
        "RESULT_ROOT": "/gscratch/makelab/ruiqi/results",
    }

    llm, processor, sampling_params = init_llm(cfg["MODEL_NAME"])
    all_scenes = detect_all_scenes(cfg["GRAPH_DIR"])

    for scene in all_scenes:
        run_scene(scene, llm, processor, sampling_params, cfg)
