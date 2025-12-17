#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spatial-MLLM Inference Script (Cluster Path Version)
----------------------------------------------------
- 使用集群路径视频文件
- 从外部 prompt 文件读取文本输入
- 保留原始推理逻辑与参数设置
"""

import os
import sys
import torch
import tyro
import time

# add workspace to sys.path
SPATIAL_MLLM_ROOT = "/scr/Spatial-MLLM"
SRC_DIR = os.path.join(SPATIAL_MLLM_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

print(f"✅ Added to sys.path: {SRC_DIR}")

from models import (
    Qwen2_5_VL_VGGTForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info


def main(
    video_path: str = "/gscratch/makelab/ruiqi/videos_64frames_1fps/HM3D00000test.mp4",
    prompt_file: str = "/gscratch/makelab/ruiqi/generated_prompts/HM3D00000test/q01_HUMAN.txt",
    model_type: str = "spatial-mllm-subset-sft",
    model_path: str = "Diankun/Spatial-MLLM-subset-sft",
    device: str = "cuda",
):
    # ==================================================
    # 读取 prompt 文件
    # ==================================================
    if not os.path.exists(prompt_file):
        raise FileNotFoundError(f"❌ Prompt file not found: {prompt_file}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # ==================================================
    # 模型加载 (保持原逻辑)
    # ==================================================
    torch.cuda.empty_cache()

    if "spatial-mllm" in model_type:
        model = Qwen2_5_VL_VGGTForConditionalGeneration.from_pretrained(
            model_path,
            #torch_dtype="bfloat16",
            torch_dtype=torch.float16,
            #attn_implementation="flash_attention_2",
            attn_implementation="eager",
        )
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    elif "qwen2-5-vl" in model_type:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            #torch_dtype="bfloat16",
            torch_dtype=torch.float16,
            #attn_implementation="flash_attention_2",
            attn_implementation="eager",
        )
        processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    model = model.to(device)

    # ==================================================
    # 构造输入消息
    # ==================================================
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "nframes": 16},
                {"type": "text", "text": text},
            ],
        }
    ]

    # ==================================================
    # 预处理与推理
    # ==================================================
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    _, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    if "spatial-mllm" in model_type:
        inputs.update({"videos_input": torch.stack(video_inputs) / 255.0})

    inputs = inputs.to(device)

    # ==================================================
    # 模型推理
    # ==================================================
    time_0 = time.time()
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.1,
        top_p=0.001,
        use_cache=True,
    )
    time_taken = time.time() - time_0

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    num_generated_tokens = sum(len(ids) for ids in generated_ids_trimmed)

    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    # ==================================================
    # 输出信息
    # ==================================================
    print(f"Time taken for inference: {time_taken:.2f} seconds")
    print(f"GPU Memory taken for inference: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    print(f"Number of generated tokens: {num_generated_tokens}")
    print(f"Time taken per token: {time_taken / num_generated_tokens:.4f} seconds/token")
    print(f"Output: {output_text}")


if __name__ == "__main__":
    tyro.cli(main, description="Run Spatial-MLLM inference (cluster path version).")
