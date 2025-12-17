#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Video-R1 Inference Script (Strict Example Parameters + Stability Optimized)
--------------------------------------------------------------------------
- 使用官方 example 中的 SamplingParams 参数
- 从自定义 prompt 文件读取内容（不修改原文）
- 自动追加 <think>/<answer> 推理模板
- 仅输出 <answer> JSON
- 修复已知警告 (rescale, max_pixels)
"""

import os
import re
import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info

# ======================================================
# 1️⃣ 路径设置
# ======================================================
model_path = "Video-R1/Video-R1-7B"
video_path = "/gscratch/makelab/ruiqi/videos_64frames_1fps/HM3D00000test.mp4"
prompt_file = "/gscratch/makelab/ruiqi/generated_prompts/HM3D00000test/q01_HUMAN.txt"

# ======================================================
# 2️⃣ 读取自定义 Prompt
# ======================================================
if not os.path.exists(prompt_file):
    raise FileNotFoundError(f"❌ Prompt file not found: {prompt_file}")

with open(prompt_file, "r", encoding="utf-8") as f:
    custom_prompt = f.read().strip()

# ======================================================
# 3️⃣ 拼接思考模板（严格保留原 prompt）
# ======================================================
final_prompt = (
    custom_prompt
    + "\n\n"
    + "Please think about this question as if you were a human pondering deeply. "
      "Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, "
      "or other natural language thought expressions. "
      "It's encouraged to include self-reflection or verification in the reasoning process. "
      "Provide your detailed reasoning between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags. "
      "Return only the <answer> section content as your final JSON output."
)

# ======================================================
# 4️⃣ 初始化模型 (严格按照 example)
# ======================================================
llm = LLM(
    model=model_path,
    tensor_parallel_size=1,
    max_model_len=81920,
    gpu_memory_utilization=0.8,
    enforce_eager=True,  # 🔧 避免 cudagraph 内存捕获警告
    limit_mm_per_prompt={"video": 1, "image": 1},
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    max_tokens=1024,
)

# ======================================================
# 5️⃣ 加载 Processor 和 Tokenizer
# ======================================================
processor = AutoProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"
processor.tokenizer = tokenizer

# 🔧 修复重复 rescale 警告
if hasattr(processor, "image_processor"):
    processor.image_processor.do_rescale = False
if hasattr(processor, "video_processor"):
    processor.video_processor.do_rescale = False

# ======================================================
# 6️⃣ 构造输入消息
# ======================================================
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "max_pixels": 102400,  # 🔧 降低以避免超限 warning (105369)
                "nframes": 32
            },
            {
                "type": "text",
                "text": final_prompt,
            },
        ],
    }
]

# ======================================================
# 7️⃣ 生成输入格式
# ======================================================
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)

llm_inputs = [{
    "prompt": prompt,
    "multi_modal_data": {"video": video_inputs[0]},
    "mm_processor_kwargs": {key: val[0] for key, val in video_kwargs.items()},
}]

# ======================================================
# 8️⃣ 模型推理
# ======================================================
print("🚀 Running inference on Video-R1...")
outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
output_text = outputs[0].outputs[0].text.strip()

# ======================================================
# 9️⃣ 提取 <answer> 内容
# ======================================================
match = re.search(r"<answer>([\s\S]*?)</answer>", output_text)
if match:
    final_answer = match.group(1).strip()
else:
    # 提示用户检查模型输出
    preview = output_text[:300].replace("\n", " ")
    final_answer = output_text.strip()
    print(f"⚠️ Warning: <answer> tag not found. Showing full output preview:\n{preview}...")

print("\n================= FINAL ANSWER (JSON) =================\n")
print(final_answer)
print("\n=======================================================\n")

