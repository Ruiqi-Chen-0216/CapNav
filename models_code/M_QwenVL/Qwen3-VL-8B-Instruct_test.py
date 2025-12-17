from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch

# =============================
# 1️⃣ 设备检测
# =============================
local_model_path = "/scr/models/Qwen3-VL-8B-Instruct"

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"🧠 Using GPU: {gpu_name} ({total_mem:.1f} GB)")
else:
    print("⚠️ No GPU detected, using CPU. This will be extremely slow.")

print("🚀 Loading model from:", local_model_path)

# =============================
# 2️⃣ 加载模型与处理器
# =============================
model = Qwen3VLForConditionalGeneration.from_pretrained(
    local_model_path,
    dtype="auto",
    device_map="auto",
    trust_remote_code=True
)

processor = AutoProcessor.from_pretrained(local_model_path, trust_remote_code=True)

# =============================
# 3️⃣ 输入（视频 + 文本）
# =============================
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/gscratch/makelab/ruiqi/videos/HM3D00000.mp4",
                "fps": 1.0  # 每秒取1帧，控制输入长度
            },
            {"type": "text", "text": "Describe what happens in this video."},
        ],
    }
]

# =============================
# 4️⃣ 构造输入张量
# =============================
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = inputs.to(model.device)

# =============================
# 5️⃣ 推理生成
# =============================
print("🎬 Running inference on video...")
generated_ids = model.generate(**inputs, max_new_tokens=256)

generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]

output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)
print("\n================= 🧾 OUTPUT =================")
print(output_text[0])
