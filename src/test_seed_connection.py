"""
Minimal Doubao Seed Video Connectivity Test
-------------------------------------------
Verifies that:
1. API key in configs/api_keys.yaml is valid
2. Network connection to Ark Beijing endpoint works
3. The model can process a short MP4 video (base64-encoded)
"""

import os
import base64
from src.utils.config_utils import load_api_keys
from src.vlm_clients import init_seed_client


def encode_video_to_base64(video_path: str) -> str:
    """Encode a local MP4 video into base64 string."""
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def test_seed_video_connection():
    # === Load API key ===
    api_keys = load_api_keys()
    if "seed" not in api_keys or "api_key" not in api_keys["seed"]:
        raise KeyError("Missing Doubao Seed API key in configs/api_keys.yaml")
    api_key = api_keys["seed"]["api_key"]

    # === Initialize client ===
    client = init_seed_client(api_key)

    # === Load and encode test video ===
    scene_name = "HM3D00000test"  # 你的视频名（可替换）
    video_path = os.path.join("videos_seed", f"{scene_name}.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"⚠️ Missing test video: {video_path}")

    print(f"🎞️ Encoding video: {video_path} ...", end="", flush=True)
    base64_video = encode_video_to_base64(video_path)
    print(" ✅ Done")

    # === Send video + question ===
    print("🚀 Sending video to Doubao Seed model ...")
    completion = client.chat.completions.create(
        model="doubao-seed-1-6-251015",  # ✅ 轻量模型，支持视频
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/mp4;base64,{base64_video}",
                            "fps": 2  # 每秒2帧
                        },
                    },
                    {"type": "text", "text": "请你介绍一下这个视频的内容。"},
                ],
            }
        ],
    )

    # === Print model output ===
    print("✅ Connection successful!")
    print("🧠 Model output:\n", completion.choices[0].message.content)


if __name__ == "__main__":
    test_seed_video_connection()
