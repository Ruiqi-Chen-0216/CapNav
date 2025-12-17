"""
Describe First 20 Images from Scene Folder (Using gpt_client.upload_file)
------------------------------------------------------------------------
Uploads the first 20 images in videos_gpt/<scene_name>
using gpt_client.upload_file(), and asks GPT-5 for a holistic description.
"""

import os
from src.utils.config_utils import load_api_keys
from src.vlm_clients import init_gpt, upload_file


def describe_scene_images(scene_name="HM3D00000test", max_images=20):
    # === Load API key ===
    api_keys = load_api_keys()
    if "openai" not in api_keys or "api_key" not in api_keys["openai"]:
        raise KeyError("❌ Missing OpenAI GPT API key in configs/api_keys.yaml")

    api_key = api_keys["openai"]["api_key"]
    client = init_gpt(api_key)

    # === Locate folder ===
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    scene_dir = os.path.join(root_dir, "videos_gpt", scene_name)
    if not os.path.exists(scene_dir):
        raise FileNotFoundError(f"❌ No folder found: {scene_dir}")

    # === Select first N images ===
    images = sorted([
        f for f in os.listdir(scene_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith("._")
    ])[:max_images]

    if not images:
        raise RuntimeError(f"❌ No valid images found in {scene_dir}")

    print(f"🖼️ Preparing to upload {len(images)} images from {scene_name} ...")

    # === Upload via gpt_client.upload_file ===
    file_ids = []
    for i, fname in enumerate(images, start=1):
        fpath = os.path.join(scene_dir, fname)
        file_obj = upload_file(client, fpath, purpose="vision", wait_until_ready=True)
        file_ids.append(file_obj.id)
        if i % 5 == 0 or i == len(images):
            print(f"📤 Uploaded {i}/{len(images)} images...")

    print(f"✅ Uploaded {len(file_ids)} images for scene {scene_name}")

    # === Ask GPT for holistic description ===
    print("\n🧠 Asking GPT-5 for scene description ...")
    response = client.responses.create(
        model="gpt-5-pro",
        input=[
            {
                "role": "user",
                "content": (
                    [{"type": "input_text", "text": f"请综合描述这些图像（来自场景 {scene_name}）的内容和空间布局。"}]
                    + [{"type": "input_image", "file_id": fid} for fid in file_ids]
                ),
            }
        ],
    )

    print("\n✅ Scene description result:")
    print(response.output_text.strip())

    if hasattr(response, "usage") and response.usage:
        usage = response.usage
        input_toks = getattr(usage, "input_tokens", 0)
        output_toks = getattr(usage, "output_tokens", 0)
        total_toks = getattr(usage, "total_tokens", 0)
        print(f"\n🧮 Token usage → input={input_toks}, output={output_toks}, total={total_toks}")


if __name__ == "__main__":
    describe_scene_images("HM3D00000", max_images=16)
