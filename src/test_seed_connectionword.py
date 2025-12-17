"""
Minimal Doubao Seed Text Connectivity Test
------------------------------------------
Verifies that:
1. API key in configs/api_keys.yaml is valid
2. Network connection to Ark Beijing endpoint works
3. The model can respond to a simple text message
"""

from src.utils.config_utils import load_api_keys
from src.vlm_clients import init_seed_client


def test_seed_text_connection():
    # === Load API key ===
    api_keys = load_api_keys()
    if "seed" not in api_keys or "api_key" not in api_keys["seed"]:
        raise KeyError("Missing Doubao Seed API key in configs/api_keys.yaml")
    api_key = api_keys["seed"]["api_key"]

    # === Initialize client ===
    client = init_seed_client(api_key)

    # === Send a simple text query ===
    print("🚀 Testing Doubao Seed text-only connection ...")
    completion = client.chat.completions.create(
        model="doubao-seed-1-6-251015",  # ✅ 文本模型同样支持
        messages=[
            {"role": "user", "content": "你好，请介绍一下你是谁。"}
        ],
    )

    # === Print model output ===
    print("✅ Connection successful!")
    print("🧠 Model output:\n", completion.choices[0].message.content)


if __name__ == "__main__":
    test_seed_text_connection()
