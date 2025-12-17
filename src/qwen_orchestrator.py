from src.utils.config_utils import load_api_keys
from src.vlm_clients import init_qwen, query_qwen

# === Load API keys ===
api_keys = load_api_keys()
api_key = api_keys["qwen"]["api_key"]

# === Init client ===
client = init_qwen(api_key, region="intl")

# === Query ===
result = query_qwen(
    client,
    video_path="videos/HM3D00000.mp4",
    prompt=open("generated_prompts/HM3D00000/q01_HUMAN.txt").read(),
    model="qwen3-max",
    fps=2,
    enable_thinking=False,
)

print(result)
