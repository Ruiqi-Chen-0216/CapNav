"""
CapNav Orchestrator (clean version)
----------------------------------
Uses:
- utils.config_utils.load_api_keys() for config loading
- vlm_clients.gemini_client for API calls
"""

import os
import time
import json
from src.utils.config_utils import load_api_keys
from src.vlm_clients import gemini_client


# ============ 1️⃣ Load prompt & video paths ============
def load_all_prompts(scene_name: str):
    """Return a sorted list of (prompt_path, prompt_text) tuples for one scene."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    prompt_dir = os.path.join(root_dir, "generated_prompts", scene_name)

    print(f"🔍 Searching in: {prompt_dir}")
    if not os.path.exists(prompt_dir):
        raise FileNotFoundError(f"No prompt folder found for scene: {scene_name}")

    prompt_files = sorted([f for f in os.listdir(prompt_dir) if f.endswith(".txt")])
    if not prompt_files:
        raise FileNotFoundError(f"No prompt files found in {prompt_dir}")

    prompts = []
    for fname in prompt_files:
        fpath = os.path.join(prompt_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read().strip()
        prompts.append((fpath, text))
    return prompts


def get_video_path(scene_name: str):
    """Return path of the scene's video."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    video_path = os.path.join(root_dir, "videos", f"{scene_name}.mp4")
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"No video found for scene {scene_name}: {video_path}")
    return video_path


# ============ 2️⃣ Query Gemini with structured output ============
def safe_query_gemini(client, prompt_text, video_uri, model="gemini-2.5-flash",
                      max_retries=3, wait_retry=6, scene_name=None, prompt_file=None):
    """
    Run Gemini with schema-enforced JSON output.
    If schema not supported or transient error occurs, retry.
    """
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n🧠 Attempt {attempt}/{max_retries} using model [{model}] ...")
            result = gemini_client.query_gemini(
                client, prompt=prompt_text, video_uri=video_uri, model=model
            )

            raw_text = result["text"].strip()
            parsed = json.loads(raw_text)

            print("✅ Structured JSON parsed successfully.")
            print("🧾 Model Output:\n" + json.dumps(parsed, indent=2, ensure_ascii=False))

            # ✅ 打印 token 用量
            if result.get("usage"):
                usage = result["usage"]
                print(f"🧮 Token usage → prompt={usage.get('prompt_token_count', 0)}, "
                    f"output={usage.get('candidates_token_count', 0)}, "
                    f"total={usage.get('total_token_count', 0)}")


            return parsed

        except json.JSONDecodeError:
            print(f"⚠️ Output not valid JSON. Retrying in {wait_retry}s ...")
            time.sleep(wait_retry)
        except Exception as e:
            print(f"⚠️ Exception: {e}. Retrying in {wait_retry}s ...")
            time.sleep(wait_retry)

    print("❌ All retries failed. Logging failed prompt.")
    log_failed_prompt(scene_name, prompt_file, model)
    return {"error": f"Failed after {max_retries} retries"}

def log_failed_prompt(scene_name, prompt_file, model_name):
    """
    Append a record of failed prompt to results/failed_prompts_log.jsonl
    """
    log_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "results")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "failed_prompts_log.jsonl")

    entry = {
        "scene": scene_name,
        "prompt_file": prompt_file,
        "model": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"📄 Logged failed prompt → {log_path}")




# ============ 3️⃣ Run full scene ============
def run_scene(scene_name: str, model_name: str = "gemini-2.5-pro", rate_limit_delay: int = 35):
    """
    Run all prompts for one scene with structured output, token tracking, and rate limiting.
    Gemini free-tier allows ~2 requests per minute (≈1 every 30s).
    """
    # ========= 1️⃣ Load API key & client =========
    api_keys = load_api_keys()
    if "gemini" not in api_keys or "api_key" not in api_keys["gemini"]:
        raise KeyError("Missing Gemini API key in configs/api_keys.yaml")

    api_key = api_keys["gemini"]["api_key"]
    client = gemini_client.init_gemini(api_key)

    # ========= 2️⃣ Load all prompt & video files =========
    all_prompts = load_all_prompts(scene_name)
    video_path = get_video_path(scene_name)

    print(f"\n🎬 Scene: {scene_name}")
    print(f"🧩 Total prompts: {len(all_prompts)}")
    print(f"🧠 Using model: {model_name}")
    print(f"📹 Video: {os.path.basename(video_path)}")

    # ========= 3️⃣ Upload video once =========
    uploaded = gemini_client.upload_file(client, video_path)
    print(f"📤 Uploaded {uploaded.name}, waiting for activation...", end="", flush=True)
    while True:
        file_info = client.files.get(name=uploaded.name)
        if file_info.state.name == "ACTIVE":
            print(" ✅ ACTIVE")
            break
        elif file_info.state.name == "FAILED":
            raise RuntimeError("File upload failed.")
        print(".", end="", flush=True)
        time.sleep(3)

    # ========= 4️⃣ Initialize result & token counters =========
    all_results = []
    total_prompt_tokens = 0
    total_response_tokens = 0
    start_time = time.time()

    # ========= 5️⃣ Process each prompt independently =========
    for idx, (prompt_path, prompt_text) in enumerate(all_prompts, start=1):
        print(f"\n🚀 Running prompt {idx}/{len(all_prompts)}: {os.path.basename(prompt_path)}")
        print(f"📨 Sending prompt file: {os.path.basename(prompt_path)} (length {len(prompt_text)} chars)")

        try:
            # === Run query ===
            result = gemini_client.query_gemini(
                client,
                prompt=prompt_text,
                video_uri=uploaded.uri,
                model=model_name,
            )

            raw_text = result["text"].strip()
            parsed = json.loads(raw_text)

            print("✅ Structured JSON parsed successfully.")
            print("🧾 Model Output:\n" + json.dumps(parsed, indent=2, ensure_ascii=False))

            # === Token usage ===
            usage = result.get("usage")
            usage = result.get("usage")
            if usage:
                # Handle object or dict automatically
                prompt_toks = getattr(usage, "prompt_token_count", getattr(usage, "get", lambda k, d=None: d)("prompt_token_count", 0))
                resp_toks = getattr(usage, "candidates_token_count", getattr(usage, "get", lambda k, d=None: d)("candidates_token_count", 0))
                total_toks = getattr(usage, "total_token_count", getattr(usage, "get", lambda k, d=None: d)("total_token_count", 0))

                total_prompt_tokens += prompt_toks
                total_response_tokens += resp_toks
                print(f"🧮 Token usage → prompt={prompt_toks}, output={resp_toks}, total={total_toks}")


            # === Store per-prompt result ===
            all_results.append({
                "scene": scene_name,
                "prompt_file": os.path.basename(prompt_path),
                "model": model_name,
                "result": parsed,
                "usage": usage
            })

        except json.JSONDecodeError:
            print("⚠️ Output not valid JSON. Logging failure.")
            log_failed_prompt(scene_name, os.path.basename(prompt_path), model_name)
            all_results.append({
                "scene": scene_name,
                "prompt_file": os.path.basename(prompt_path),
                "model": model_name,
                "error": "Invalid JSON output"
            })
        except Exception as e:
            print(f"⚠️ Exception: {e}")
            log_failed_prompt(scene_name, os.path.basename(prompt_path), model_name)
            all_results.append({
                "scene": scene_name,
                "prompt_file": os.path.basename(prompt_path),
                "model": model_name,
                "error": str(e)
            })

        # === Enforce API rate limit ===
        if idx < len(all_prompts):
            elapsed = time.time() - start_time
            if elapsed < rate_limit_delay:
                wait_time = rate_limit_delay - elapsed
                print(f"⏳ Waiting {wait_time:.1f}s to respect API rate limit...")
                time.sleep(wait_time)
            start_time = time.time()

    # ========= 6️⃣ Save all results =========
    out_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scene_name}_{model_name}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    # ========= 7️⃣ Print token summary =========
    print(f"\n📊 Token usage summary for scene [{scene_name}] ({model_name}):")
    print(f"   Prompt tokens: {total_prompt_tokens}")
    print(f"   Response tokens: {total_response_tokens}")
    print(f"   Total tokens: {total_prompt_tokens + total_response_tokens}")

    print(f"\n✅ All {len(all_prompts)} prompts processed. Results saved to {out_path}")
    return all_results

# ============ 4️⃣ Direct Run ============
if __name__ == "__main__":
    # You can edit these values directly ↓↓↓
    SCENE_NAME = "HM3D00000"
    MODEL_NAME = "gemini-2.5-pro"     # e.g. "gemini-2.0-pro", "gemini-1.5-pro"
    RATE_LIMIT_DELAY = 35               # seconds between requests

    run_scene(SCENE_NAME, model_name=MODEL_NAME, rate_limit_delay=RATE_LIMIT_DELAY)