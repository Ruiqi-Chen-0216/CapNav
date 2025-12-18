import os
import re
import json
import time
import subprocess
from typing import List, Tuple, Dict, Any

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


# ============================================================
# Repo-relative defaults (open-source friendly)
# ============================================================

DEFAULT_PROMPT_ROOT = "generated_prompts"
DEFAULT_GRAPH_DIR = "ground_truth/graphs"
DEFAULT_VIDEO_ROOT = "videos_64frames_1fps"   # IMPORTANT: fixed for all open-source models
DEFAULT_RESULT_ROOT = "results"
DEFAULT_MODEL_CACHE_DIR = "models"

DEFAULT_ORG = "XiaomiMiMo"

# MiMo-VL uses /no_think for disabling reasoning
NO_THINK_SUFFIX = "/no_think"

# Video control: the source videos are fixed at 64 frames, 1 FPS
BASE_FPS = 1.0
TOTAL_FRAMES = 64

# Keep internal (do not expose to CLI)
DEFAULT_MAX_NEW_TOKENS = 8192
DEFAULT_DELAY = 0.0


# ============================================================
# Download helpers (bash on Unix, snapshot_download on Windows)
# ============================================================

def model_basename(model_id: str) -> str:
    return os.path.basename(model_id.rstrip("/"))


def normalize_hf_model_id(user_model: str) -> str:
    """
    Allows:
      - "MiMo-VL-7B-RL-2508" -> "XiaomiMiMo/MiMo-VL-7B-RL-2508"
      - "XiaomiMiMo/MiMo-VL-7B-RL-2508" -> keep as-is
    """
    if "/" in user_model:
        return user_model
    return f"{DEFAULT_ORG}/{user_model}"


def ensure_model_downloaded(hf_model_id: str, cache_dir: str = DEFAULT_MODEL_CACHE_DIR) -> str:
    """
    Ensure model exists under models/<MODEL_NAME>.
    - If exists, return local dir.
    - Else try:
        * Linux/macOS: bash scripts/download_model.sh <hf_id> <cache_dir>
        * Windows: huggingface_hub.snapshot_download -> <cache_dir>/<MODEL_NAME>
    - If all fail, return hf_model_id (transformers will use HF cache).
    """
    name = model_basename(hf_model_id)
    local_dir = os.path.join(cache_dir, name)
    if os.path.isdir(local_dir):
        return local_dir

    os.makedirs(cache_dir, exist_ok=True)

    script = os.path.join("scripts", "download_model.sh")
    is_windows = (os.name == "nt")

    # 1) Prefer portable shell downloader on Unix-like systems
    if os.path.exists(script) and (not is_windows):
        print(f"[AUTO-DOWNLOAD] {hf_model_id} -> {cache_dir} (via bash)")
        subprocess.check_call(["bash", script, hf_model_id, cache_dir])
        if os.path.isdir(local_dir):
            return local_dir

    # 2) Windows (or no bash): use huggingface_hub snapshot_download
    try:
        from huggingface_hub import snapshot_download
        print(f"[AUTO-DOWNLOAD] {hf_model_id} -> {local_dir} (via huggingface_hub)")
        snapshot_download(
            repo_id=hf_model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        if os.path.isdir(local_dir):
            return local_dir
    except Exception as e:
        print(f"[WARN] snapshot_download failed: {type(e).__name__}: {e}")

    # 3) Fallback: let transformers download to HF cache (requires internet)
    print("[FALLBACK] Using HF model id directly (transformers cache).")
    return hf_model_id


# ============================================================
# Scene/prompt/video helpers
# ============================================================

def detect_scenes_from_graphs(graph_dir: str) -> List[str]:
    scenes: List[str] = []
    for f in os.listdir(graph_dir):
        if f.endswith("-graph.json"):
            scenes.append(f.split("-graph.json")[0])
    scenes.sort()
    return scenes


def load_prompts(prompt_root: str, scene: str) -> List[Tuple[str, str]]:
    scene_dir = os.path.join(prompt_root, scene)
    if not os.path.isdir(scene_dir):
        raise FileNotFoundError(f"Prompt folder not found: {scene_dir}")

    files = sorted([f for f in os.listdir(scene_dir) if f.endswith(".txt")])
    out: List[Tuple[str, str]] = []
    for fname in files:
        with open(os.path.join(scene_dir, fname), "r", encoding="utf-8") as fp:
            out.append((fname, fp.read().strip()))
    return out


def get_video_path(video_root: str, scene: str) -> str:
    v = os.path.join(video_root, f"{scene}.mp4")
    if not os.path.exists(v):
        raise FileNotFoundError(f"Video not found: {v}")
    return v


# ============================================================
# Output parsing helpers (aligned with your previous behavior)
# ============================================================

def strip_think_block(text: str) -> str:
    """
    Remove <think>...</think> if present.
    """
    t = (text or "").strip()
    if "<think>" in t:
        t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()
    return t


def strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return t


def extract_json_candidate(text: str) -> str:
    """
    Robustly try to pull a JSON array from the text:
      - Strip code fences
      - Find the outermost [...] if present
      - Otherwise return trimmed text as-is
    """
    t = strip_code_fences(text)

    a = t.find("[")
    b = t.rfind("]")
    if a != -1 and b != -1 and b > a:
        return t[a:b + 1].strip()

    # fallback
    return t.strip()


def log_failure(fail_path: str, prompt_name: str, error_message: str, elapsed: float) -> None:
    record = {
        "prompt": prompt_name,
        "error": error_message,
        "time_sec": round(elapsed, 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(fail_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ============================================================
# MiMo init + single run
# ============================================================

def init_mimo(model_path: str):
    """
    MiMo-VL uses Qwen2.5-VL architecture in transformers.
    """
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def num_frames_to_fps(num_frames: int) -> float:
    """
    Videos are stored as 64 frames @ 1 FPS.
    To sample fewer frames from the same video, reduce fps proportionally:
      64 -> 1.0
      32 -> 0.5
      16 -> 0.25
    """
    if num_frames not in (16, 32, 64):
        raise ValueError("--num_frames must be one of {16, 32, 64}.")
    return float(BASE_FPS) * (float(num_frames) / float(TOTAL_FRAMES))


def maybe_append_no_think(prompt_text: str, thinking: str) -> str:
    """
    Strict equivalence to your old code:
      - thinking=off => enforce trailing ' /no_think'
      - thinking=on  => do not modify prompt
    """
    thinking_norm = thinking.lower().strip()
    if thinking_norm == "off":
        p = (prompt_text or "").strip()
        if not p.endswith(NO_THINK_SUFFIX):
            p = p + " " + NO_THINK_SUFFIX
        return p
    elif thinking_norm == "on":
        return (prompt_text or "").strip()
    else:
        raise ValueError("--thinking must be one of {on, off}.")


def run_single_prompt(
    model,
    processor,
    video_path: str,
    prompt_text: str,
    fps: float,
    max_new_tokens: int,
) -> str:
    conversation = [{
        "role": "user",
        "content": [
            {"type": "video", "path": os.path.abspath(video_path)},
            {"type": "text", "text": prompt_text},
        ],
    }]

    inputs = processor.apply_chat_template(
        conversation,
        fps=fps,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            top_k=20,
            temperature=0.3,
            repetition_penalty=1.0,
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    text = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()
    return text


# ============================================================
# Public entry point (adapter API)
# ============================================================

def run_mimo_vl(user_model: str, num_frames: int, thinking: str) -> None:
    """
    Public entry point:
      user_model: "MiMo-VL-7B-RL-2508" or "XiaomiMiMo/MiMo-VL-7B-RL-2508"
      num_frames: 16 / 32 / 64  (controls fps sampling from the SAME video)
      thinking:   on / off      (on: strip <think>; off: append /no_think)
    """
    hf_model_id = normalize_hf_model_id(user_model)
    model_name = model_basename(hf_model_id)

    prompt_root = DEFAULT_PROMPT_ROOT
    graph_dir = DEFAULT_GRAPH_DIR
    video_root = DEFAULT_VIDEO_ROOT
    result_root = DEFAULT_RESULT_ROOT
    cache_dir = DEFAULT_MODEL_CACHE_DIR

    # Required paths
    for p in [prompt_root, graph_dir, video_root]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required path missing: {p}")

    model_path = ensure_model_downloaded(hf_model_id, cache_dir=cache_dir)
    print(f"[MODEL] using: {model_path}")

    model, processor = init_mimo(model_path)

    fps = num_frames_to_fps(num_frames)
    print(f"[VIDEO] total_frames={TOTAL_FRAMES} base_fps={BASE_FPS} num_frames={num_frames} -> fps={fps}")

    scenes = detect_scenes_from_graphs(graph_dir)
    print(f"[SCENES] detected {len(scenes)} scenes")

    thinking_tag = thinking.lower().strip()
    if thinking_tag not in ("on", "off"):
        raise ValueError("--thinking must be one of {on, off}.")

    for scene in scenes:
        prompts = load_prompts(prompt_root, scene)
        video_path = get_video_path(video_root, scene)

        out_dir = os.path.join(result_root, f"{model_name}_{num_frames}frames_{thinking_tag}", scene)
        os.makedirs(out_dir, exist_ok=True)
        fail_path = os.path.join(out_dir, "failed_prompts.jsonl")

        print(f"\n[SCENE] {scene} | prompts={len(prompts)}")

        for fname, ptext in prompts:
            prompt_stem = os.path.splitext(fname)[0]
            out_file = os.path.join(out_dir, f"{prompt_stem}.json")
            if os.path.exists(out_file):
                continue

            t0 = time.time()
            entry: Dict[str, Any] = {
                "scene": scene,
                "prompt_file": fname,
                "model": model_name,
                "hf_model_id": hf_model_id,
                "num_frames": num_frames,
                "fps": fps,
                "thinking": thinking_tag,
            }

            try:
                prompt_used = maybe_append_no_think(ptext, thinking=thinking)
                raw_text = run_single_prompt(
                    model=model,
                    processor=processor,
                    video_path=video_path,
                    prompt_text=prompt_used,
                    fps=fps,
                    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                )

                entry["raw_text"] = raw_text
                entry["time_sec"] = round(time.time() - t0, 2)

                # Strict equivalence:
                # - thinking=on: strip <think>...</think>
                # - thinking=off: DO NOT strip (model is instructed via /no_think)
                text_for_parse = raw_text
                if thinking_tag == "on":
                    text_for_parse = strip_think_block(text_for_parse)

                json_candidate = extract_json_candidate(text_for_parse)

                try:
                    entry["result"] = json.loads(json_candidate)
                except Exception:
                    entry["result"] = None
                    entry["parse_error"] = True
                    entry["json_str"] = json_candidate[:20000]
                    log_failure(fail_path, prompt_stem, "JSONParseError", time.time() - t0)

                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(entry, f, indent=2, ensure_ascii=False)

            except Exception as e:
                entry["time_sec"] = round(time.time() - t0, 2)
                entry["error"] = repr(e)
                log_failure(fail_path, prompt_stem, repr(e), time.time() - t0)

                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(entry, f, indent=2, ensure_ascii=False)

                torch.cuda.empty_cache()

            if DEFAULT_DELAY > 0:
                time.sleep(DEFAULT_DELAY)
