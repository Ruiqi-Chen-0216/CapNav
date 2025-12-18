import os
import re
import json
import time
import subprocess
from typing import List, Tuple, Dict, Any

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# ============================================================
# Repo-relative defaults (open-source friendly)
# ============================================================

DEFAULT_PROMPT_ROOT = "generated_prompts"
DEFAULT_GRAPH_DIR = "ground_truth/graphs"
DEFAULT_VIDEO_ROOT = "videos_64frames_1fps"   # IMPORTANT: fixed for all open-source models
DEFAULT_RESULT_ROOT = "results"
DEFAULT_MODEL_CACHE_DIR = "models"

DEFAULT_ORG = "Qwen"

# Video folder semantics in this repo:
BASE_FPS = 1.0
TOTAL_FRAMES = 64  # videos_64frames_1fps

# Strict token detection (segment match, not suffix match)
_THINKING_TOKEN = re.compile(r"(?:^|-)Thinking(?:-|$)")
_INSTRUCT_TOKEN = re.compile(r"(?:^|-)Instruct(?:-|$)")


# ============================================================
# Download helpers (bash on Unix, snapshot_download on Windows)
# ============================================================

def model_basename(model_id: str) -> str:
    return os.path.basename(model_id.rstrip("/"))


def normalize_hf_model_id(user_model: str) -> str:
    """
    Allows:
      - "Qwen3-VL-30B-A3B-Thinking-FP8" -> "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8"
      - "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8" -> keep as-is
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

    if os.path.exists(script) and (not is_windows):
        print(f"[AUTO-DOWNLOAD] {hf_model_id} -> {cache_dir} (via bash)")
        subprocess.check_call(["bash", script, hf_model_id, cache_dir])
        if os.path.isdir(local_dir):
            return local_dir

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
# Qwen3-VL think/no-think capability inference (STRICT)
# ============================================================

def infer_required_thinking_from_checkpoint(hf_model_id: str) -> str:
    """
    Returns:
      - "on"  if checkpoint name contains '-Thinking-' segment
      - "off" if checkpoint name contains '-Instruct-' segment
    Raises if neither / both found.
    """
    name = model_basename(hf_model_id)

    has_thinking = _THINKING_TOKEN.search(name) is not None
    has_instruct = _INSTRUCT_TOKEN.search(name) is not None

    if has_thinking and has_instruct:
        raise ValueError(
            f"Ambiguous Qwen3-VL checkpoint name (contains both Thinking and Instruct): {hf_model_id}"
        )
    if has_thinking:
        return "on"
    if has_instruct:
        return "off"

    raise ValueError(
        "Qwen3-VL checkpoint name must contain a '-Thinking-' or '-Instruct-' segment "
        f"to determine think mode. Got: {hf_model_id}"
    )


# ============================================================
# Output parsing helpers (keep consistent with other adapters)
# ============================================================

def strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = t.strip("`").strip()
        if t.lower().startswith("json"):
            t = t[4:].strip()
    return t


def extract_json_candidate(text: str) -> str:
    t = strip_code_fences(text)

    # trim leading text before first { or [
    start_candidates = [i for i in [t.find("["), t.find("{")] if i != -1]
    if start_candidates:
        start = min(start_candidates)
        if start > 0:
            t = t[start:].strip()

    a = t.find("[")
    b = t.rfind("]")
    if a != -1 and b != -1 and b > a:
        return t[a:b + 1].strip()

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
# Model init + single run
# ============================================================

def init_qwen3_vl(model_path: str):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor


def generation_params_for_mode(required_thinking: str) -> Dict[str, Any]:
    """
    You required:
      - no-think (Instruct): top_p=0.8, top_k=20, temperature=0.7
      - think (Thinking):   top_p=0.95, top_k=20, temperature=1.0
    And unify max_new_tokens=8196.
    """
    if required_thinking == "off":
        return {"top_p": 0.8, "top_k": 20, "temperature": 0.7}
    return {"top_p": 0.95, "top_k": 20, "temperature": 1.0}


def run_one(model, processor, video_path: str, prompt_text: str, fps: float, max_new_tokens: int,
            required_thinking: str) -> Dict[str, Any]:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": os.path.abspath(video_path),
                    "fps": fps,
                    "min_pixels": 4 * 32 * 32,
                    "max_pixels": 256 * 32 * 32,
                },
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    gen_kw = generation_params_for_mode(required_thinking)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=gen_kw["top_p"],
            top_k=gen_kw["top_k"],
            temperature=gen_kw["temperature"],
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, outputs)]
    raw_text = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()

    # Strict equivalence:
    # - If checkpoint is Thinking, strip <think> if present
    # - If checkpoint is Instruct, do NOT attempt to strip unless it unexpectedly appears
    cleaned = raw_text
    json_str = extract_json_candidate(cleaned)
    return {"raw_text": raw_text, "json_str": json_str}


# ============================================================
# Public entry point
# ============================================================

def run_qwen3_vl(user_model: str, num_frames: int, thinking: str) -> None:
    """
    Public entry point:
      user_model: "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8" OR "Qwen3-VL-30B-A3B-Thinking-FP8"
      num_frames: 16 / 32 / 64  (sampling count from the SAME video under videos_64frames_1fps)
      thinking:   on / off  (MUST match checkpoint type)
    """
    thinking_norm = thinking.lower().strip()
    if thinking_norm not in {"on", "off"}:
        raise ValueError('--thinking must be exactly "on" or "off".')

    hf_model_id = normalize_hf_model_id(user_model)
    model_name = model_basename(hf_model_id)

    # Determine required mode from checkpoint naming
    required_thinking = infer_required_thinking_from_checkpoint(hf_model_id)

    # Enforce user-provided thinking matches checkpoint
    if thinking_norm != required_thinking:
        if required_thinking == "on":
            raise ValueError(
                f"Invalid --thinking for {hf_model_id}.\n"
                "This is a Thinking checkpoint, it only supports: --thinking on"
            )
        raise ValueError(
            f"Invalid --thinking for {hf_model_id}.\n"
            "This is an Instruct checkpoint, it only supports: --thinking off"
        )

    # Repo-relative defaults
    prompt_root = DEFAULT_PROMPT_ROOT
    graph_dir = DEFAULT_GRAPH_DIR
    video_root = DEFAULT_VIDEO_ROOT
    result_root = DEFAULT_RESULT_ROOT
    cache_dir = DEFAULT_MODEL_CACHE_DIR

    for p in [prompt_root, graph_dir, video_root]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required path missing: {p}")

    model_path = ensure_model_downloaded(hf_model_id, cache_dir=cache_dir)
    print(f"[MODEL] using: {model_path}")

    model, processor = init_qwen3_vl(model_path)

    # IMPORTANT: videos are fixed 64 frames @ 1fps in repo
    # We emulate your original fps-based behavior:
    #   frames = 64 * fps  => fps = num_frames / 64
    fps = float(BASE_FPS) * (float(num_frames) / float(TOTAL_FRAMES))
    print(f"[VIDEO] base_fps={BASE_FPS} total_frames={TOTAL_FRAMES} num_frames={num_frames} -> effective_fps={fps:.4f}")

    max_new_tokens = 8196
    scenes = detect_scenes_from_graphs(graph_dir)
    print(f"[SCENES] detected {len(scenes)} scenes")

    for scene in scenes:
        prompts = load_prompts(prompt_root, scene)
        video_path = get_video_path(video_root, scene)

        out_dir = os.path.join(result_root, f"{model_name}_{num_frames}frames_{thinking_norm}", scene)
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
                "thinking": thinking_norm,
                "effective_fps": fps,
            }

            try:
                out = run_one(
                    model, processor,
                    video_path=video_path,
                    prompt_text=ptext,
                    fps=fps,
                    max_new_tokens=max_new_tokens,
                    required_thinking=required_thinking,
                )

                entry["raw_text"] = out["raw_text"]
                entry["time_sec"] = round(time.time() - t0, 2)

                try:
                    entry["result"] = json.loads(out["json_str"])
                except Exception:
                    entry["result"] = None
                    entry["parse_error"] = True
                    entry["json_str"] = out["json_str"][:20000]
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
