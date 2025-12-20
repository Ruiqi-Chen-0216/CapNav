# src/model_adapters/spatial_mllm_adapter.py

import os
import re
import sys
import json
import time
import subprocess
from typing import List, Tuple, Dict, Any, Optional

import torch


# ============================================================
# Repo-relative defaults (open-source friendly)
# ============================================================

DEFAULT_PROMPT_ROOT = "generated_prompts"
DEFAULT_GRAPH_DIR = "ground_truth/graphs"
DEFAULT_VIDEO_ROOT = "videos_64frames_1fps"   
DEFAULT_RESULT_ROOT = "results"
DEFAULT_MODEL_CACHE_DIR = "models"

# Spatial-MLLM repo location (for importing its src/)
DEFAULT_SPATIAL_MLLM_ROOT = os.environ.get("SPATIAL_MLLM_ROOT", "/scr/Spatial-MLLM")


# ============================================================
# Download helpers (same style as InternVL)
# ============================================================

def model_basename(model_id: str) -> str:
    return os.path.basename(model_id.rstrip("/"))


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
        print(f"[WARN] Prompt folder not found: {scene_dir}")
        return []

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
# Spatial-MLLM import bootstrap
# ============================================================

def _ensure_spatial_mllm_src_on_path(spatial_mllm_root: str = DEFAULT_SPATIAL_MLLM_ROOT) -> None:
    """
    Spatial-MLLM's code expects imports from its own `src/`.
    We add <SPATIAL_MLLM_ROOT>/src to sys.path if present.
    """
    src_dir = os.path.join(spatial_mllm_root, "src")
    if os.path.isdir(src_dir) and (src_dir not in sys.path):
        sys.path.insert(0, src_dir)
        print(f"[PATH] Added Spatial-MLLM src to sys.path: {src_dir}")


# ============================================================
# Model init + single prompt runner (ported from your script)
# ============================================================

def init_spatial_mllm(model_path: str, device: str = "cuda"):
    """
    NOTE: Spatial-MLLM uses its own model classes under Spatial-MLLM/src.
    """
    _ensure_spatial_mllm_src_on_path()

    from models import Qwen2_5_VL_VGGTForConditionalGeneration, Qwen2_5_VLProcessor  # type: ignore
    from qwen_vl_utils import process_vision_info  # type: ignore

    print(f"[MODEL] Loading Spatial-MLLM from: {model_path}")
    model = Qwen2_5_VL_VGGTForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    ).to(device)
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path)
    return model, processor, process_vision_info


def run_single_prompt(
    model,
    processor,
    process_vision_info_fn,
    video_path: str,
    text: str,
    num_frames: int,
    device: str = "cuda",
) -> Tuple[str, float]:
    """
    Spatial-MLLM inference.
    IMPORTANT:
      - video path is passed as-is
      - num_frames is mapped to message field "nframes"
    """
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "nframes": int(num_frames)},
                {"type": "text", "text": text},
            ],
        }
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    _, video_inputs = process_vision_info_fn(messages)

    inputs = processor(
        text=[text_input],
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs.update({"videos_input": torch.stack(video_inputs) / 255.0})
    inputs = inputs.to(device)

    torch.cuda.empty_cache()
    t0 = time.time()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=8196,        
            do_sample=True,
            temperature=0.1,
            top_p=0.001,
            use_cache=True,
        )
    elapsed = time.time() - t0

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return (output_texts[0], elapsed)


# ============================================================
# Output parsing (STRICTLY follows your <json>...</json> + fallback)
# ============================================================

def parse_spatial_mllm_output(raw_text: str) -> Dict[str, Any]:
    """
    STRICT equivalence to your script:
      - require <json>...</json> block
      - regex-extract answer/reason/path
      - else fallback
    """
    fallback = {"answer": "failed to answer the question", "path": []}

    match = re.search(r"<json>([\s\S]*?)</json>", raw_text or "")
    if not match:
        return dict(fallback)

    content = match.group(1).strip()

    ans_match = re.search(r'"answer"\s*:\s*"([^"]+)"', content)
    answer = ans_match.group(1).strip() if ans_match else fallback["answer"]

    reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', content)
    reason = reason_match.group(1).strip() if reason_match else None

    path_match = re.search(r'"path"\s*:\s*\[([^\]]*)\]', content)
    if path_match:
        path_items = [
            item.strip().strip('"').strip("'")
            for item in path_match.group(1).split(",")
            if item.strip()
        ]
    else:
        path_items = fallback["path"]

    out: Dict[str, Any] = {"answer": answer, "path": path_items}
    if reason:
        out["reason"] = reason
    return out


# ============================================================
# Public entry point (adapter API)
# ============================================================

def run_spatial_mllm(user_model: str, num_frames: int) -> None:
    """
    Public entry point for run.py routing.
    Inputs:
      - user_model: HF id or local dir
      - num_frames: 16 / 32 / 64 (controls nframes for Spatial-MLLM)
    NOTE:
      - thinking is implicitly ON for Spatial-MLLM; not exposed.
    """
    if num_frames not in (16, 32, 64):
        raise ValueError("--num_frames must be one of {16, 32, 64}.")

    prompt_root = DEFAULT_PROMPT_ROOT
    graph_dir = DEFAULT_GRAPH_DIR
    video_root = DEFAULT_VIDEO_ROOT
    result_root = DEFAULT_RESULT_ROOT
    cache_dir = DEFAULT_MODEL_CACHE_DIR

    for p in [prompt_root, graph_dir, video_root]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Required path missing: {p}")

    # Download model if needed (same behavior as InternVL)
    model_path = ensure_model_downloaded(user_model, cache_dir=cache_dir)
    model_name = model_basename(user_model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("Spatial-MLLM adapter requires CUDA in your current setup.")

    model, processor, process_vision_info_fn = init_spatial_mllm(model_path, device=device)
    scenes = detect_scenes_from_graphs(graph_dir)
    print(f"[SCENES] detected {len(scenes)} scenes")

    # Output tag: keep stable + explicit
    model_tag = f"{model_name}_{num_frames}frames_thinking_on"

    for scene in scenes:
        prompts = load_prompts(prompt_root, scene)
        if not prompts:
            continue

        video_path = get_video_path(video_root, scene)

        out_dir = os.path.join(result_root, "spatial_mllm", model_tag, scene)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[SCENE] {scene} | prompts={len(prompts)} | out={out_dir}")

        results: List[Dict[str, Any]] = []

        for i, (fname, ptext) in enumerate(prompts, 1):
            stem = os.path.splitext(fname)[0]
            out_file = os.path.join(out_dir, f"{stem}.json")

            if os.path.exists(out_file):
                with open(out_file, "r", encoding="utf-8") as f:
                    results.append(json.load(f))
                continue

            t0 = time.time()
            entry: Dict[str, Any] = {
                "scene": scene,
                "prompt_file": fname,
                "model": model_name,
                "user_model": user_model,
                "num_frames": num_frames,
                "thinking": "on",
            }

            try:
                raw_text, elapsed = run_single_prompt(
                    model=model,
                    processor=processor,
                    process_vision_info_fn=process_vision_info_fn,
                    video_path=video_path,
                    text=ptext,
                    num_frames=num_frames,
                    device=device,
                )
                entry["raw_text"] = raw_text
                entry["time_sec"] = round(time.time() - t0, 2)
                entry["elapsed_sec_model"] = round(elapsed, 2)

                entry["result"] = parse_spatial_mllm_output(raw_text)

            except Exception as e:
                entry["time_sec"] = round(time.time() - t0, 2)
                entry["error"] = repr(e)
                entry["raw_text"] = ""
                entry["result"] = {"answer": "failed to answer the question", "path": []}
                torch.cuda.empty_cache()

            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)
            results.append(entry)

        merged = os.path.join(out_dir, f"{scene}_results.json")
        with open(merged, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[DONE] Scene completed → {merged}")
