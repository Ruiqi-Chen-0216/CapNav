#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CapNav Orchestrator (GPT-5-pro Text Mode — Image-Frames, Windows paths)
----------------------------------------------------------------------
✅ 特性：
- 每个场景仅上传一次帧图片
- 均匀采样 64 / 32 / 16 帧
- 不使用结构化输出（非 Pydantic 模式）
- 支持 reasoning_effort: low / medium / high
- 断点续跑（只重跑缺失的 prompt）
- 输出路径: results/{model}_{Nframes}frames/{scene}/
"""

import os
import json
import time
from typing import List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from src.utils.config_utils import load_api_keys
from src.vlm_clients import init_gpt, upload_file, query_gpt


# ===============================
# 0️⃣ 通用：超时包装器
# ===============================
def call_with_timeout(func, timeout_seconds: int, **kwargs):
    """在单线程池中以超时方式调用 func(**kwargs)。抛出 FuturesTimeoutError 表示超时。"""
    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(func, **kwargs)
        return fut.result(timeout=timeout_seconds)


# ======================================================
# 1️⃣ 工具函数
# ======================================================
def detect_all_scenes(graph_dir: str) -> List[str]:
    """自动检测 graph 文件夹下的所有场景"""
    scenes = [f.split("-graph.json")[0] for f in os.listdir(graph_dir) if f.endswith("-graph.json")]
    scenes.sort()
    print(f"📦 Detected {len(scenes)} scenes in {graph_dir}")
    return scenes


def load_prompts(scene_name: str, prompt_root: str) -> List[Tuple[str, str]]:
    """加载某个场景的全部 prompt"""
    prompt_dir = os.path.join(prompt_root, scene_name)
    if not os.path.exists(prompt_dir):
        raise FileNotFoundError(f"No prompt folder found: {prompt_dir}")
    prompt_files = sorted([f for f in os.listdir(prompt_dir) if f.endswith(".txt")])
    if not prompt_files:
        raise FileNotFoundError(f"No prompt files in {prompt_dir}")
    prompts = []
    for fname in prompt_files:
        with open(os.path.join(prompt_dir, fname), "r", encoding="utf-8") as f:
            prompts.append((fname, f.read().strip()))
    return prompts


def _evenly_pick_indices(total: int, target: int) -> List[int]:
    """在 [0, total-1] 上均匀取 target 个索引"""
    if target <= 1:
        return [0]
    if target >= total:
        return list(range(total))
    step = (total - 1) / (target - 1)
    return [round(i * step) for i in range(target)]


def select_frame_files(frame_dir: str,
                       frames_target: Optional[int],
                       fps_ratio: Optional[float]) -> List[str]:
    """从帧目录选取需要上传的帧文件（均匀采样）"""
    all_imgs = sorted([
        f for f in os.listdir(frame_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg")) and not f.startswith("._")
    ])
    if not all_imgs:
        raise RuntimeError(f"No valid images in {frame_dir}")

    N = len(all_imgs)
    if frames_target is not None:
        target = max(1, min(frames_target, N))
    else:
        ratio = 1.0 if fps_ratio is None else max(0.0, min(fps_ratio, 1.0))
        target = max(1, min(N, int(round(N * ratio))))

    idxs = _evenly_pick_indices(N, target)
    picked = [all_imgs[i] for i in idxs]
    return picked


def upload_scene_images(client,
                        scene_name: str,
                        frame_root: str,
                        frames_target: Optional[int] = None,
                        fps_ratio: Optional[float] = None,
                        max_images: int = 500) -> Tuple[List[str], int]:
    """均匀抽帧上传：返回 (file_ids, used_frames_count)"""
    frame_dir = os.path.join(frame_root, scene_name)
    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"No image folder found for scene: {frame_dir}")

    picked_files = select_frame_files(frame_dir, frames_target, fps_ratio)
    if len(picked_files) > max_images:
        picked_files = picked_files[:max_images]

    print(f"🖼️ Uploading {len(picked_files)} images from {frame_dir}")
    file_ids = []
    for i, fname in enumerate(picked_files, 1):
        path = os.path.join(frame_dir, fname)
        file_obj = upload_file(client, path, purpose="vision", wait_until_ready=True)
        file_ids.append(file_obj.id)
        if i % 50 == 0 or i == len(picked_files):
            print(f"📤 Uploaded {i}/{len(picked_files)} frames ...")
    print(f"✅ Done uploading {len(picked_files)} frames.")
    return file_ids, len(picked_files)


# ======================================================
# 2️⃣ 主函数：运行单场景
# ======================================================
def run_scene_text(scene_name: str,
                   client,
                   model_name: str,
                   prompt_root: str,
                   result_root: str,
                   frame_root: str,
                   reason_level: str = "medium",   # low / medium / high
                   rate_limit_delay: int = 0,
                   per_prompt_timeout: int = 600,
                   frames_target: Optional[int] = None,
                   fps_ratio: Optional[float] = None):
    """运行一个场景下的所有 prompts（帧图片作为视觉输入）"""

    # === Reason level → reasoning_effort 映射 ===
    effort_map = {"low": "minimal", "medium": "medium", "high": "high"}
    reasoning_effort = effort_map.get(reason_level, "medium")

    # === 上传帧 ===
    file_ids, used_frames = upload_scene_images(
        client,
        scene_name=scene_name,
        frame_root=frame_root,
        frames_target=frames_target,
        fps_ratio=fps_ratio,
    )

    # === 输出路径 ===
    model_tag = f"{model_name}_{used_frames}frames_text"
    out_dir = os.path.join(result_root, model_tag, scene_name)
    os.makedirs(out_dir, exist_ok=True)

    # === 加载 prompts ===
    prompts = load_prompts(scene_name, prompt_root)

    print(f"\n🎬 Scene: {scene_name}")
    print(f"🧩 Prompts: {len(prompts)}")
    print(f"🎞️ Frames used: {used_frames}")
    print(f"🧠 Reasoning effort: {reasoning_effort}")
    print(f"📂 Output directory: {out_dir}")

    summary = []
    failed_list = []
    last_tick = time.time()

    for idx, (prompt_file, prompt_text) in enumerate(prompts, start=1):
        prompt_name = os.path.splitext(prompt_file)[0]
        out_file = os.path.join(out_dir, f"{prompt_name}.json")

        # ✅ 跳过已完成
        if os.path.exists(out_file):
            print(f"⏭️ Skipping {prompt_name} (already done).")
            with open(out_file, "r", encoding="utf-8") as f:
                summary.append(json.load(f))
            continue

        print(f"\n🚀 Prompt {idx}/{len(prompts)} → {prompt_file}")
        t0 = time.time()
        entry = {
            "scene": scene_name,
            "prompt_file": prompt_file,
            "model": model_name,
            "reason_level": reason_level,
        }

        try:
            # === 带超时的文本调用 ===
            result = call_with_timeout(
                query_gpt,
                timeout_seconds=per_prompt_timeout,
                client=client,
                prompt=prompt_text,
                model=model_name,
                image_file_ids=file_ids,
                output_model=None,  # ✅ 非结构化模式
                reasoning_effort=reasoning_effort,
            )

            text_output = result.get("text")
            usage = result.get("usage")

            if text_output:
                entry["result"] = {"raw_text": text_output.strip()}
            else:
                print("⚠️ No text output — logging as failed (no save).")
                failed_list.append({
                    "scene": scene_name,
                    "prompt_file": prompt_file,
                    "error": "empty_output"
                })
                continue

            if usage:
                entry["usage"] = {
                    "input_tokens": getattr(usage, "input_tokens", None),
                    "output_tokens": getattr(usage, "output_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }

            entry["time_sec"] = round(time.time() - t0, 2)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved → {out_file}")

            summary.append(entry)

        except FuturesTimeoutError:
            # ✅ 超时：写入 fallback 占位
            print(f"⏰ Timeout (> {per_prompt_timeout}s). Marking as failed.")
            entry["result"] = {
                "raw_text": "failed to answer the question (timeout)"
            }
            entry["time_sec"] = round(time.time() - t0, 2)
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)
            print(f"💾 Saved (timeout fallback) → {out_file}")
            summary.append(entry)

        except Exception as e:
            err_msg = str(e)
            print(f"❌ Exception: {err_msg}")
            failed_list.append({
                "scene": scene_name,
                "prompt_file": prompt_file,
                "error": err_msg
            })
            continue

        # 限速控制
        if idx < len(prompts):
            elapsed = time.time() - last_tick
            if elapsed < rate_limit_delay:
                wait_t = rate_limit_delay - elapsed
                print(f"⏳ Waiting {wait_t:.1f}s ...")
                time.sleep(wait_t)
            last_tick = time.time()

    # ✅ 场景级汇总
    merged = os.path.join(out_dir, f"{scene_name}_results.json")
    with open(merged, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Scene completed → {merged}")

    if failed_list:
        failed_path = os.path.join(out_dir, f"{scene_name}_failed.json")
        with open(failed_path, "w", encoding="utf-8") as f:
            json.dump(failed_list, f, indent=2, ensure_ascii=False)
        print(f"⚠️ {len(failed_list)} prompts failed → {failed_path}")


# ======================================================
# 3️⃣ 主入口（Windows 路径）
# ======================================================
if __name__ == "__main__":
    GRAPH_DIR  = r"C:\Users\pendr\Desktop\CapNav\graph"
    PROMPT_DIR = r"C:\Users\pendr\Desktop\CapNav\generated_prompts"
    RESULT_ROOT = r"C:\Users\pendr\Desktop\CapNav\results"
    FRAME_ROOT  = r"C:\Users\pendr\Desktop\CapNav\videos_gpt"

    MODEL_NAME = "gpt-5-pro"     # ✅ 专为 gpt-5-pro 设计
    REASON_LEVEL = "high"        # low / medium / high
    RATE_LIMIT_DELAY = 0
    PER_PROMPT_TIMEOUT = 600

    FRAMES_TARGET = 32           # e.g. 64 / 32 / 16
    FPS_RATIO = None             # 若 FRAMES_TARGET=None 时生效

    # === 初始化 API ===
    api_keys = load_api_keys()
    client = init_gpt(api_keys["openai"]["api_key"])

    # === 自动检测场景 ===
    scene_list = detect_all_scenes(GRAPH_DIR)
    SCENE_LIST = [s for s in scene_list if s.startswith("MP")]
    print(f"📂 Running {len(SCENE_LIST)} MP scenes:")
    print("   " + ", ".join(SCENE_LIST))
    for scene in SCENE_LIST:
        run_scene_text(
            scene_name=scene,
            client=client,
            model_name=MODEL_NAME,
            prompt_root=PROMPT_DIR,
            result_root=RESULT_ROOT,
            frame_root=FRAME_ROOT,
            reason_level=REASON_LEVEL,
            rate_limit_delay=RATE_LIMIT_DELAY,
            per_prompt_timeout=PER_PROMPT_TIMEOUT,
            frames_target=FRAMES_TARGET,
            fps_ratio=FPS_RATIO,
        )
