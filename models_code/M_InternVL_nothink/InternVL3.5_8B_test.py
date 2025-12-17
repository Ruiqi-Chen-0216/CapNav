#python /gscratch/makelab/ruiqi/M_InternVL/InternVL3.5_8B_test.py 
import math
import os
import json
import time
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# ===== 模型加载 =====
path = '/scr/models/InternVL3_5-8B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
generation_config = dict(max_new_tokens=1024, do_sample=True)

# ===== 视频多轮问答部分（保留单轮） =====
def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    max_frame = len(vr) - 1
    fps = float(vr.get_avg_fps())

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
        img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in img]
        pixel_values = torch.stack(pixel_values)
        num_patches_list.append(pixel_values.shape[0])
        pixel_values_list.append(pixel_values)
    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list


# ======================================================
# 🧩 新增：Prompt 批量读取与运行逻辑（仿 Qwen 脚本）
# ======================================================
def load_prompts(scene_name: str):
    """读取指定场景的 prompt 文件"""
    base = "/gscratch/makelab/ruiqi/generated_prompts"
    prompt_dir = os.path.join(base, scene_name)
    if not os.path.exists(prompt_dir):
        raise FileNotFoundError(f"No prompt folder found: {prompt_dir}")
    files = sorted([f for f in os.listdir(prompt_dir) if f.endswith(".txt")])
    if not files:
        raise FileNotFoundError(f"No prompt files in {prompt_dir}")
    prompts = []
    for fname in files:
        fpath = os.path.join(prompt_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            prompts.append((fname, f.read().strip()))
    return prompts


def run_scene(scene_name, video_path, model, tokenizer,
              num_segments=16,
              out_dir="/gscratch/makelab/ruiqi/results/InternVL3.5"):
    """对单个视频运行所有 prompt 并统一输出格式"""
    os.makedirs(out_dir, exist_ok=True)
    scene_out_dir = os.path.join(out_dir, scene_name)
    os.makedirs(scene_out_dir, exist_ok=True)

    prompts = load_prompts(scene_name)
    print(f"\n🎬 Scene: {scene_name}")
    print(f"📹 Video: {os.path.basename(video_path)}")
    print(f"🧩 Total prompts: {len(prompts)}")
    print(f"📂 Output directory: {scene_out_dir}\n")

    # ===== 视频帧预处理只执行一次 =====
    pixel_values, num_patches_list = load_video(video_path, num_segments=num_segments, max_num=1)
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])

    for i, (prompt_file, prompt_text) in enumerate(prompts, 1):
        print(f"🚀 Running {i}/{len(prompts)} → {prompt_file}")
        question = video_prefix + "\n" + prompt_text

        # 输出文件路径
        out_path = os.path.join(scene_out_dir, f"{os.path.splitext(prompt_file)[0]}.json")
        if os.path.exists(out_path):
            print(f"⏭️ Skipping {prompt_file} (already exists).")
            continue

        entry = {"scene": scene_name, "prompt_file": prompt_file}
        t0 = time.time()

        try:
            response, _ = model.chat(
                tokenizer,
                pixel_values,
                question,
                generation_config,
                num_patches_list=num_patches_list,
                history=None,
                return_history=True
            )

            text = response.strip()

            # ======== 🔍 清洗模型输出 ========
            # 去除 markdown 代码块
            if text.startswith("```"):
                text = text.strip("`")
                if text.lower().startswith("json"):
                    text = text[4:].strip()

            # 截取第一个 JSON 数组
            start_idx = min(
                [i for i in [text.find("["), text.find("{")] if i != -1],
                default=-1
            )
            if start_idx > 0:
                text = text[start_idx:].strip()

            # ======== 🧩 尝试解析为 JSON ========
            try:
                parsed = json.loads(text)
                entry["result"] = parsed
                print("✅ Parsed JSON successfully.")
            except json.JSONDecodeError:
                entry["raw_output"] = text
                print("⚠️ Could not parse JSON; saved raw text.")

        except Exception as e:
            entry["error"] = str(e)
            print(f"❌ Error in {prompt_file}: {e}")

        # ======== 记录耗时 ========
        entry["time_sec"] = round(time.time() - t0, 2)

        # ======== 保存 ========
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved → {out_path}\n")


# ======================================================
# 主入口
# ======================================================
if __name__ == "__main__":
    scene_name = "HM3D00000test"
    video_path = f"/gscratch/makelab/ruiqi/videos_64frames_1fps/{scene_name}.mp4"
    run_scene(scene_name, video_path, model, tokenizer, num_segments=32)
