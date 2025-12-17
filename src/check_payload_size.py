import os

def check_payload(scene_name: str, root="videos_gpt", limit_mb: float = 50.0):
    """
    Check if total image payload size exceeds GPT's 50MB per request limit.
    """
    frame_dir = os.path.join(root, scene_name)
    if not os.path.exists(frame_dir):
        raise FileNotFoundError(f"❌ Folder not found: {frame_dir}")

    total_bytes = 0
    files = [
        f for f in os.listdir(frame_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg")) and not f.startswith("._")
    ]

    if not files:
        print(f"⚠️ No valid image files found in {frame_dir}")
        return

    for f in files:
        path = os.path.join(frame_dir, f)
        total_bytes += os.path.getsize(path)

    total_mb = total_bytes / (1024 * 1024)
    print(f"🧾 Scene: {scene_name}")
    print(f"📸 Image count: {len(files)}")
    print(f"📦 Total size: {total_mb:.2f} MB")

    if total_mb > limit_mb:
        print(f"🚨 Payload too large! Exceeds {limit_mb} MB limit by {total_mb - limit_mb:.2f} MB.")
    else:
        print(f"✅ Within limit ({limit_mb} MB). Safe to send to GPT.")

if __name__ == "__main__":
    # Change scene_name as needed
    check_payload("HM3D00000test")
