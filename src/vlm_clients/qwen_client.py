"""
qwen_client.py
--------------
DashScope-native Qwen client for video reasoning tasks.
Implements:
    - init_qwen(api_key, region)
    - query_qwen(api_key, video_path, prompt, ...)
"""

import os
import json
import dashscope
from dashscope import MultiModalConversation


# ======================================================
# 1️⃣ Initialize client environment
# ======================================================
def init_qwen(api_key: str, region: str = "intl"):
    """
    Initialize DashScope API endpoint (Qwen) environment.

    Args:
        api_key (str): DashScope API key (passed from orchestrator)
        region (str): "intl" (Singapore) or "cn" (Beijing)
    Returns:
        dict: configuration dict containing api_key and region
    """
    dashscope.base_http_api_url = (
        "https://dashscope-intl.aliyuncs.com/api/v1"
        if region == "intl"
        else "https://dashscope.aliyuncs.com/api/v1"
    )
    return {"api_key": api_key, "region": region}


# ======================================================
# 2️⃣ Utility: ensure file:// URI for video input
# ======================================================
def ensure_video_uri(video_path: str) -> str:
    """Convert local video path to file:// URI for DashScope."""
    if not video_path.startswith("file://"):
        return f"file://{os.path.abspath(video_path)}"
    return video_path


# ======================================================
# 3️⃣ Non-stream structured call
# ======================================================
def query_qwen(
    client_cfg: dict,
    video_path: str,
    prompt: str,
    model: str = "qwen3-vl-plus",
    fps: int = 2,
    enable_thinking: bool = False,
):
    """
    Perform structured video reasoning with Qwen.

    Args:
        client_cfg (dict): From init_qwen(api_key, region)
        video_path (str): Local path to video file
        prompt (str): Task prompt (must define JSON schema)
        model (str): Model name
        fps (int): Video frame sampling rate
        enable_thinking (bool): Enable deep reasoning

    Returns:
        dict or list: Parsed JSON output, or None on failure.
    """
    api_key = client_cfg["api_key"]
    video_uri = ensure_video_uri(video_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"video": video_uri, "fps": fps},
                {"text": prompt},
            ],
        }
    ]

    response = MultiModalConversation.call(
        api_key=api_key,
        model=model,
        messages=messages,
        enable_thinking=enable_thinking,
        response_format={"type": "json_object"},
    )

    try:
        text_output = response["output"]["choices"][0]["message"]["content"][0]["text"]
        return json.loads(text_output)
    except Exception as e:
        print("⚠️ JSON 解析失败：", e)
        print("原始输出：\n", json.dumps(response, indent=2, ensure_ascii=False))
        return None
