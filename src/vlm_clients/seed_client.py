"""
Doubao Seed Structured Video Query (via OpenAI SDK)
---------------------------------------------------
This version fully supports structured JSON output using `pydantic.BaseModel`
and the new `.beta.chat.completions.parse()` API, while accepting a local MP4
video (base64-encoded) with adjustable FPS.

✅ 100% compliant with Ark Doubao Seed API specification.
"""

import os
import base64
from typing import List
from pydantic import BaseModel, RootModel
from openai import OpenAI


# ======================================================
# 1. Initialize client
# ======================================================
def init_seed_client(api_key: str = None):
    """
    Initialize OpenAI-compatible client for Ark Doubao Seed.
    """
    return OpenAI(
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        api_key=api_key or os.environ.get("ARK_API_KEY"),
    )


# ======================================================
# 2. Encode video to base64
# ======================================================
def encode_video_to_base64(video_path: str) -> str:
    """
    Encode a local MP4 video into base64 string.
    """
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ======================================================
# 3. Define structured output schema
# ======================================================
class Result(BaseModel):
    answer: str
    path: List[str]
    reason: str


class QAItem(BaseModel):
    question: str
    agent: str
    result: Result


class NavigationResponse(RootModel[List[QAItem]]):
    """Root model wrapping a list of QAItem results."""
    pass


# ======================================================
# 4. Structured video query
# ======================================================
def query_seed_video_structured(
    client,
    video_path: str,
    prompt: str,
    model: str = "doubao-seed-1-6-251015",
    thinking_type: str = "disabled",  # 可选: "enabled" 或 "disabled"
    fps: int = 2                      # ✅ 每秒截取帧数
):
    """
    Send an MP4 video and text query to Doubao Seed with structured output parsing.

    Args:
        client: OpenAI client from init_seed_client()
        video_path: path to local MP4 file
        prompt: user instruction text
        model: Seed model ID
        thinking_type: "enabled" or "disabled" for deep reasoning
        fps: integer, number of frames per second used for analysis

    Returns:
        dict with keys:
            - "parsed": structured Pydantic object
            - "json": JSON string version
            - "raw": raw API response
    """

    # === Base64 encoding ===
    base64_video = encode_video_to_base64(video_path)
    print(f"🎞️ Encoded video ({os.path.basename(video_path)}) with fps={fps}")

    # === Send video to model ===
    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {
                            "url": f"data:video/mp4;base64,{base64_video}",
                            "fps": fps,  # ✅ now explicitly supported
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        response_format=NavigationResponse,  # ✅ structured output
        extra_body={
            "thinking": {"type": thinking_type}
        }
    )

    parsed = completion.choices[0].message.parsed
    return {
        "parsed": parsed,
        "json": parsed.model_dump_json(indent=2),
        "raw": completion
    }
