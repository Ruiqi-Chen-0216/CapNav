"""
Gemini API Client (Credential / Vertex version)
-----------------------------------------------
This module provides a clean, low-level interface for calling Gemini
in Vertex AI mode using Service Account credentials (JSON key).

Key differences from the API-key version:
✅ Uses vertexai=True (for Google Cloud projects)
✅ Accepts gs:// URIs for video
✅ Handles response_schema for structured JSON output
"""

import time
from google import genai
from google.genai import types

# =============================
# Structured Output Schema
# =============================

NAVIGATION_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "question": {"type": "string"},
            "agent": {"type": "string"},
            "result": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string"},
                    "path": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "reason": {"type": "string"},
                },
                "required": ["answer", "path", "reason"],
            },
        },
        "required": ["question", "agent", "result"],
    },
}

# =============================
# Client Initialization
# =============================

def init_gemini_credential(project_id: str, location: str = "us-central1"):
    """
    Initialize Gemini client via Vertex AI + service account JSON.

    Requires:
        export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json

    Args:
        project_id (str): GCP project id
        location (str): Vertex AI region
    Returns:
        genai.Client: Authenticated Vertex AI client
    """
    client = genai.Client(vertexai=True, project=project_id, location=location)
    print(f"✅ Initialized Vertex Gemini client (project={project_id}, location={location})")
    return client


# =============================
# Query Function (Vertex-style)
# =============================

def query_gemini_credential(
    client,
    prompt: str,
    video_uri: str = None,
    model: str = "gemini-2.5-pro",
    fps: float = 1.0,
):
    """
    Send text + (optional) GCS video to Gemini with structured output and fps control.
    """

    parts = []

    # === 视频输入（GCS 视频） ===
    if video_uri:
        video_part = types.Part(
            file_data=types.FileData(
                file_uri=video_uri,
                mime_type="video/mp4"
            ),
            video_metadata=types.VideoMetadata(fps=fps)
        )
        parts.append(video_part)

    # === 文本输入 ===
    text_part = types.Part(text=prompt)
    parts.append(text_part)

    # === Structured output config ===
    generation_config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=NAVIGATION_SCHEMA,
    )

    # === 调用 Gemini ===
    response = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],  # ✅ 必须加 role="user"
        config=generation_config,
    )

    # === 提取用量 ===
    usage_meta = getattr(response, "usage_metadata", None)
    usage_info = None
    if usage_meta:
        usage_info = {
            "prompt_tokens": getattr(usage_meta, "input_tokens", None),
            "response_tokens": getattr(usage_meta, "output_tokens", None),
            "total_tokens": (
                getattr(usage_meta, "input_tokens", 0)
                + getattr(usage_meta, "output_tokens", 0)
            ),
        }

    return {
        "text": getattr(response, "text", ""),
        "raw": response,
        "usage": usage_info,
    }
