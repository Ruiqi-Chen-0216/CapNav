"""
Gemini API Client (Low-Level Wrapper)
------------------------------------
This module provides a clean, low-level interface for calling the Gemini API.
It does NOT handle any local I/O (reading files, printing logs, etc.)
and can be safely imported by higher-level orchestration modules.

Usage example (upper layer):
    from src.vlm_clients.gemini_client import init_gemini, upload_file, query_gemini
"""

from google import genai
import time

# =============================
# Client Initialization
# =============================

# === Define structured output Schema ===
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
                        "items": {"type": "string"}
                    },
                    "reason": {"type": "string"}
                },
                "required": ["answer", "path", "reason"]
            }
        },
        "required": ["question", "agent", "result"]
    }
}

def init_gemini(api_key: str):
    """
    Initialize Gemini client using a provided API key.

    Args:
        api_key (str): Your Gemini API key.
    Returns:
        genai.Client: An authenticated Gemini client.
    """
    return genai.Client(api_key=api_key)


# =============================
# File Upload
# =============================


def upload_file(client, file_path: str, wait_until_active: bool = True, check_interval: int = 3):
    """
    Upload a local file to Gemini API and optionally wait until it becomes ACTIVE.

    Args:
        client: Gemini client instance
        file_path (str): Local file path
        wait_until_active (bool): Wait until the uploaded file is ready
        check_interval (int): How many seconds to wait between checks

    Returns:
        file_obj: Active file object
    """
    file_obj = client.files.upload(file=file_path)

    if wait_until_active:
        print(f"📤 Uploaded file {file_obj.name}, waiting for activation...", end="", flush=True)
        while True:
            file_obj = client.files.get(name=file_obj.name)
            if file_obj.state.name == "ACTIVE":
                print(" ✅ ACTIVE")
                break
            elif file_obj.state.name == "FAILED":
                raise RuntimeError(f"File upload failed: {file_obj.error_message}")
            else:
                print(".", end="", flush=True)
                time.sleep(check_interval)
    return file_obj


# =============================
# Query Function
# =============================

def query_gemini(
    client,
    prompt: str,
    model: str = "gemini-2.5-flash",
    image_bytes: bytes = None,
    video_uri: str = None,
) -> dict:
    """
    Send text / image / video request to Gemini API with structured JSON output.

    Args:
        client: Gemini client (from init_gemini).
        prompt (str): User prompt text.
        model (str): Gemini model name.
        image_bytes (bytes, optional): Binary image data.
        video_uri (str, optional): URI of uploaded video.

    Returns:
        dict: {
            "text": JSON-formatted string (always valid JSON),
            "raw": Gemini response object
        }
    """
    parts = [{"text": prompt}]

    # optional image or video
    if image_bytes:
        parts.append({"inline_data": {"mime_type": "image/jpeg", "data": image_bytes}})
    if video_uri:
        parts.append({"file_data": {"file_uri": video_uri}})

    # --- Structured Output Config ---
    generation_config = {
        "response_mime_type": "application/json",
        "response_schema": NAVIGATION_SCHEMA
    }

    # --- Send request ---
    response = client.models.generate_content(
        model=model,
        contents=[{"role": "user", "parts": parts}],
        config=generation_config
    )

    return {
    "text": getattr(response, "text", ""),
    "raw": response,
    "usage": getattr(response, "usage_metadata", None)  # ✅ 新增：token 信息
    }
