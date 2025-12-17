"""
GPT API Client (Structured Output, Pydantic)
--------------------------------------------
Low-level wrapper for GPT Responses API, aligned with Gemini_client style,
but using the new structured output interface via pydantic.BaseModel.

Supports:
    - init_gpt(): client initialization
    - upload_file(): File API upload (for vision inputs)
    - query_gpt(): send request + structured parsing into Pydantic model
"""

from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional
import time


# ======================================
# 1️⃣ Client Initialization
# ======================================

def init_gpt(api_key: Optional[str] = None):
    """
    Initialize GPT client using a provided API key or environment variable.
    """
    return OpenAI(api_key=api_key)


# ======================================
# 2️⃣ File Upload
# ======================================

def upload_file(
    client,
    file_path: str,
    purpose: str = "vision",
    wait_until_ready: bool = True,
    check_interval: int = 3,
):
    """
    Upload a file (image, etc.) to the OpenAI File API.

    Args:
        client: OpenAI client instance.
        file_path (str): Local path to file (supports .png/.jpg/.jpeg).
        purpose (str): Upload purpose. Usually "vision".
        wait_until_ready (bool): Whether to poll until the file is processed.
        check_interval (int): Polling interval in seconds.

    Returns:
        file_obj: The uploaded file object.
    """
    with open(file_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose=purpose)
    print(f"📤 Uploaded {file_path} → file_id={file_obj.id}")

    if wait_until_ready:
        while True:
            status = client.files.retrieve(file_obj.id)
            if status.status in ["processed", "uploaded"]:
                break
            time.sleep(check_interval)

    return file_obj


# ======================================
# 3️⃣ Structured Output Models
# ======================================

class NavigationResult(BaseModel):
    """Result of one navigation QA item."""
    answer: str
    path: List[str]
    reason: str


class NavigationItem(BaseModel):
    """One QA item, including question, agent, and reasoning result."""
    question: str
    agent: str
    result: NavigationResult


class NavigationOutput(BaseModel):
    """
    Top-level structured output for GPT Responses API.
    Must be an object (not a root array).
    """
    items: List[NavigationItem]


# ======================================
# 4️⃣ Query Function (with Pydantic parsing)
# ======================================

def query_gpt(
    client,
    prompt: str,
    model: str = "gpt-5",
    image_file_ids: Optional[List[str]] = None,
    output_model: Optional[BaseModel] = None,
    reasoning_effort: str = "medium",
):
    """
    Send text/multimodal request to GPT and optionally parse into a Pydantic model.

    Args:
        client: OpenAI client instance.
        prompt (str): Instruction or task prompt.
        model (str): Model name (e.g., "gpt-5").
        image_file_ids (List[str], optional): List of uploaded image file IDs.
        output_model (BaseModel, optional): Expected structured output schema.
        reasoning_effort (str): "minimal" | "medium" | "high".

    Returns:
        dict:
            {
                "parsed": structured output (if schema provided),
                "text": raw text output (if no schema),
                "raw": original API response,
                "usage": token usage object
            }
    """


    # === 构建输入内容 ===
    content = [{"type": "input_text", "text": prompt}]
    if image_file_ids:
        for fid in image_file_ids:
            content.append({"type": "input_image", "file_id": fid})

    # === 检查模型是否支持 reasoning 参数 ===
    supports_reasoning = model.startswith("gpt-5") or model.startswith("o1")

    # === 支持结构化输出的路径（GPT-5 / o1 系列）===
    if output_model and supports_reasoning:
        response = client.responses.parse(
            model=model,
            input=[{"role": "user", "content": content}],
            text_format=output_model,
            reasoning={"effort": reasoning_effort},
        )
        return {
            "parsed": response.output_parsed,
            "raw": response,
            "usage": getattr(response, "usage", None),
        }

    # === 支持 reasoning 但不结构化的路径 ===
    if supports_reasoning and not output_model:
        response = client.responses.create(
            model=model,
            input=[{"role": "user", "content": content}],
            reasoning={"effort": reasoning_effort},
        )
        return {
            "text": response.output_text,
            "raw": response,
            "usage": getattr(response, "usage", None),
        }

    # === GPT-4.1 或其他旧模型路径（不带 reasoning 参数）===
    response = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content}],
    )
    return {
        "text": response.output_text,
        "raw": response,
        "usage": getattr(response, "usage", None),
    }
