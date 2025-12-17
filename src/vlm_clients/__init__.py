"""
VLM Client Package
------------------
Unified interface for all Vision-Language Model clients.
Each client module implements:
    - init_<model>()
    - query_<model>()
"""

# ==============================
# Gemini Client
# ==============================
from .gemini_client import init_gemini, query_gemini

# ==============================
# Doubao Seed Client
# ==============================
from .seed_client import (
    init_seed_client,
    query_seed_video_structured,
    encode_video_to_base64,
    Result,
    QAItem,
    NavigationResponse,
)

# ==============================
# Qwen Client
# ==============================
from .qwen_client import init_qwen, query_qwen


# ==============================
# GPT Client
# ==============================
from .gpt_client import (
    init_gpt,
    upload_file,
    query_gpt,
    NavigationResult,
    NavigationItem,
    NavigationOutput,
)

__all__ = [
    # Gemini
    "init_gemini", "query_gemini",
    # Seed
    "init_seed_client", "query_seed_video_structured",
    "encode_video_to_base64", "Result", "QAItem", "NavigationResponse",
    # Qwen
    "init_qwen", "query_qwen",
    # GPT
    "init_gpt", "upload_file", "query_gpt",
    "NavigationResult", "NavigationItem", "NavigationOutput",
]