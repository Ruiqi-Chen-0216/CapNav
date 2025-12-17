from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod


@dataclass
class RunArgs:
    # User-facing minimal inputs
    model: str                      # HF model id (recommended) or local path
    num_frames: int = 64

    # Optional override
    model_dir: Optional[str] = None  # if provided, prefer local dir

    # Repo-relative defaults (open-source friendly)
    graph_dir: str = "ground_truth/graphs"
    prompt_root: str = "generated_prompts"
    video_root: str = "videos_64frames_1fps"     # contains <scene>.mp4
    result_root: str = "results"

    # Sampling / generation defaults
    base_fps: int = 1
    total_frames: int = 64
    max_new_tokens: int = 8192
    delay: float = 0.0

    # Model download location for scripts/download_model.sh
    download_dir: str = "models"


class BaseAdapter(ABC):
    """
    A thin interface so scripts/run.py can call any backend without
    knowing model-specific engineering details.
    """
    name: str

    @abstractmethod
    def run(self, args: RunArgs) -> None:
        raise NotImplementedError
