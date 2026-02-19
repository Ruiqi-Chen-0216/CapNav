from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class DataSample:
    name: str
    scenes: List[str]


# ====================================
# Curated sample (~200 prompts total)
# Maintain this list internally.
# ====================================

CAPNAV_SAMPLE_200 = DataSample(
    name="sample_200",
    scenes=[
        # TODO: replace with your curated scenes
        "HM3D00010",
        "HM3D00014",
        "HM3D00027",
        "MP3D00030",
    ],
)
