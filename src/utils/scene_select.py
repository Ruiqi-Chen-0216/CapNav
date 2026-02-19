from __future__ import annotations

import os
from typing import List, Optional


def detect_scenes_from_graphs(graph_dir: str) -> List[str]:
    """
    Detect scene ids from files named like: <SCENE>-graph.json
    """
    scenes: List[str] = []
    for f in os.listdir(graph_dir):
        if f.endswith("-graph.json"):
            scenes.append(f.split("-graph.json")[0])
    scenes.sort()
    return scenes


def _dedupe_keep_order(xs: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in xs:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def resolve_scenes(
    graph_dir: str,
    scenes_allowlist: Optional[List[str]] = None,
    strict: bool = True,
) -> List[str]:
    """
    Resolve which scenes to run.

    - If scenes_allowlist is None: return all detected scenes (sorted).
    - If provided: return intersection of allowlist and detected scenes,
      preserving the allowlist order.

    strict:
      - True: if any allowlisted scene is missing, raise FileNotFoundError.
      - False: silently ignore missing ones (caller may print warnings).
    """
    detected = detect_scenes_from_graphs(graph_dir)
    if scenes_allowlist is None:
        return detected

    allow = _dedupe_keep_order(scenes_allowlist)
    detected_set = set(detected)

    selected = [s for s in allow if s in detected_set]
    missing = [s for s in allow if s not in detected_set]

    if missing and strict:
        raise FileNotFoundError(
            "Some allowlisted scenes were not found in GRAPH_DIR.\n"
            f"GRAPH_DIR: {graph_dir}\n"
            f"Missing scenes: {missing}\n"
            f"Detected scenes (count={len(detected)}): (showing first 20) {detected[:20]}"
        )

    return selected
