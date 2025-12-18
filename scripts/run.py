import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model_adapters.glm4v_thinking_adapter import run_glm4v_thinking
from src.model_adapters.internvl3_5_adapter import run_internvl3_5
from src.model_adapters.mimo_vl_adapter import run_mimo_vl


# ----------------------------
# Strict allowlist / patterns
# ----------------------------

# GLM: strict allowlist (exact strings only)
ALLOWED_MODELS_EXACT = {
    "GLM-4.1V-9B-Thinking",
    "zai-org/GLM-4.1V-9B-Thinking",
}

# InternVL3_5: strict prefix pattern allowlist (scalable, not size-locked)
# Accept:
#   - InternVL3_5-8B
#   - OpenGVLab/InternVL3_5-14B
#   - OpenGVLab/InternVL3_5-241B-A28B
INTERNVL3_5_PATTERN = re.compile(r"^(?:OpenGVLab/)?InternVL3_5-[A-Za-z0-9._-]+$")

# MiMo-VL: strict prefix pattern allowlist (scalable, not size-locked)
# Accept:
#   - MiMo-VL-7B-RL-2508
#   - XiaomiMiMo/MiMo-VL-7B-SFT
#   - XiaomiMiMo/MiMo-VL-7B-RL-GGUF
MIMO_VL_PATTERN = re.compile(r"^(?:XiaomiMiMo/)?MiMo-VL-[A-Za-z0-9._-]+$")


def canonicalize_model(model: str) -> str:
    """
    Convert allowed aliases to canonical identifiers for routing.
    IMPORTANT:
      - No fuzzy matching.
      - Input must either be in exact allowlist OR match strict patterns.
    """
    # GLM: map org-prefixed alias to canonical short name for adapter routing
    if model == "zai-org/GLM-4.1V-9B-Thinking":
        return "GLM-4.1V-9B-Thinking"

    # InternVL3_5: allow missing org; canonicalize to OpenGVLab/<...>
    if INTERNVL3_5_PATTERN.match(model):
        if model.startswith("OpenGVLab/"):
            return model
        return f"OpenGVLab/{model}"

    # MiMo-VL: allow missing org; canonicalize to XiaomiMiMo/<...>
    if MIMO_VL_PATTERN.match(model):
        if model.startswith("XiaomiMiMo/"):
            return model
        return f"XiaomiMiMo/{model}"

    return model


def _is_glm(model: str) -> bool:
    return model in ALLOWED_MODELS_EXACT


def _is_internvl3_5(model: str) -> bool:
    return INTERNVL3_5_PATTERN.match(model) is not None


def _is_mimo_vl(model: str) -> bool:
    return MIMO_VL_PATTERN.match(model) is not None


def route_and_run(model: str, num_frames: int, thinking: str) -> None:
    # 1) Strict validation: either exact allowlist OR strict regex patterns
    if (model not in ALLOWED_MODELS_EXACT) and (not _is_internvl3_5(model)) and (not _is_mimo_vl(model)):
        allowed_glm = "\n  - ".join(sorted(ALLOWED_MODELS_EXACT))
        raise ValueError(
            "Unsupported --model value.\n"
            f"Received: {model}\n\n"
            "Allowed GLM values (exact match):\n"
            f"  - {allowed_glm}\n\n"
            "Allowed InternVL3_5 format (regex strict):\n"
            "  - InternVL3_5-<CHECKPOINT>\n"
            "  - OpenGVLab/InternVL3_5-<CHECKPOINT>\n"
            '    where <CHECKPOINT> matches [A-Za-z0-9._-]+ (no spaces)\n\n'
            "Allowed MiMo-VL format (regex strict):\n"
            "  - MiMo-VL-<CHECKPOINT>\n"
            "  - XiaomiMiMo/MiMo-VL-<CHECKPOINT>\n"
            '    where <CHECKPOINT> matches [A-Za-z0-9._-]+ (no spaces)\n'
        )

    canonical = canonicalize_model(model)

    # 2) Strict routing (deterministic)
    if canonical == "GLM-4.1V-9B-Thinking":
        # GLM has no "thinking off" mode in your design
        if thinking != "on":
            raise ValueError(
                'Invalid --thinking for GLM-4.1V-9B-Thinking.\n'
                'This model has no "off" mode. Please use: --thinking on'
            )
        run_glm4v_thinking(user_model=canonical, num_frames=num_frames, thinking=thinking)
        return

    if canonical.startswith("OpenGVLab/InternVL3_5-"):
        run_internvl3_5(user_model=canonical, num_frames=num_frames, thinking=thinking)
        return

    if canonical.startswith("XiaomiMiMo/MiMo-VL-"):
        run_mimo_vl(user_model=canonical, num_frames=num_frames, thinking=thinking)
        return

    # Defensive: should never happen if allowlist + routing are correct
    raise RuntimeError(f"Internal routing error for model: {model} (canonical={canonical})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CapNav runner (open-source models, strict argument enforcement)."
    )

    # All required: users MUST provide them explicitly (no defaults)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help=(
            "Strict. Either one of the allowed GLM ids, or an InternVL3_5 / MiMo-VL checkpoint id.\n"
            "InternVL3_5: InternVL3_5-... or OpenGVLab/InternVL3_5-...\n"
            "MiMo-VL:      MiMo-VL-...      or XiaomiMiMo/MiMo-VL-..."
        ),
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        required=True,
        choices=[16, 32, 64],
        help="Must be one of {16, 32, 64}.",
    )
    parser.add_argument(
        "--thinking",
        type=str,
        required=True,
        choices=["on", "off"],
        help='Must be explicitly provided: "on" or "off".',
    )

    args = parser.parse_args()
    route_and_run(args.model, args.num_frames, args.thinking)


if __name__ == "__main__":
    main()
