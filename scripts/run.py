import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model_adapters.glm4v_thinking_adapter import run_glm4v_thinking
from src.model_adapters.internvl3_5_adapter import run_internvl3_5
from src.model_adapters.mimo_vl_adapter import run_mimo_vl
from src.model_adapters.qwen3_vl_adapter import run_qwen3_vl


# ----------------------------
# Strict allowlist / patterns
# ----------------------------

# GLM: exact allowlist only
ALLOWED_MODELS_EXACT = {
    "GLM-4.1V-9B-Thinking",
    "zai-org/GLM-4.1V-9B-Thinking",
}

# InternVL3_5: strict prefix + safe chars, scalable across sizes/variants
INTERNVL3_5_PATTERN = re.compile(r"^(?:OpenGVLab/)?InternVL3_5-[A-Za-z0-9._-]+$")

# MiMo-VL: strict prefix + safe chars, scalable across sizes/variants
# Accept examples:
#   - MiMo-VL-7B-RL-2508
#   - XiaomiMiMo/MiMo-VL-7B-SFT
MIMOVL_PATTERN = re.compile(r"^(?:XiaomiMiMo/)?MiMo-VL-[A-Za-z0-9._-]+$")

# Qwen3-VL: strict prefix + safe chars, scalable across sizes/variants
# Accept examples:
#   - Qwen3-VL-30B-A3B-Thinking-FP8
#   - Qwen/Qwen3-VL-235B-A22B-Instruct
QWEN3_VL_PATTERN = re.compile(r"^(?:Qwen/)?Qwen3-VL-[A-Za-z0-9._-]+$")


# ----------------------------
# Canonicalization
# ----------------------------

def canonicalize_model(model: str) -> str:
    """
    Convert allowed aliases to canonical identifiers for downstream adapters.
    IMPORTANT: No fuzzy matching. Input must already be valid by exact allowlist
    or strict regex patterns.
    """
    # GLM: allow short form and canonicalize to non-org name for routing
    if model == "zai-org/GLM-4.1V-9B-Thinking":
        return "GLM-4.1V-9B-Thinking"

    # InternVL3_5: canonicalize to OpenGVLab/<...>
    if INTERNVL3_5_PATTERN.match(model):
        if model.startswith("OpenGVLab/"):
            return model
        return f"OpenGVLab/{model}"

    # MiMo-VL: canonicalize to XiaomiMiMo/<...>
    if MIMOVL_PATTERN.match(model):
        if model.startswith("XiaomiMiMo/"):
            return model
        return f"XiaomiMiMo/{model}"

    # Qwen3-VL: canonicalize to Qwen/<...>
    if QWEN3_VL_PATTERN.match(model):
        if model.startswith("Qwen/"):
            return model
        return f"Qwen/{model}"

    return model


# ----------------------------
# Type checks
# ----------------------------

def _is_glm(model: str) -> bool:
    return model in ALLOWED_MODELS_EXACT


def _is_internvl3_5(model: str) -> bool:
    return INTERNVL3_5_PATTERN.match(model) is not None


def _is_mimo_vl(model: str) -> bool:
    return MIMOVL_PATTERN.match(model) is not None


def _is_qwen3_vl(model: str) -> bool:
    return QWEN3_VL_PATTERN.match(model) is not None


# ----------------------------
# Qwen3-VL thinking mode validation
# ----------------------------

def _qwen3_vl_checkpoint_mode(canonical_model: str) -> str:
    """
    Determine whether the Qwen3-VL checkpoint is Thinking or Instruct.
    IMPORTANT: This is model-name-based because Qwen3-VL splits checkpoints.
    Returns: "thinking" or "instruct"
    """
    name = canonical_model.split("/", 1)[-1]  # remove org
    # Must be strict: require token presence in the name.
    # Not necessarily at the end, e.g., Thinking-FP8.
    if "Thinking" in name:
        return "thinking"
    if "Instruct" in name:
        return "instruct"
    raise ValueError(
        "Qwen3-VL checkpoint name must include either 'Thinking' or 'Instruct'.\n"
        f"Received: {canonical_model}\n"
        "Examples:\n"
        "  - Qwen/Qwen3-VL-30B-A3B-Thinking\n"
        "  - Qwen/Qwen3-VL-30B-A3B-Instruct\n"
        "  - Qwen/Qwen3-VL-30B-A3B-Thinking-FP8\n"
    )


def _enforce_qwen3_vl_thinking(canonical_model: str, thinking: str) -> None:
    mode = _qwen3_vl_checkpoint_mode(canonical_model)
    thinking_norm = thinking.lower().strip()

    if mode == "thinking" and thinking_norm != "on":
        raise ValueError(
            "Invalid --thinking for this Qwen3-VL checkpoint.\n"
            f"Model: {canonical_model}\n"
            "This is a Thinking checkpoint, so you must use: --thinking on"
        )

    if mode == "instruct" and thinking_norm != "off":
        raise ValueError(
            "Invalid --thinking for this Qwen3-VL checkpoint.\n"
            f"Model: {canonical_model}\n"
            "This is an Instruct checkpoint (no-think), so you must use: --thinking off"
        )


# ----------------------------
# Routing
# ----------------------------

def route_and_run(model: str, num_frames: int, thinking: str) -> None:
    # 1) Strict validation: either exact allowlist OR strict regex pattern
    if (model not in ALLOWED_MODELS_EXACT) and (not _is_internvl3_5(model)) and (not _is_mimo_vl(model)) and (not _is_qwen3_vl(model)):
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
            '    where <CHECKPOINT> matches [A-Za-z0-9._-]+ (no spaces)\n\n'
            "Allowed Qwen3-VL format (regex strict):\n"
            "  - Qwen3-VL-<CHECKPOINT>\n"
            "  - Qwen/Qwen3-VL-<CHECKPOINT>\n"
            '    where <CHECKPOINT> matches [A-Za-z0-9._-]+ (no spaces)\n'
            "    and the checkpoint name must include 'Thinking' or 'Instruct'.\n"
        )

    canonical = canonicalize_model(model)

    # 2) 100% strict routing
    if canonical == "GLM-4.1V-9B-Thinking":
        # GLM: no thinking-off in your design
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

    if canonical.startswith("Qwen/Qwen3-VL-"):
        # Qwen3-VL: checkpoint determines whether thinking is allowed
        _enforce_qwen3_vl_thinking(canonical, thinking)
        run_qwen3_vl(user_model=canonical, num_frames=num_frames, thinking=thinking)
        return

    # Defensive: should never happen if allowlist + routing are correct
    raise RuntimeError(f"Internal routing error for model: {model} (canonical={canonical})")


# ----------------------------
# CLI
# ----------------------------

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
            "Strict. Either one of the allowed GLM ids, or a checkpoint id matching:\n"
            "  - InternVL3_5-...  (or OpenGVLab/InternVL3_5-...)\n"
            "  - MiMo-VL-...      (or XiaomiMiMo/MiMo-VL-...)\n"
            "  - Qwen3-VL-...     (or Qwen/Qwen3-VL-...; must contain Thinking or Instruct)\n"
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
