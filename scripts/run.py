import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model_adapters.glm4v_thinking_adapter import run_glm4v_thinking
from src.model_adapters.internvl3_5_adapter import run_internvl3_5


# ----------------------------
# Strict allowlist / patterns
# ----------------------------

# GLM: strict allowlist (exact strings only)
ALLOWED_MODELS_EXACT = {
    "GLM-4.1V-9B-Thinking",
    "zai-org/GLM-4.1V-9B-Thinking",
}

# InternVL3_5: strict pattern allowlist (still strict, but scalable)
# Accept:
#   - InternVL3_5-8B
#   - OpenGVLab/InternVL3_5-14B
#   - OpenGVLab/InternVL3_5-241B-A28B
# ... as long as it starts with InternVL3_5- (with optional "OpenGVLab/")
# and uses only safe characters.
INTERNVL3_5_PATTERN = re.compile(r"^(?:OpenGVLab/)?InternVL3_5-[A-Za-z0-9._-]+$")


def canonicalize_model(model: str) -> str:
    """
    Convert allowed aliases to canonical identifiers for routing.
    IMPORTANT: No fuzzy matching. Input must already be exactly allowed
    (for exact allowlist) OR match a strict allowlist pattern.
    """
    if model == "zai-org/GLM-4.1V-9B-Thinking":
        return "GLM-4.1V-9B-Thinking"

    # For InternVL3_5, allow missing org; canonicalize to OpenGVLab/<...>
    if INTERNVL3_5_PATTERN.match(model):
        if model.startswith("OpenGVLab/"):
            return model
        return f"OpenGVLab/{model}"

    return model


def _is_glm(model: str) -> bool:
    return model in ALLOWED_MODELS_EXACT


def _is_internvl3_5(model: str) -> bool:
    return INTERNVL3_5_PATTERN.match(model) is not None


def route_and_run(model: str, num_frames: int, thinking: str) -> None:
    # 1) Strict validation: either exact allowlist OR strict regex pattern
    if (model not in ALLOWED_MODELS_EXACT) and (not _is_internvl3_5(model)):
        allowed_glm = "\n  - ".join(sorted(ALLOWED_MODELS_EXACT))
        raise ValueError(
            "Unsupported --model value.\n"
            f"Received: {model}\n\n"
            "Allowed GLM values (exact match):\n"
            f"  - {allowed_glm}\n\n"
            "Allowed InternVL3_5 format (regex strict):\n"
            "  - InternVL3_5-<CHECKPOINT>\n"
            "  - OpenGVLab/InternVL3_5-<CHECKPOINT>\n"
            '    where <CHECKPOINT> matches [A-Za-z0-9._-]+ (no spaces)\n'
        )

    canonical = canonicalize_model(model)

    # 2) 100% strict routing
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
        # InternVL3_5 supports both on/off (your adapter will implement it)
        run_internvl3_5(user_model=canonical, num_frames=num_frames, thinking=thinking)
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
        help="Strict. Either one of the allowed GLM ids, or InternVL3_5 checkpoint id "
             '(InternVL3_5-... or OpenGVLab/InternVL3_5-...).',
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
