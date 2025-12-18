import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model_adapters.glm4v_thinking_adapter import run_glm4v_thinking


# Strict allowlist: only these exact strings are accepted
ALLOWED_MODELS = {
    "GLM-4.1V-9B-Thinking",
    "zai-org/GLM-4.1V-9B-Thinking",
}


def canonicalize_model(model: str) -> str:
    """
    Convert an allowed alias to a canonical identifier for routing.
    IMPORTANT: No fuzzy matching. Input must already be exactly allowed.
    """
    if model == "zai-org/GLM-4.1V-9B-Thinking":
        return "GLM-4.1V-9B-Thinking"
    return model


def route_and_run(model: str, num_frames: int, thinking: str) -> None:
    # Enforce 100% exact model match
    if model not in ALLOWED_MODELS:
        allowed = "\n  - ".join(sorted(ALLOWED_MODELS))
        raise ValueError(
            "Unsupported --model value (strict matching is enabled).\n"
            f"Received: {model}\n"
            "Allowed values:\n"
            f"  - {allowed}\n"
        )

    canonical = canonicalize_model(model)

    # 100% exact routing
    if canonical == "GLM-4.1V-9B-Thinking":
        run_glm4v_thinking(user_model=canonical, num_frames=num_frames, thinking=thinking)
        return

    # Defensive: should never happen if allowlist is correct
    raise RuntimeError(f"Internal routing error for model: {model}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CapNav runner (open-source models, strict argument enforcement)."
    )

    # All required: users MUST provide them explicitly (no defaults)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help='Must match exactly one allowed model id (strict).'
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        required=True,
        choices=[16, 32, 64],
        help="Must be one of {16, 32, 64}."
    )
    parser.add_argument(
        "--thinking",
        type=str,
        required=True,
        choices=["on", "off"],
        help='Must be explicitly provided: "on" or "off".'
    )

    args = parser.parse_args()
    route_and_run(args.model, args.num_frames, args.thinking)


if __name__ == "__main__":
    main()
