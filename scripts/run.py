import argparse
import sys

# Ensure repo root is on PYTHONPATH so `src.*` imports work when running from scripts/
# This avoids requiring users to export PYTHONPATH manually.
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.model_adapters.glm4v_thinking_adapter import run_glm4v_thinking


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CapNav runner (open-source models)."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help='Model name or HF repo id. Example: "GLM-4.1V-9B-Thinking" or "zai-org/GLM-4.1V-9B-Thinking".'
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        required=True,
        choices=[16, 32, 64],
        help="Number of frames to sample from each video."
    )
    parser.add_argument(
        "--thinking",
        type=str,
        default="on",
        choices=["on", "off"],
        help='Thinking mode: "on" or "off". Note: GLM-4.1V-9B-Thinking only supports "on".'
    )

    args = parser.parse_args()

    # For now, route all runs to the GLM adapter as the first open-source backend.
    run_glm4v_thinking(
        user_model=args.model,
        num_frames=args.num_frames,
        thinking=args.thinking,
    )


if __name__ == "__main__":
    main()
