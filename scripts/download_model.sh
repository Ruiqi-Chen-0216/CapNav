#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Portable model downloader for open-source use.
#
# Usage:
#   bash scripts/download_model.sh <hf_repo_id> [output_dir]
#
# Examples:
#   bash scripts/download_model.sh zai-org/GLM-4.1V-9B-Thinking
#   bash scripts/download_model.sh Qwen/Qwen3-VL-8B-Instruct models
#
# Notes:
# - If you have a gated model, run: huggingface-cli login
# - You can control caches with env vars (optional):
#     export HF_HOME=/path/to/hf_home
#     export HF_HUB_CACHE=/path/to/hf_hub_cache
#     export TRANSFORMERS_CACHE=/path/to/transformers_cache
# ============================================================

if [[ $# -lt 1 ]]; then
  echo "Usage: bash $0 <huggingface_repo_id> [output_dir]"
  echo 'Example: bash $0 zai-org/GLM-4.1V-9B-Thinking models'
  exit 1
fi

MODEL_ID="$1"
OUT_DIR="${2:-models}"

MODEL_NAME="$(basename "${MODEL_ID}")"
TARGET_DIR="${OUT_DIR}/${MODEL_NAME}"

echo "[INFO] Model repo: ${MODEL_ID}"
echo "[INFO] Target dir: ${TARGET_DIR}"

# Ensure CLI exists
if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "[INFO] huggingface-cli not found; installing huggingface_hub..."
  python -m pip install -q --upgrade huggingface_hub
fi

# Create output dir
mkdir -p "${TARGET_DIR}"

# Download model snapshot to local directory
huggingface-cli download "${MODEL_ID}" \
  --local-dir "${TARGET_DIR}" \
  --local-dir-use-symlinks False \
  --resume-download

echo "[DONE] Model is ready at: ${TARGET_DIR}"
