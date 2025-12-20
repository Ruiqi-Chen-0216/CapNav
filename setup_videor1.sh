#!/usr/bin/env bash
# =====================================================
# Video-R1 Environment Bootstrap (Reference, Version-Locked)
# Author: Ruiqi
#
# NOTE (Open-Source / Safety):
# - This is a *reference* setup script that follows the official Video-R1 repo.
# - It intentionally does NOT override user cache locations (HF_HOME, TRANSFORMERS_CACHE,
#   PIP_CACHE_DIR, TMPDIR, etc.). Configure those in your own shell/cluster profile if needed.
# - Versions are strictly pinned for reproducibility. If your system differs, it may fail by design.
# - Users MUST download the pinned transformers zip themselves (see README / official Video-R1 notes).
#
# Usage:
#   bash scripts/setup_videoR1.sh
#
# Optional overrides:
#   SCR_ROOT=/scr
#   CONDA_DIR=/scr/anaconda3_ruiqi
#   INSTALLER=Anaconda3-2025.06-0-Linux-x86_64.sh
#   REPO_DIR=/scr/Video-R1
#   ENV_NAME=video-r1
#   PYTHON_VERSION=3.11
#   TORCH_INDEX_URL=https://download.pytorch.org/whl/cu130
#   TRANSFORMERS_ZIP=/path/to/transformers-main.zip
# =====================================================

set -euo pipefail
trap 'echo "❌ Failed at line $LINENO"; exit 1' ERR

echo "🚀 Setting up Video-R1 environment (reference, version-locked)"

# -----------------------------------------------------
# Hard prerequisites (fail fast)
# -----------------------------------------------------
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "❌ This reference script is Linux-only."
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "❌ git not found."
  exit 1
fi

if ! command -v wget >/dev/null 2>&1; then
  echo "❌ wget not found."
  exit 1
fi

# CUDA driver presence check (Video-R1 + cu130 expectation)
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "❌ nvidia-smi not found. CUDA GPU environment required for this reference setup."
  exit 1
fi
nvidia-smi >/dev/null 2>&1 || { echo "❌ nvidia-smi failed; driver/GPU not ready."; exit 1; }

# -----------------------------------------------------
# Config (override-friendly)
# -----------------------------------------------------
SCR_ROOT="${SCR_ROOT:-/scr}"
CONDA_DIR="${CONDA_DIR:-/scr/anaconda3_ruiqi}"
INSTALLER="${INSTALLER:-Anaconda3-2025.06-0-Linux-x86_64.sh}"
REPO_DIR="${REPO_DIR:-/scr/Video-R1}"
ENV_NAME="${ENV_NAME:-video-r1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

# Official requirement in your workflow: cu130 wheels
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

# User-provided transformers zip (must be downloaded by user)
TRANSFORMERS_ZIP="${TRANSFORMERS_ZIP:-transformers-main.zip}"

# -----------------------------------------------------
# Step 1. Move to SCR_ROOT
# -----------------------------------------------------
cd "$SCR_ROOT"

# -----------------------------------------------------
# Step 2. Install Anaconda if needed
# -----------------------------------------------------
if [[ ! -f "$INSTALLER" ]]; then
  echo "⬇️ Downloading Anaconda installer..."
  wget "https://repo.anaconda.com/archive/${INSTALLER}"
fi

if [[ ! -d "$CONDA_DIR" ]]; then
  echo "📦 Installing Anaconda to $CONDA_DIR..."
  bash "$INSTALLER" -b -p "$CONDA_DIR"
fi

# -----------------------------------------------------
# Step 3. Initialize conda (non-invasive; DO NOT conda init)
# -----------------------------------------------------
export PATH="${CONDA_DIR}/bin:${PATH}"
# shellcheck disable=SC1091
source "${CONDA_DIR}/etc/profile.d/conda.sh"

# -----------------------------------------------------
# Step 4. Create / Activate environment
# -----------------------------------------------------
if ! conda info --envs | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "🧩 Creating conda environment: $ENV_NAME (python=$PYTHON_VERSION)"
  conda create -n "$ENV_NAME" "python=${PYTHON_VERSION}" -y
fi
conda activate "$ENV_NAME"

# -----------------------------------------------------
# Step 5. Base dependencies (strict where required)
# -----------------------------------------------------
echo "📦 Installing base dependencies..."
python -m pip install --upgrade pip setuptools wheel

# Torch + CUDA (STRICT cu130)
python -m pip install torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"

# Common runtime deps (keep minimal here; Video-R1/setup.sh may install more)
python -m pip install einops timm decord accelerate av tiktoken

# -----------------------------------------------------
# Step 6. Clone Video-R1 repo
# -----------------------------------------------------
cd "$SCR_ROOT"
if [[ ! -d "$REPO_DIR" ]]; then
  echo "📂 Cloning Video-R1 repository..."
  git clone https://github.com/tulerfeng/Video-R1 "$REPO_DIR"
fi
cd "$REPO_DIR"

# -----------------------------------------------------
# Step 7. Run official Video-R1 setup.sh
# -----------------------------------------------------
if [[ ! -f "setup.sh" ]]; then
  echo "❌ setup.sh not found in $REPO_DIR. Repo structure may have changed."
  exit 1
fi

echo "🔧 Running Video-R1/setup.sh (official)"
bash setup.sh

# -----------------------------------------------------
# Step 8. Install qwen-vl-utils in editable mode
# -----------------------------------------------------
if [[ ! -d "src/qwen-vl-utils" ]]; then
  echo "❌ src/qwen-vl-utils not found. Repo structure may have changed."
  exit 1
fi

echo "🔧 Installing qwen-vl-utils (editable) ..."
cd "$REPO_DIR/src/qwen-vl-utils"
python -m pip install -e .[decord]

# -----------------------------------------------------
# Step 9. Version fixes (STRICT pins you validated)
# -----------------------------------------------------
echo "📌 Pinning critical library versions..."
python -m pip install "vllm==0.7.2" "trl==0.16.0"

# -----------------------------------------------------
# Step 10. Install pinned transformers from user-provided zip
# -----------------------------------------------------
cd "$REPO_DIR/src"

# TRANSFORMERS_ZIP can be:
#   - absolute path: /path/to/transformers-main.zip
#   - relative path: transformers-main.zip (must exist in current dir)
ZIP_PATH="$TRANSFORMERS_ZIP"
if [[ ! -f "$ZIP_PATH" ]]; then
  # If user passed only filename but it's not here, try repo root as fallback.
  if [[ -f "$REPO_DIR/$TRANSFORMERS_ZIP" ]]; then
    ZIP_PATH="$REPO_DIR/$TRANSFORMERS_ZIP"
  else
    echo "❌ Pinned transformers zip not found: $TRANSFORMERS_ZIP"
    echo "   Please download it yourself (as instructed in README / official Video-R1 notes), then re-run:"
    echo "   TRANSFORMERS_ZIP=/absolute/path/to/transformers-main.zip bash scripts/setup_videoR1.sh"
    exit 1
  fi
fi

echo "📦 Installing pinned transformers from zip: $ZIP_PATH"
rm -rf transformers-main
unzip -o "$ZIP_PATH" -d transformers-main
cd transformers-main

# Prefer a clean install; no cache assumptions
python -m pip install .

# -----------------------------------------------------
# Step 11. Sanity check / summary
# -----------------------------------------------------
echo ""
echo "🔍 Verifying key packages..."
python - <<'EOF'
import torch
print("✅ torch:", torch.__version__)
print("✅ torch cuda:", torch.version.cuda)
try:
    import vllm
    print("✅ vllm:", vllm.__version__)
except Exception as e:
    print("⚠️ vllm import failed:", repr(e))
try:
    import trl
    print("✅ trl:", trl.__version__)
except Exception as e:
    print("⚠️ trl import failed:", repr(e))
try:
    import transformers
    print("✅ transformers:", transformers.__version__)
except Exception as e:
    print("⚠️ transformers import failed:", repr(e))
EOF

echo ""
echo "✅ Video-R1 environment setup complete! (reference)"
echo "---------------------------------------------"
echo "Conda env: $ENV_NAME"
echo "Python: $(python --version)"
echo "Repo: $REPO_DIR"
echo "Torch CUDA (reported): $(python -c 'import torch; print(torch.version.cuda)')"
echo "---------------------------------------------"
