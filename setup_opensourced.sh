#!/usr/bin/env bash
set -e

echo "Setting up CapNav open-source environment"

# ----------------------------
# 1. Check conda availability
# ----------------------------
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed."
    echo "Please install Miniconda or Anaconda before proceeding."
    exit 1
fi

# ----------------------------
# 2. Create and activate env
# ----------------------------
ENV_NAME=CapNav

if conda env list | grep -q "^${ENV_NAME}\s"; then
    echo "Conda environment '${ENV_NAME}' already exists."
else
    echo "Creating conda environment '${ENV_NAME}'"
    conda create -n ${ENV_NAME} python=3.10 -y
fi

# Activate environment (non-interactive safe)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

# ----------------------------
# 3. Install dependencies
# ----------------------------
echo "Installing Python dependencies"

pip install --upgrade pip

pip install torch torchvision torchaudio
pip install transformers
pip install einops timm
pip install decord
pip install accelerate
pip install av
pip install tiktoken

# ----------------------------
# 4. Done
# ----------------------------
echo "CapNav environment setup complete."
echo "Please ensure your CUDA / GPU configuration matches your local system."
