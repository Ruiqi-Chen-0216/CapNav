# =====================================================
# Spatial-MLLM Environment Bootstrap (Cluster-Safe + Version-Locked)
# Author: Ruiqi
# =====================================================
#source setup_spatialMLLM.sh

echo "🚀 Setting up Spatial-MLLM environment in /scr"

# -----------------------------------------------------
# Step 1. Move to /scr and prepare base paths
# -----------------------------------------------------
cd /scr

# -----------------------------------------------------
# Step 2. Install Anaconda if needed
# -----------------------------------------------------
CONDA_DIR=/scr/anaconda3_ruiqi
INSTALLER=Anaconda3-2025.06-0-Linux-x86_64.sh

if [ ! -f "$INSTALLER" ]; then
    echo "⬇️ Downloading Anaconda installer..."
    wget https://repo.anaconda.com/archive/$INSTALLER
fi

if [ ! -d "$CONDA_DIR" ]; then
    echo "📦 Installing Anaconda to $CONDA_DIR..."
    bash $INSTALLER -b -p ${CONDA_DIR}
fi

# -----------------------------------------------------
# Step 3. Initialize conda
# -----------------------------------------------------
export PATH=${CONDA_DIR}/bin:${PATH}
$CONDA_DIR/bin/conda init
source ~/.bashrc

# -----------------------------------------------------
# Step 4. Create / Activate environment
# -----------------------------------------------------
if ! conda info --envs | grep -q "spatial-mllm"; then
    echo "🧩 Creating conda environment: spatial-mllm"
    conda create -n spatial-mllm python=3.10 -y
fi
conda activate spatial-mllm

# -----------------------------------------------------
# Step 5. Redirect all pip caches/temp dirs to /scr
# -----------------------------------------------------
export PIP_CACHE_DIR=/scr/pip_cache
export TMPDIR=/scr/tmp
mkdir -p $PIP_CACHE_DIR $TMPDIR

echo "✅ pip cache → $PIP_CACHE_DIR"
echo "✅ tmp build  → $TMPDIR"

# -----------------------------------------------------
# Step 6. Hugging Face cache setup
# -----------------------------------------------------
mkdir -p /scr/ruiqi/{hf_cache,hf_datasets,hf_hub_cache}
mkdir -p /gscratch/makelab/ruiqi/hf_home

export TRANSFORMERS_CACHE=/scr/ruiqi/hf_cache
export HF_HOME=/gscratch/makelab/ruiqi/hf_home
export HF_DATASETS_CACHE=/scr/ruiqi/hf_datasets
export HF_HUB_CACHE=/scr/ruiqi/hf_hub_cache

echo "✅ HF_HOME     → $HF_HOME"
echo "✅ HF_CACHE    → $TRANSFORMERS_CACHE"

# -----------------------------------------------------
# Step 7. Clone Spatial-MLLM repo
# -----------------------------------------------------
cd /scr
if [ ! -d "/scr/Spatial-MLLM" ]; then
    echo "📂 Cloning Spatial-MLLM repository..."
    git clone https://github.com/diankun-wu/Spatial-MLLM
fi
cd /scr/Spatial-MLLM

# -----------------------------------------------------
# Step 8. Install dependencies (version locked)
# -----------------------------------------------------
echo "📦 Installing core dependencies..."

pip install --upgrade pip setuptools wheel

# --- Torch + CUDA ---
# Adjust CUDA version as needed for your cluster
TMPDIR=/scr/tmp PIP_CACHE_DIR=/scr/pip_cache pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# --- Core packages ---
TMPDIR=/scr/tmp PIP_CACHE_DIR=/scr/pip_cache pip install \
    transformers==4.51.3 \
    accelerate==1.5.2 \
    qwen_vl_utils \
    decord \
    ray \
    Levenshtein \
    tyro \
    einops \
    timm \
    av

# --- Optional flash-attn optimization ---
TMPDIR=/scr/tmp PIP_CACHE_DIR=/scr/pip_cache pip install flash-attn --no-build-isolation

# -----------------------------------------------------
# Step 9. Post-install sanity check
# -----------------------------------------------------
echo ""
echo "🔍 Verifying key packages..."
python - <<'EOF'
import torch, transformers, accelerate, decord, Levenshtein, tyro, einops, timm, av
print("✅ torch:", torch.__version__)
print("✅ transformers:", transformers.__version__)
print("✅ accelerate:", accelerate.__version__)
print("✅ einops:", einops.__version__)
print("✅ timm:", timm.__version__)
EOF

# -----------------------------------------------------
# Step 10. Summary
# -----------------------------------------------------
echo ""
echo "✅ Spatial-MLLM environment setup complete!"
echo "---------------------------------------------"
echo "Conda env: spatial-mllm"
echo "Python: $(python --version)"
echo "PIP cache: $PIP_CACHE_DIR"
echo "TMPDIR: $TMPDIR"
echo "HF_HOME: $HF_HOME"
echo "Torch CUDA: $(python -c 'import torch; print(torch.version.cuda)')"
echo "---------------------------------------------"
