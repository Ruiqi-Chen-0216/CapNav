#source setup_videor1.sh

#!/bin/bash
# =====================================================
# Video-R1 Environment Bootstrap (Cluster-Safe + Version-Locked)
# Author: Ruiqi
# =====================================================

echo "🚀 Setting up environment in /scr"

# --- Step 1. Move to /scr and prepare base paths ---
cd /scr

# --- Step 2. Install Anaconda if needed ---
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

# --- Step 3. Initialize conda ---
export PATH=${CONDA_DIR}/bin:${PATH}
$CONDA_DIR/bin/conda init
source ~/.bashrc

# --- Step 4. Create / Activate environment ---
if ! conda info --envs | grep -q "video-r1"; then
    echo "🧩 Creating conda environment: video-r1"
    conda create -n video-r1 python=3.11 -y
fi
conda activate video-r1

# =====================================================
# Step 5. Redirect all pip caches/temp dirs to /scr
# =====================================================
export PIP_CACHE_DIR=/scr/pip_cache
export TMPDIR=/scr/tmp
mkdir -p $PIP_CACHE_DIR $TMPDIR

echo "✅ pip cache → $PIP_CACHE_DIR"
echo "✅ tmp build  → $TMPDIR"

# =====================================================
# Step 6. Hugging Face cache setup
# =====================================================
mkdir -p /scr/ruiqi/{hf_cache,hf_datasets,hf_hub_cache}
mkdir -p /gscratch/makelab/ruiqi/hf_home

export TRANSFORMERS_CACHE=/scr/ruiqi/hf_cache
export HF_HOME=/gscratch/makelab/ruiqi/hf_home
export HF_DATASETS_CACHE=/scr/ruiqi/hf_datasets
export HF_HUB_CACHE=/scr/ruiqi/hf_hub_cache

echo "✅ HF_HOME     → $HF_HOME"
echo "✅ HF_CACHE    → $TRANSFORMERS_CACHE"

# =====================================================
# Step 7. Install essential base packages
# =====================================================
echo "📦 Installing base dependencies..."

pip install --upgrade pip setuptools wheel
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install einops timm decord accelerate av tiktoken

# =====================================================
# Step 8. Clone and setup Video-R1
# =====================================================
cd /scr
if [ ! -d "/scr/Video-R1" ]; then
    git clone https://github.com/tulerfeng/Video-R1
fi
cd /scr/Video-R1

# ⚠️ Run Video-R1 setup.sh under controlled environment
echo "🔧 Running Video-R1/setup.sh (pip cache in /scr)"
TMPDIR=/scr/tmp PIP_CACHE_DIR=/scr/pip_cache bash setup.sh

# =====================================================
# Step 9. Build Qwen-VL Utils
# =====================================================
echo "🔧 Setting up Qwen-VL Utils..."
cd /scr/Video-R1/src/qwen-vl-utils
TMPDIR=/scr/tmp PIP_CACHE_DIR=/scr/pip_cache pip install -e .[decord]

# =====================================================
# Step 10. Version fixes (ensure compatibility)
# =====================================================
echo "📌 Fixing critical library versions..."
TMPDIR=/scr/tmp PIP_CACHE_DIR=/scr/pip_cache pip install vllm==0.7.2
TMPDIR=/scr/tmp PIP_CACHE_DIR=/scr/pip_cache pip install trl==0.16.0

# =====================================================
# Step 11. Local Transformers install (after Video-R1 setup)
# =====================================================
echo "📦 Installing local transformers-main.zip"
cd /scr/Video-R1/src
cp /gscratch/makelab/ruiqi/M_Video-R1/transformers-main.zip .
unzip -o transformers-main.zip
cd transformers-main
TMPDIR=/scr/tmp PIP_CACHE_DIR=/scr/pip_cache pip install .

# =====================================================
# Step 12. Verify environment
# =====================================================
echo ""
echo "✅ Environment setup complete!"
echo "---------------------------------------------"
echo "Conda env: video-r1"
echo "Python: $(python --version)"
echo "PIP cache: $PIP_CACHE_DIR"
echo "TMPDIR: $TMPDIR"
echo "HF_HOME: $HF_HOME"
echo "vLLM version: $(python -c 'import vllm; print(vllm.__version__)')"
echo "TRL version:  $(python -c 'import trl; print(trl.__version__)')"
echo "---------------------------------------------"
