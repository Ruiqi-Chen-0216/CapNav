#!/bin/bash
#SBATCH --job-name=qwen_scene_array
#SBATCH --account=makelab
#SBATCH --partition=gpu-l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=0-08:00:00
#SBATCH --array=0-44%8                  # 45 scenes, 8 at a time
#SBATCH --chdir=/gscratch/makelab/ruiqi/QwenVL
#SBATCH --export=ALL
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ruiqich@uw.edu

set -euo pipefail

echo "=== ENVIRONMENT INFO ==="
hostname
date
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi || echo "⚠️ GPU not visible!"

# ============================================================
# 1️⃣ Activate Conda environment via helper script
# ============================================================
echo "Activating Conda environment..."
if [ -f /gscratch/makelab/ruiqi/setup_hyak_conda.sh ]; then
  source /gscratch/makelab/ruiqi/setup_hyak_conda.sh general
else
  echo "❌ Missing setup_hyak_conda.sh at /gscratch/makelab/ruiqi/"
  exit 1
fi

echo "✅ Python executable: $(which python)"
python --version

# ============================================================
# 2️⃣ Check and prepare model
# ============================================================
MODEL_DIR="/scr/models/Qwen3-VL-8B-Instruct"
if [ ! -d "$MODEL_DIR" ]; then
  echo "📦 Model not found → downloading..."
  bash /gscratch/makelab/ruiqi/download_model.sh Qwen/Qwen3-VL-8B-Instruct
else
  echo "✅ Model already exists: $MODEL_DIR"
fi

# ============================================================
# 3️⃣ Select scene based on array task ID
# ============================================================
mapfile -t SCENE_LIST < <(ls /gscratch/makelab/ruiqi/graph | grep '\-graph.json$' | sed 's/-graph.json//')
TOTAL=${#SCENE_LIST[@]}
if [ "$TOTAL" -eq 0 ]; then
  echo "❌ No scenes found in /gscratch/makelab/ruiqi/graph"
  exit 1
fi

IDX=${SLURM_ARRAY_TASK_ID}
if [ "$IDX" -ge "$TOTAL" ]; then
  echo "❌ TASK_ID=$IDX exceeds total scenes ($TOTAL)"
  exit 1
fi

SCENE="${SCENE_LIST[$IDX]}"
echo "🚀 Launching scene: $SCENE  (index $IDX / $TOTAL)"

# ============================================================
# 4️⃣ Run main inference script for this scene
# ============================================================
python /gscratch/makelab/ruiqi/QwenVL/Qwen3-VL-8B-Instruct.py --scene "$SCENE"

# ============================================================
# 5️⃣ Wrap up
# ============================================================
echo "✅ Finished scene: $SCENE"
date
