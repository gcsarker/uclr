#!/bin/bash
#SBATCH --job-name=UCLR
#SBATCH --propagate=NONE
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem=80G
#SBATCH --output=/network/rit/dgx/dgx_dahlin_lab/Gobinda/uclr/logs/slurm/slurm_%j.out

# Usage: sbatch train.sh <ENV_ID> <script.py> [model_name] [seed] [kmeans_k]
#   ENV_ID      Hopper-v5 | Walker2d-v5 | HalfCheetah-v5 | Ant-v5
#   script.py   training script in the repo root (baseline.py, ldr.py, kmeans_fn.py, ...)
#   model_name  log folder / tag name (default: script name without .py)
#   seed        training seed (default: LDRConfig.seed); pass "" to skip
#   kmeans_k    number of K-means clusters, kmeans_fn.py only (default: LDRConfig.kmeans_k)
#
# Examples:
#   sbatch train.sh Hopper-v5 baseline.py                          # defaults
#   sbatch train.sh Walker2d-v5 ldr.py                             # defaults
#   sbatch train.sh Walker2d-v5 ldr.py ldr_dpgmm                   # custom model name
#   sbatch train.sh Ant-v5 ldr.py ldr 1000                         # seed 1000
#   sbatch train.sh Ant-v5 kmeans_fn.py kmeans_fn 1000 64          # seed 1000, K=64
#   sbatch train.sh Hopper-v5 kmeans_fn.py kmeans_fn "" 32         # default seed, K=32
#   sbatch --time=24:00:00 train.sh Ant-v5 kmeans_fn.py            # longer time limit
#
# K sweep:
#   for k in 4 8 16 32 64 128; do sbatch train.sh Ant-v5 kmeans_fn.py kmeans_fn "" "$k"; done
set -e

ENV_ID=${1:?usage: sbatch train.sh <ENV_ID> <script.py> [model_name] [seed] [kmeans_k]}
SCRIPT=${2:?usage: sbatch train.sh <ENV_ID> <script.py> [model_name] [seed] [kmeans_k]}
MODEL_NAME=${3:-$(basename "$SCRIPT" .py)}
SEED=${4:-}
KMEANS_K=${5:-}

case "$ENV_ID" in
    Hopper-v5|Walker2d-v5|HalfCheetah-v5|Ant-v5) ;;
    *) echo "unknown ENV_ID '$ENV_ID'" >&2; exit 1 ;;
esac

# Run from the repo root, so the script path resolves and checkpoints/ lands in the repo
WORK_DIR=/network/rit/dgx/dgx_dahlin_lab/Gobinda/uclr
cd "$WORK_DIR"

if [ ! -f "$SCRIPT" ]; then
    echo "script '$SCRIPT' not found in $WORK_DIR" >&2; exit 1
fi

ENV_NAME=$(echo "${ENV_ID%%-*}" | tr '[:upper:]' '[:lower:]')   # Walker2d-v5 -> walker2d
TAG="${MODEL_NAME}_${ENV_NAME}${KMEANS_K:+_k$KMEANS_K}${SEED:+_seed$SEED}"

# One folder per model: logs/<model>/<model>_<env>_output_<jobid>.out and ..._error_<jobid>.err
LOG_DIR="$WORK_DIR/logs/$MODEL_NAME"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/${TAG}_output_${SLURM_JOB_ID}.out" 2> "$LOG_DIR/${TAG}_error_${SLURM_JOB_ID}.err"

# Show the run's name in squeue as well (harmless if the cluster doesn't allow it)
scontrol update JobId="$SLURM_JOB_ID" JobName="$TAG" 2>/dev/null || true

source /network/rit/dgx/dgx_dahlin_lab/Gobinda/anaconda3/etc/profile.d/conda.sh
conda activate ldr

export ENV_ID                                   # the Python script reads it with os.environ
if [ -n "$SEED" ]; then export SEED; else unset SEED; fi
if [ -n "$KMEANS_K" ]; then export KMEANS_K; else unset KMEANS_K; fi
export LDR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "job $SLURM_JOB_ID | $SCRIPT | $ENV_ID | seed ${SEED:-default} | K ${KMEANS_K:-default} | node $(hostname)"
python -u "$SCRIPT"