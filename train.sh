#!/bin/bash
#SBATCH --job-name=UCLR
#SBATCH --propagate=NONE
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem=80G
#SBATCH --output=/network/rit/dgx/dgx_dahlin_lab/Gobinda/uclr/logs/slurm/slurm_%j.out

# Usage: sbatch train.sh <ENV_ID> <script.py> [model_name] [seed]
#   sbatch train.sh Hopper-v5 baseline.py
#   sbatch train.sh Ant-v5 kmeans_fn.py kmeans_fn 1000
set -e

ENV_ID=${1:?usage: sbatch train.sh <ENV_ID> <script.py> [model_name]}
SCRIPT=${2:?usage: sbatch train.sh <ENV_ID> <script.py> [model_name]}
MODEL_NAME=${3:-$(basename "$SCRIPT" .py)}
SEED=${4:-}

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
TAG="${MODEL_NAME}_${ENV_NAME}"

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
export LDR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "job $SLURM_JOB_ID | $SCRIPT | $ENV_ID | seed ${SEED:-default} | node $(hostname)"
python -u "$SCRIPT"