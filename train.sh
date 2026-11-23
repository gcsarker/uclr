#!/bin/bash
#SBATCH --job-name=LDR_Training
#SBATCH --propagate=NONE
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=16:00:00
#SBATCH --mem=80G
# SLURM's own log: only messages from SLURM itself (time limit, out of memory, early errors) land here
#SBATCH --output=/network/rit/dgx/dgx_dahlin_lab/Gobinda/Latent-Dynamics-Geometries/batch_training/baseline_batch_training/logs/slurm_%j.out
#SBATCH --error=/network/rit/dgx/dgx_dahlin_lab/Gobinda/Latent-Dynamics-Geometries/batch_training/baseline_batch_training/logs/slurm_%j.err

# Usage: sbatch train.sh <ENV_ID> <script.py> [model_name]
#   sbatch train.sh Hopper-v5 baseline_hopper.py
#   sbatch train.sh Walker2d-v5 ldr.py ldr_dpgmm
set -e

ENV_ID=${1:?usage: sbatch train.sh <ENV_ID> <script.py> [model_name]}
SCRIPT=${2:?usage: sbatch train.sh <ENV_ID> <script.py> [model_name]}
MODEL_NAME=${3:-$(basename "$SCRIPT" .py)}

case "$ENV_ID" in
    Hopper-v5|Walker2d-v5|HalfCheetah-v5|Ant-v5) ;;
    *) echo "unknown ENV_ID '$ENV_ID'" >&2; exit 1 ;;
esac

WORK_DIR=/network/rit/dgx/dgx_dahlin_lab/Gobinda/Latent-Dynamics-Geometries/batch_training/baseline_batch_training
cd "$WORK_DIR"

ENV_NAME=$(echo "${ENV_ID%%-*}" | tr '[:upper:]' '[:lower:]')   # Walker2d-v5 -> walker2d
TAG="${MODEL_NAME}_${ENV_NAME}"

# From here on, everything goes to logs/<model>_<env>_output_<jobid>.out and ..._error_<jobid>.err
LOG_DIR="$WORK_DIR/logs"
mkdir -p "$LOG_DIR"
exec > "$LOG_DIR/${TAG}_output_${SLURM_JOB_ID}.out" 2> "$LOG_DIR/${TAG}_error_${SLURM_JOB_ID}.err"

# Show the run's name in squeue as well (harmless if the cluster doesn't allow it)
scontrol update JobId="$SLURM_JOB_ID" JobName="$TAG" 2>/dev/null || true

source /network/rit/dgx/dgx_dahlin_lab/Gobinda/anaconda3/etc/profile.d/conda.sh
conda activate ldr

export ENV_ID                                   # the Python script reads it with os.environ
export LDR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "job $SLURM_JOB_ID | $SCRIPT | $ENV_ID | node $(hostname)"
python -u "$SCRIPT"