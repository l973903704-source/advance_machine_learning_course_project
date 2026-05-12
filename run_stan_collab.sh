#!/bin/bash
#SBATCH --job-name=stan_collab
#SBATCH --partition=gbunchQ
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=slurm-%j_stan_collab.log
#SBATCH --error=slurm-%j_stan_collab.err

echo "HOSTNAME: $(hostname)"
echo "DATE: $(date)"
echo "PWD(before cd): $(pwd)"

source "${CONDA_ROOT:-$HOME/miniconda3}/bin/activate" "${CONDA_ENV:-ai_study}"

export HF_HOME=$HOME/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export XDG_CACHE_HOME=$HOME/.cache
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8

unset HF_DATASETS_CACHE_DIR
unset HUGGINGFACE_CACHE
unset TRANSFORMERS_OFFLINE

PROJECT_DIR=${PROJECT_DIR:-/path/to/your/project}
RUN_DIR=${RUN_DIR:-${PROJECT_DIR}/test}

if [ "${PROJECT_DIR}" = "/path/to/your/project" ]; then
    echo "Please set PROJECT_DIR before submitting this job."
    echo "Example: PROJECT_DIR=/your/project/path sbatch run_stan_collab.sh neighbor NYC_min20 0 30"
    exit 1
fi

mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}" || exit 1

echo "PWD(after cd): $(pwd)"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "RUN_DIR=${RUN_DIR}"
echo "Python: $(which python)"
python --version

echo "CUDA check:"
nvidia-smi || true

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY

VARIANT=${1:-proto}
DNAME=${2:-NYC}
PART=${3:-0}
EPOCHS=${4:-30}
LR=${LR:-1e-3}
NUM_NEG=${NUM_NEG:-10}
COLLAB_WEIGHT=${COLLAB_WEIGHT:-0.5}
AUX_WEIGHT=${AUX_WEIGHT:-0.02}
SEED=${SEED:-0}
BEST_METRIC=${BEST_METRIC:-r5}
LOG_C_EVERY=${LOG_C_EVERY:-1000}
WARN_C_ABS=${WARN_C_ABS:-2.0}
WARN_C_RATIO=${WARN_C_RATIO:-2.0}
C_CLIP=${C_CLIP:-0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0}
LR_STEP=${LR_STEP:-1000}
LR_GAMMA=${LR_GAMMA:-1.0}
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-0}
SAVE_PREFIX=${SAVE_PREFIX:-best_${VARIANT}_stan}
TOP_K=${TOP_K:-16}
MOMENTUM=${MOMENTUM:-0.9}
NEIGHBOR_TEMPERATURE=${NEIGHBOR_TEMPERATURE:-1.0}
CONTRASTIVE_TEMPERATURE=${CONTRASTIVE_TEMPERATURE:-0.1}

echo "VARIANT=${VARIANT}"
echo "DNAME=${DNAME}"
echo "PART=${PART}"
echo "EPOCHS=${EPOCHS}"
echo "LR=${LR}"
echo "NUM_NEG=${NUM_NEG}"
echo "COLLAB_WEIGHT=${COLLAB_WEIGHT}"
echo "AUX_WEIGHT=${AUX_WEIGHT}"
echo "SEED=${SEED}"
echo "BEST_METRIC=${BEST_METRIC}"
echo "LOG_C_EVERY=${LOG_C_EVERY}"
echo "C_CLIP=${C_CLIP}"
echo "SAVE_PREFIX=${SAVE_PREFIX}"

if [ ! -f "${PROJECT_DIR}/data/${DNAME}_data.pkl" ]; then
    echo "Missing ${PROJECT_DIR}/data/${DNAME}_data.pkl, running load.py ..."
    cd "${PROJECT_DIR}" || exit 1
    python -u load.py --dname ${DNAME}
    cd "${RUN_DIR}" || exit 1
fi

echo "Files under project data/:"
ls -lh "${PROJECT_DIR}/data"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

if [ "${VARIANT}" = "proto" ]; then
    python -u "${PROJECT_DIR}/train_proto.py" \
        --dname ${DNAME} \
        --part ${PART} \
        --epochs ${EPOCHS} \
        --embed_dim 50 \
        --lr ${LR} \
        --num_neg ${NUM_NEG} \
        --collab_weight ${COLLAB_WEIGHT} \
        --momentum ${MOMENTUM} \
        --seed ${SEED} \
        --save_prefix ${SAVE_PREFIX} \
        --best_metric ${BEST_METRIC} \
        --log_c_every ${LOG_C_EVERY} \
        --warn_c_abs ${WARN_C_ABS} \
        --warn_c_ratio ${WARN_C_RATIO} \
        --c_clip ${C_CLIP} \
        --weight_decay ${WEIGHT_DECAY} \
        --lr_step ${LR_STEP} \
        --lr_gamma ${LR_GAMMA} \
        --early_stop_patience ${EARLY_STOP_PATIENCE}

elif [ "${VARIANT}" = "neighbor" ]; then
    python -u "${PROJECT_DIR}/train_neighbor.py" \
        --dname ${DNAME} \
        --part ${PART} \
        --epochs ${EPOCHS} \
        --embed_dim 50 \
        --lr ${LR} \
        --num_neg ${NUM_NEG} \
        --collab_weight ${COLLAB_WEIGHT} \
        --top_k ${TOP_K} \
        --neighbor_temperature ${NEIGHBOR_TEMPERATURE} \
        --momentum ${MOMENTUM} \
        --seed ${SEED} \
        --save_prefix ${SAVE_PREFIX} \
        --best_metric ${BEST_METRIC} \
        --log_c_every ${LOG_C_EVERY} \
        --warn_c_abs ${WARN_C_ABS} \
        --warn_c_ratio ${WARN_C_RATIO} \
        --c_clip ${C_CLIP} \
        --weight_decay ${WEIGHT_DECAY} \
        --lr_step ${LR_STEP} \
        --lr_gamma ${LR_GAMMA} \
        --early_stop_patience ${EARLY_STOP_PATIENCE}

elif [ "${VARIANT}" = "contrastive" ]; then
    python -u "${PROJECT_DIR}/train_contrastive.py" \
        --dname ${DNAME} \
        --part ${PART} \
        --epochs ${EPOCHS} \
        --embed_dim 50 \
        --lr ${LR} \
        --num_neg ${NUM_NEG} \
        --collab_weight ${COLLAB_WEIGHT} \
        --aux_weight ${AUX_WEIGHT} \
        --contrastive_temperature ${CONTRASTIVE_TEMPERATURE} \
        --seed ${SEED} \
        --save_prefix ${SAVE_PREFIX} \
        --best_metric ${BEST_METRIC} \
        --log_c_every ${LOG_C_EVERY} \
        --warn_c_abs ${WARN_C_ABS} \
        --warn_c_ratio ${WARN_C_RATIO} \
        --c_clip ${C_CLIP} \
        --weight_decay ${WEIGHT_DECAY} \
        --lr_step ${LR_STEP} \
        --lr_gamma ${LR_GAMMA} \
        --early_stop_patience ${EARLY_STOP_PATIENCE}

else
    echo "Unknown VARIANT: ${VARIANT}"
    echo "Use one of: proto | neighbor | contrastive"
    exit 1
fi
