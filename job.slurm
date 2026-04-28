#!/bin/bash
#SBATCH --job-name=sgcrl_non_epi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --gres=gpu:1
#SBATCH --time=5:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wl2527@princeton.edu

module purge
module load anaconda3/2023.3
if module load cudatoolkit/11.3 cudnn/cuda-11.x/8.2.0 2>/dev/null; then
  echo "Loaded Della CUDA 11.3/cuDNN 8.2 modules."
else
  echo "CUDA 11.3/cuDNN 8.2 modules unavailable; relying on conda env libraries."
fi

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BASE_DIR="${BASE_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONDA_ENV="${CONDA_ENV:-${BASE_DIR}/conda_envs/sgcrl}"
BRANCH_DIR="${BRANCH_DIR:-${SCRIPT_DIR}}"

ENV_NAME="${ENV_NAME:-sawyer_box}"
ALG="${ALG:-contrastive_cpc}"
SEED="${SEED:-99887766}"
NUM_STEPS="${NUM_STEPS:-8000000}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

export MUJOCO_PY_MUJOCO_PATH="${HOME}/.mujoco/mujoco210"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${MUJOCO_PY_MUJOCO_PATH}/bin:/usr/lib/nvidia:${LD_LIBRARY_PATH:-}"
if [[ -n "${CUDA_HOME:-}" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export TF_FORCE_GPU_ALLOW_GROWTH=true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_HOME: $CUDA_HOME"
echo "BRANCH_DIR: $BRANCH_DIR"
echo "ENV_NAME: $ENV_NAME"
echo "ALG: $ALG"
echo "SEED: $SEED"
echo "NUM_STEPS: $NUM_STEPS"
echo "EXTRA_ARGS: $EXTRA_ARGS"
nvidia-smi

cd "${BRANCH_DIR}" || exit 1

python -u lp_contrastive.py \
  --seed "${SEED}" \
  --env "${ENV_NAME}" \
  --alg "${ALG}" \
  --num_steps "${NUM_STEPS}" \
  ${EXTRA_ARGS}

exit 0  
