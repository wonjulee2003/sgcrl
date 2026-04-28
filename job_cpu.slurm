#!/bin/bash
#SBATCH --job-name=sgcrl_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=wl2527@princeton.edu

module purge
module load anaconda3/2023.3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${MUJOCO_PY_MUJOCO_PATH}/bin:${LD_LIBRARY_PATH:-}"
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

echo "CONDA_PREFIX: $CONDA_PREFIX"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "BRANCH_DIR: $BRANCH_DIR"
echo "ENV_NAME: $ENV_NAME"
echo "ALG: $ALG"
echo "SEED: $SEED"
echo "NUM_STEPS: $NUM_STEPS"
echo "EXTRA_ARGS: $EXTRA_ARGS"

cd "${BRANCH_DIR}" || exit 1

python -u lp_contrastive.py \
  --seed "${SEED}" \
  --env "${ENV_NAME}" \
  --alg "${ALG}" \
  --num_steps "${NUM_STEPS}" \
  ${EXTRA_ARGS}

exit 0
