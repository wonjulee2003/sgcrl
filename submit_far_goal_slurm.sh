#!/usr/bin/env bash
set -euo pipefail

# Submit the first far-goal A/B matrix:
#   episodic      = true simulator reset every CHUNK_STEPS
#   non_episodic  = fake replay boundary every CHUNK_STEPS, simulator continues
#
# NUM_STEPS controls the total actor-step budget. CHUNK_STEPS controls the
# per-run horizon in episodic mode and the replay chunk length in non-episodic
# mode. With CHUNK_STEPS=25, these fixed goals are intentionally farther than
# one episodic run, so reset-at-horizon SGCRL should be structurally stressed.

SBATCH_SCRIPT="${SBATCH_SCRIPT:-job.slurm}"
ALG="${ALG:-contrastive_cpc}"
NUM_STEPS="${NUM_STEPS:-500000}"
CHUNK_STEPS="${CHUNK_STEPS:-25}"
SEEDS="${SEEDS:-0 1 2}"
MODES="${MODES:-episodic non_episodic}"
LOG_ROOT="${LOG_ROOT:-logs/far_goal}"
SNAPSHOT_EVERY="${SNAPSHOT_EVERY:-0}"

mkdir -p "${LOG_ROOT}"

ENVS=(
  "point_Spiral11x11"
  "point_Maze11x11"
  "point_Wall11x11"
)
STARTS=(
  "5,5"
  "0,0"
  "2,0"
)
GOALS=(
  "10,10"
  "0,10"
  "0,0"
)
TASKS=(
  "spiral_far"
  "maze_far"
  "wall_far"
)

for idx in "${!ENVS[@]}"; do
  env_name="${ENVS[$idx]}"
  start="${STARTS[$idx]}"
  goal="${GOALS[$idx]}"
  task="${TASKS[$idx]}"

  for mode in ${MODES}; do
    for seed in ${SEEDS}; do
      log_dir="${LOG_ROOT}/${task}/${mode}/seed${seed}/"
      mkdir -p "${log_dir}"
      extra_args="--episode_mode ${mode} --chunk_steps ${CHUNK_STEPS} --fixed_start ${start} --fixed_goal ${goal} --log_dir_path ${log_dir} --maze_traj_snapshot_every ${SNAPSHOT_EVERY}"
      job_name="fg_${task}_${mode}_s${seed}"

      echo "Submitting ${job_name}: env=${env_name} start=${start} goal=${goal} steps=${NUM_STEPS} chunk=${CHUNK_STEPS}"
      ENV_NAME="${env_name}" \
      ALG="${ALG}" \
      SEED="${seed}" \
      NUM_STEPS="${NUM_STEPS}" \
      EXTRA_ARGS="${extra_args}" \
        sbatch --job-name "${job_name}" "${SBATCH_SCRIPT}"
    done
  done
done
