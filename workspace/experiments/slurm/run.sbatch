#!/usr/bin/env bash
#SBATCH --job-name=specdec
#SBATCH --partition=gen_pop_rhel8
#SBATCH --qos=asic
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results/slurm-%x-%A_%a.out
# Usage:
#   sbatch --array=0-7 experiments/slurm/run.sbatch experiments/run_scheduler.py
# The array index becomes SHARD_ID; the array size becomes SHARD_COUNT, so each task
# computes a disjoint slice of the sweep (see specdec.get_shard).
set -euo pipefail

SCRIPT="$1"; shift || true

export REPO_WORKSPACE="${REPO_WORKSPACE:-/home/mihika/65930-final/workspace}"
export AF_JOBS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
export SHARD_ID="${SLURM_ARRAY_TASK_ID:-0}"
# SLURM_ARRAY_TASK_COUNT is the number of tasks in the array.
export SHARD_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"

cd "$REPO_WORKSPACE"
echo "[run.sbatch] node=$(hostname) script=$SCRIPT shard=$SHARD_ID/$SHARD_COUNT jobs=$AF_JOBS"
bash experiments/slurm/podman_run.sh "$SCRIPT" "$@" 2>&1 | grep -vi "rootless runtime" || true
