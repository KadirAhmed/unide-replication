#!/bin/bash
# UniDE on the Virga cluster (SLURM). Each array task runs one configuration of scripts/run_local.sh, so the
# cluster and a rented GPU use exactly the same protocol. Submit from the project folder.
#
# Tasks: 0 main | 1 wo_filter | 2 wo_multitask | 3 score_prompt | 4 mc_prompt | 5 level_groups
#        6 best_weights | 7 two_level | 8 roberta_large | 9 bart_large | 10 untrained backbones
#
#   mkdir -p logs
#   sbatch --export=ALL,TEST=1 -p h2gpu --time=01:30:00 --array=0 scripts/unide_array.sbatch   # quick check
#   sbatch --array=0 scripts/unide_array.sbatch       # main model
#   sbatch --array=1-10 scripts/unide_array.sbatch    # everything else
#SBATCH --job-name=unide
#SBATCH --account=OD-214219
#SBATCH --partition=gpu
#SBATCH --array=0-10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
command -v module >/dev/null 2>&1 || source /etc/profile.d/modules.sh 2>/dev/null || true
module load python/3.12.0
export HF_HOME="${HF_HOME:-$SLURM_SUBMIT_DIR/hf_cache}"   # models pre-downloaded here from the login node
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1          # compute nodes may have no internet access

TASK="$SLURM_ARRAY_TASK_ID"
if [ "$TASK" = "10" ]; then TASK=vanilla; fi
bash scripts/run_local.sh "$TASK"
