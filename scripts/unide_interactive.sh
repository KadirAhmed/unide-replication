#!/bin/bash

# ---
# This script starts an interactive Slurm GPU session, ready to run UniDE.
# It takes two or three arguments:
# 1. GPU_NUMBER: The number of GPUs to request (UniDE uses 1)
# 2. HOURS: The number of hours to request (e.g., 2, 5)
# 3. MEMORY (optional): CPU memory to request, default 32G
# ---

# --- 0. Settings: check these once ---
PROJECT_DIR="/scratch3/$USER/unide"                     # the unide folder on scratch3
ACCOUNT="OD-214219"                                     # your O2D project code (see: get_project_codes)

# --- 1. Check for correct number of arguments ---
if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
    echo "Error: Incorrect number of arguments."
    echo "Usage: $0 <gpu_number> <hours> [memory]"
    echo "Example: $0 1 2"
    echo "(This requests 1 GPU for 2 hours with 32G of memory)"
    exit 1
fi

# --- 2. Assign arguments to descriptive variable names ---
GPU_NUMBER="$1"
HOURS="$2"
MEMORY="${3:-32G}"

# --- 3. Run the setup commands ---

echo "Changing directory to $PROJECT_DIR ..."
cd "$PROJECT_DIR" || { echo "Error: folder not found. Edit PROJECT_DIR at the top of $0"; exit 1; }

# Load the Python module BEFORE activating the virtual environment, so that `python`
# points to the environment's Python (loading the module afterwards would override it).
echo "Loading Python module..."
module load python/3.12.0

echo "Activating virtual environment (.venv)..."
source .venv/bin/activate || { echo "Error: .venv not found. Create it first (setup step)."; exit 1; }

export HF_HOME="$PROJECT_DIR/hf_cache"            # models downloaded earlier on the login node
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1    # compute nodes may have no internet access

# --- 4. Construct and run the sinteractive command ---

TIME_REQUEST="${HOURS}:00:00"
GPU_REQUEST="gpu:${GPU_NUMBER}"

echo "------------------------------------------------"
echo "Requesting interactive session with:"
echo "  Account: ${ACCOUNT}"
echo "  GPUs:    ${GPU_REQUEST}"
echo "  Time:    ${TIME_REQUEST}"
echo "  Memory:  ${MEMORY}"
echo "------------------------------------------------"
echo "When the session starts, run the quick test:"
echo "  python -m unide.train --train data/unide_data/train.jsonl --val data/unide_data/val.jsonl \\"
echo "      --backbone facebook/opt-1.3b --out runs/test_main --bf16 --epochs 1 --max_train 3000 --max_val 500"
echo "  python -m unide.evaluate --model runs/test_main --benchmark fed --data data/bench/fed.jsonl"
echo "------------------------------------------------"

sinteractive -A "$ACCOUNT" -n1 -m"$MEMORY" -g "$GPU_REQUEST" -t "$TIME_REQUEST"

echo "Interactive session ended."
