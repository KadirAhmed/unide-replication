#!/usr/bin/env bash
# End-to-end record of the UniDE reproduction (Ye et al., IP&M 2025), in the order it was run.
# Needs OPENAI_API_KEY for steps 1-2 and one GPU with 40 GB+ for step 4. Each step can be run on its own.
set -euo pipefail

# ---- 1. task division & representative dimensions (Sec. 3.3), optional: the paper's six dimensions are built in
python -m unide.data_generation generate --n 400 --out data/pilot_raw.jsonl
python -m unide.data_generation verify   --inp data/pilot_raw.jsonl --out data/pilot_verified.jsonl
python -m unide.task_grouping annotate --inp data/pilot_verified.jsonl --n 300 --out data/sample12.jsonl
python -m unide.task_grouping group    --inp data/sample12.jsonl --out_dir data/grouping --plot
python -m unide.task_grouping template --inp data/sample12.jsonl --out data/grouping/overall_scores.csv
echo ">> Fill in data/grouping/overall_scores.csv (human Flu/Coh/Eng overall scores), then run:"
echo "   python -m unide.task_grouping weights --inp data/sample12.jsonl --overall data/grouping/overall_scores.csv"

# ---- 2. UniDE-data (Sec. 3.2): generate with gpt-3.5-turbo-0125, verify with GPT-4 Turbo,
#         drop labels the verifier contradicts (bad <-> good)
python -m unide.data_generation generate --n 8200 --out data/raw_generated.jsonl --workers 2
python -m unide.data_generation verify   --inp data/raw_generated.jsonl --out data/verified.jsonl --model gpt-4-turbo-2024-04-09 --workers 3
python -m unide.data_generation build    --inp data/verified.jsonl --out_dir data/unide_data --n_val 500
python -m unide.data_generation build    --inp data/verified.jsonl --out_dir data/unide_data_nofilter --filter none --n_val 500

# ---- 3. benchmarks: FED and PersonaChat (test), Topical-USR (development set for choosing the checkpoint)
mkdir -p data/raw data/bench
curl -L -o data/raw/fed_data.json    http://shikib.com/fed_data.json
curl -L -o data/raw/pc_usr_data.json https://shikib.com/pc_usr_data.json
curl -L -o data/raw/tc_usr_data.json https://shikib.com/tc_usr_data.json
curl -L -o data/raw/see_logs_v1.tar.gz https://parl.ai/downloads/controllable_dialogue/evaluationlogs_v1.tar.gz
mkdir -p data/raw/see_logs_v1 && tar -xzf data/raw/see_logs_v1.tar.gz -C data/raw/see_logs_v1
python -m unide.benchmarks fed    data/raw/fed_data.json    data/bench/fed.jsonl
python -m unide.benchmarks usr    data/raw/pc_usr_data.json data/bench/persona_usr.jsonl
python -m unide.benchmarks usr    data/raw/tc_usr_data.json data/bench/tc_usr.jsonl
python -m unide.benchmarks parlai data/raw/see_logs_v1      data/bench/persona_see.jsonl

# ---- 4. training + evaluation (Tables 2-4, 7-10, Fig. 7) with the final protocol, then one results table
bash scripts/run_local.sh all || true          # failed configs are reported; the rest still run
SEED=43 bash scripts/run_local.sh 0 || true    # two more seeds of the main model, to gauge run-to-run noise
SEED=44 bash scripts/run_local.sh 0 || true
bash scripts/run_local.sh vanilla || true      # untrained backbones
python -m unide.summarize
