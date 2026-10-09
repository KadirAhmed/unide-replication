#!/bin/bash
# Run UniDE training + evaluation on a single GPU machine (e.g. a rented cloud GPU), without SLURM.
# Run it inside tmux so a dropped SSH connection does not stop it.
#
# Final protocol (chosen on the Topical-USR development set, never on FED / PersonaChat):
#   at most 1 epoch, learning rate 5e-6, Topical-USR check every 50 steps, keep the best checkpoint,
#   stop after 10 checks without improvement. Every run is evaluated on FED and PersonaChat with the
#   paper's scoring (argmax) and continuous scoring (expected).
#
#   bash scripts/run_local.sh 0                  # main model
#   bash scripts/run_local.sh 1 2 3              # several configs, one after another
#   bash scripts/run_local.sh all                # every config: main, ablations, other backbones
#   bash scripts/run_local.sh vanilla            # untrained backbones (the paper's RQ1 baselines)
#   SEED=43 bash scripts/run_local.sh 0          # same config, another random seed (run name gets _s43)
#   TEST=1 bash scripts/run_local.sh 0           # quick check on a small slice of the data
#   python -m unide.summarize                    # one table with all results
#
# Another backbone for the OPT configs (run names get _TAG so nothing is overwritten), e.g. the DeepSpeed-Chat
# OPT-1.3B RLHF actor, the public recipe closest to the paper's "InstructGPT (1.3B)":
#   BACKBONE=AdamG012/chat-opt-1.3b-rlhf-actor-ema-deepspeed TOKENIZER=facebook/opt-1.3b TAG=rlhf \
#     bash scripts/run_local.sh vanilla 0        # untrained + main config with that backbone
#
# Configs: 0 main | 1 wo_filter | 2 wo_multitask | 3 score_prompt | 4 mc_prompt | 5 level_groups
#          6 best_weights | 7 two_level | 8 roberta_large | 9 bart_large
# A config that fails is reported and skipped; the remaining configs still run.
set -uo pipefail
cd "$(dirname "$0")/.."
if [ -f .venv/bin/activate ]; then source .venv/bin/activate; fi
mkdir -p logs runs

USER_BACKBONE="${BACKBONE:-}"
BACKBONE="${BACKBONE:-facebook/opt-1.3b}"
TOKENIZER="${TOKENIZER:-}"
TAG="${TAG:-}"
SEED="${SEED:-42}"
PROTOCOL="--epochs 1 --max_val 2000 --dev_data data/bench/tc_usr.jsonl --eval_every 50 --dev_patience 10 --lr 5e-6"
#        data dir            run name       extra training flags
CONFIGS=(
  "unide_data          main          "
  "unide_data_nofilter wo_filter     "
  "unide_data          wo_multitask  --grouping single"
  "unide_data          score_prompt  --prompt score"
  "unide_data          mc_prompt     --prompt mc"
  "unide_data          level_groups  --grouping level"
  "unide_data          best_weights  --task_weights 0.29,0.36,0.35"
  "unide_data          two_level     --two_level"
  "unide_data          roberta_large --backbone FacebookAI/roberta-large"
  "unide_data          bart_large    --backbone facebook/bart-large"
)

evaluate_all () {  # $1 = "--model DIR" or "--vanilla NAME", $2 = output dir, $3 = extra evaluation flags
  local model_flag="$1" out="$2" extra="${3:-}" rc=0
  for mode in argmax expected; do
    python -m unide.evaluate $model_flag --benchmark fed --data data/bench/fed.jsonl \
      --score_mode $mode --out "$out/fed_${mode}.json" $extra || rc=1
    python -m unide.evaluate $model_flag --benchmark personachat \
      --data data/bench/persona_usr.jsonl data/bench/persona_see.jsonl \
      --score_mode $mode --out "$out/personachat_${mode}.json" $extra || rc=1
  done
  return $rc
}

# ---- check the request before spending any GPU time
if [ "$#" -eq 0 ]; then
  echo "Usage: [SEED=n] [TEST=1] [BACKBONE=.. TOKENIZER=.. TAG=..] bash $0 <config number(s) | all | vanilla>"
  exit 1
fi
RUN_VANILLA=0; IDS=()
for a in "$@"; do
  case "$a" in
    vanilla) RUN_VANILLA=1 ;;
    all) IDS+=(0 1 2 3 4 5 6 7 8 9) ;;
    *) if ! [[ "$a" =~ ^[0-9]+$ ]] || [ "$a" -ge "${#CONFIGS[@]}" ]; then
         echo "Unknown config '$a'. Use 0-9, 'all' or 'vanilla'."; exit 1
       fi
       IDS+=("$a") ;;
  esac
done
if [ -n "$USER_BACKBONE" ] && [ -z "$TAG" ]; then
  echo "With BACKBONE set, also set TAG (e.g. TAG=rlhf) so the runs do not overwrite the default-backbone runs."
  exit 1
fi
if [ "${#IDS[@]}" -gt 0 ]; then
  if [ ! -f data/bench/tc_usr.jsonl ]; then
    echo "Missing data/bench/tc_usr.jsonl (the Topical-USR development set). Create it with:"
    echo "  curl -L -o data/raw/tc_usr_data.json https://shikib.com/tc_usr_data.json"
    echo "  python -m unide.benchmarks usr data/raw/tc_usr_data.json data/bench/tc_usr.jsonl"
    exit 1
  fi
fi
for f in data/bench/fed.jsonl data/bench/persona_usr.jsonl data/bench/persona_see.jsonl; do
  if [ ! -f "$f" ]; then echo "Missing $f (see the benchmarks step)."; exit 1; fi
done

nvidia-smi --query-gpu=name,memory.total --format=csv || true
python -c "import torch; print('torch', torch.__version__, '| CUDA available:', torch.cuda.is_available())"
FAILED=()

# ---- untrained backbones (the three of the paper, or only BACKBONE when it is set)
if [ "$RUN_VANILLA" = "1" ]; then
  if [ -n "$USER_BACKBONE" ]; then VANILLA=("$USER_BACKBONE"); else
    VANILLA=(facebook/opt-1.3b FacebookAI/roberta-large facebook/bart-large); fi
  for B in "${VANILLA[@]}"; do
    NAME="vanilla_$(basename "$B")"; OUT="runs/$NAME"; mkdir -p "$OUT"; echo "$B" > "$OUT/backbone.txt"
    TOK_FLAG=""
    if [ -n "$TOKENIZER" ] && [ "$B" = "$USER_BACKBONE" ]; then TOK_FLAG="--tokenizer $TOKENIZER"; fi
    echo "=== untrained $B  (output: $OUT)  started $(date)"
    if evaluate_all "--vanilla $B $TOK_FLAG" "$OUT" 2>&1 | tee "logs/$NAME.log"; then
      echo "=== $NAME finished $(date)"
    else
      echo "=== $NAME FAILED (see logs/$NAME.log) $(date)"; FAILED+=("$NAME")
    fi
  done
fi

# ---- trained configs
if [ "${#IDS[@]}" -gt 0 ]; then
  SUFFIX=""
  if [ "$SEED" != "42" ]; then SUFFIX="_s$SEED"; fi
  for i in "${IDS[@]}"; do
    read -r DATA NAME EXTRA <<< "${CONFIGS[$i]}"
    EVAL_EXTRA=""
    if [[ "$NAME" == "two_level" ]]; then EVAL_EXTRA="--two_level"; fi
    TOK_FLAG=""   # configs that set their own --backbone (RoBERTa, BART) keep their own tokenizer
    if [ -n "$TOKENIZER" ] && [[ "$EXTRA" != *"--backbone"* ]]; then TOK_FLAG="--tokenizer $TOKENIZER"; fi
    RUN="final_${NAME}${TAG:+_$TAG}${SUFFIX}"; TEST_FLAGS=""
    if [[ "${TEST:-0}" == "1" ]]; then RUN="test_$RUN"; TEST_FLAGS="--max_train 3000 --max_val 500"; fi
    OUT="runs/$RUN"; LOG="logs/$RUN.log"
    echo "=== config $i: $NAME, seed $SEED  (output: $OUT, log: $LOG)  started $(date)"
    # later flags override earlier ones (argparse keeps the last value), e.g. --backbone in $EXTRA
    if {
      python -m unide.train --train "data/$DATA/train.jsonl" --val "data/$DATA/val.jsonl" \
        --backbone "$BACKBONE" $TOK_FLAG --out "$OUT" --bf16 --seed "$SEED" $PROTOCOL $EXTRA $TEST_FLAGS &&
      evaluate_all "--model $OUT" "$OUT" "$EVAL_EXTRA"
    } 2>&1 | tee "$LOG"; then
      echo "=== config $i: $NAME finished $(date)"
    else
      echo "=== config $i: $NAME FAILED (see $LOG) $(date)"; FAILED+=("$RUN")
    fi
  done
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "Finished with failures: ${FAILED[*]}"
  exit 1
fi
echo "All done. Results table: python -m unide.summarize"
