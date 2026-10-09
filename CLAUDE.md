# UniDE replication

Independent reimplementation of UniDE (Ye, Zhao, Zhang and Jiang, *Information Processing & Management* 62 (2025) 104035, doi:10.1016/j.ipm.2024.104035), written from the paper because the authors released no code or data. `README.md` maps each paper section to the code and lists every deviation from the paper.

## Layout

- `unide/`: the package. `data_generation` (LLM generate, verify, build), `llm` (OpenAI client), `templates` (prompts and label words), `task_grouping`, `task_selector`, `model` (backbone with three task heads), `train`, `evaluate`, `benchmarks` (FED, Persona-usr, Persona-see and Topical-USR converters), `summarize` (results table next to the paper's numbers), `dimensions`.
- `scripts/run_local.sh`: train and evaluate configs 0-9, or the untrained backbones, on one GPU with the final protocol. `scripts/run_all.sh`: the whole pipeline in order. `scripts/unide_array.sbatch` and `scripts/unide_interactive.sh`: SLURM versions for the Virga cluster.
- `tests/`: pytest with a mock LLM. No GPU, model download or API key needed.
- Not in git: `data/` (generated UniDE-data and benchmarks), `runs/` (checkpoints and result JSON), `logs/`, `hf_cache/`.

## Commands

- Install: `pip install -r requirements.txt` (Python 3.10 or later)
- Tests, after every change: `python -m pytest tests -q`
- Quick GPU check: `TEST=1 bash scripts/run_local.sh 0`
- Main model: `bash scripts/run_local.sh 0`. Another seed: `SEED=43 bash scripts/run_local.sh 0`
- Results table: `python -m unide.summarize`

## Rules

- FED and PersonaChat are test sets. Never choose checkpoints, hyperparameters or prompts on them. Model selection uses only Topical-USR (`data/bench/tc_usr.jsonl`).
- `unide/train.py` defaults follow the paper (30 epochs, lr 3e-5). The final protocol is in `scripts/run_local.sh`: at most 1 epoch, lr 5e-6, Topical-USR check every 50 steps, keep the best checkpoint, stop after 10 checks without improvement. Use the script for real runs.
- Report both scorings: `argmax` (the paper's) and `expected` (continuous).
- Add any new deviation from the paper to the README's "Deviations" list.
- Data generation reads `OPENAI_API_KEY` from the environment. Never write keys into files or commits.
- Runs take hours: start them inside `tmux`.

## Machines and known problems

- Mac: editing, tests and smoke tests (no GPU).
- Rented GPU (Ubuntu, H100 NVL, `~/llm/unide`, miniconda Python 3.10, PyTorch 2.3): training. Its preinstalled `flash-attn` was built for another PyTorch and crashes the OPT import (`undefined symbol ... SetDevice`). Fix: `pip uninstall -y flash-attn` (UniDE does not use it).
- Virga cluster (SLURM): compute nodes may have no internet, so download models on the login node into `hf_cache/` first. The scripts set `HF_HUB_OFFLINE=1`.
- Load RoBERTa-large as `FacebookAI/roberta-large`. The old short name `roberta-large` returns 404 with newer Hugging Face downloads.

## Status (9 Oct 2026)

- Main model, OPT-1.3B, mean of 3 seeds, argmax scoring (Pearson r / Spearman rho x 100, averaged over the three levels): FED 31.6 / 28.3, PersonaChat 17.5 / 17.3. Continuous scoring: 36.2 / 32.6 and 19.8 / 19.9. Paper: 52.1 / 53.4 and 46.4 / 48.0.
- Next: test the DeepSpeed-Chat RLHF actor backbone, `BACKBONE=AdamG012/chat-opt-1.3b-rlhf-actor-ema-deepspeed TOKENIZER=facebook/opt-1.3b TAG=rlhf bash scripts/run_local.sh vanilla 0`.
