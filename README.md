# UniDE — reimplementation

Implementation of *UniDE: A multi-level and low-resource framework for automatic dialogue evaluation via LLM-based data augmentation and multitask learning* (Ye, Zhao, Zhang, Jiang — Information Processing & Management 62, 2025). The authors did not release code, so this is an independent implementation written from the paper.

## Paper → code

| Paper | Code |
|---|---|
| §3.2 / Fig. 2a: generative LLM + verifiable LLM, instance filtering | `unide/data_generation.py` (`generate`, `verify`, `build`) |
| §3.2 / Fig. 2b: numeric label → natural-language polarity words | `unide/templates.py` (`VERBALIZERS`, `build_prompt`) |
| §3.3 / Fig. 3b-c, Eq. 2: Spearman matrix, re-grouping into Flu/Coh/Eng | `unide/task_grouping.py group` |
| §3.3 / Fig. 4, Eqs. 3-5: linear weights, top-2 representative dimensions | `unide/task_grouping.py weights` |
| §3.3 Eq. 6: hard-parameter-sharing backbone + 3 task heads, weighted CE | `unide/model.py`, `unide/train.py` |
| §3.4 / Fig. 5, Eqs. 7-14: level-specific templates, task selector, verbaliser | `unide/templates.py`, `unide/task_selector.py`, `UniDE.score` |
| §4.1 / App. C: FED, Persona-usr, Persona-see; Tables 14-15 | `unide/benchmarks.py` |
| §4.2, §5.1 / Tables 2-3: min-max normalised Pearson/Spearman per level | `unide/evaluate.py` |
| §5.2 / Fig. 6: MSE / ACC and scatter plot | `evaluate.py --plot` |
| §5.3 / Table 4, §5.5 RQ1-3, §5.6 Discussions 1-4 | configs 0-9 of `scripts/run_local.sh`; results table: `unide/summarize.py` |

## How to run

`scripts/run_all.sh` lists every step in order. In short:

```bash
pip install -r requirements.txt
python -m pytest tests -q                 # unit tests (mock LLM, no GPU)
export OPENAI_API_KEY=...                 # data generation only
# steps 1-3 of scripts/run_all.sh: generate + verify UniDE-data, download and convert the benchmarks
bash scripts/run_local.sh all             # train + evaluate every config on one GPU (inside tmux)
bash scripts/run_local.sh vanilla         # untrained backbones
python -m unide.summarize                 # one table, with the paper's numbers alongside
```

On a SLURM cluster, `scripts/unide_array.sbatch` runs the same configs as an array job (task 10 = untrained backbones).

Benchmarks: FED and Persona-usr come from the authors' site (`shikib.com`). Persona-see is the See et al. (2019) human-evaluation log released with ParlAI (`evaluationlogs_v1.tar.gz`, 3,316 conversations; the later "reproducible" re-release has 2,724). Topical-USR (`tc_usr_data.json`) is used only as a development set.

## Deviations from the paper

Record these in any write-up.

1. **Backbone.** The paper's "InstructGPT (1.3B)" is its own reproduction made with `LanXiu0523/RLHF_instructGPT` (footnote 3), a DeepSpeed-Chat fork that trains `facebook/opt-1.3b` (actor) and `facebook/opt-350m` (reward model): so the paper's backbone is OPT-1.3B after supervised fine-tuning and RLHF. This implementation's default is plain `facebook/opt-1.3b`. The closest public equivalent of the paper's backbone is a DeepSpeed-Chat OPT-1.3B RLHF actor such as `AdamG012/chat-opt-1.3b-rlhf-actor-ema-deepspeed` (same pipeline, different preference data); it ships without tokenizer files, so use `--tokenizer facebook/opt-1.3b` (`BACKBONE=... TOKENIZER=facebook/opt-1.3b TAG=rlhf bash scripts/run_local.sh vanilla 0`). The instruction-tuned `facebook/opt-iml-max-1.3b` was also tried and scored lower on the development set. RoBERTa-large (355M) and BART-large (406M) are supported for RQ1; the paper swaps their sizes in §5.5.
2. **`[MASK]` with a decoder.** A left-to-right model cannot see text after the slot, so for decoders the dimension moves before the slot ("… Is the entire conversation coherent? The entire conversation is [SLOT]"). RoBERTa and BART use the paper's template ("… the entire conversation is [MASK] coherent.") verbatim.
3. **Heads.** Each task head is a linear layer over the hidden state at the slot with 3 outputs (not/moderate/very), initialised from the label words' embeddings, trained with cross entropy.
4. **Data generation.** The paper's prompts are unpublished, and its gpt-3.5-turbo-1106 was retired on 2026-09-28, so generation used gpt-3.5-turbo-0125. Both LLM calls share a written rubric with calibration examples; response-level problems (fluency, understandability) are added by a separate focused rewrite call, because GPT-3.5 would not write broken replies inside a conversation. Of 8,200 generated conversations, 8,195 were verified (5 verifier outputs were unusable).
5. **Verifier and filtering.** Verification used GPT-4 Turbo (gpt-4-turbo-2024-04-09), the alternative of the paper's Discussion 4, because it agreed with the plan more often than GPT-3.5 in a pilot. Rather than dropping whole conversations (the paper: 8.54%), labels where the verifier says the opposite of the plan (bad vs good) are dropped: about 4% of labels. Exact-agreement filtering removed most "moderate" labels, so it was not used. Labels are the preset quality, as in the paper.
6. **Turn-level instances** are single-turn pairs (one Human utterance and one Chatbot reply), following Table 13.
7. **Batching.** Batches are balanced across tasks (5 examples per task at batch size 16).
8. **Checkpoint selection (the largest change).** Selecting on synthetic validation data, as the paper does, picked models that fit the generated data but transferred poorly to human ratings (FED 27.2/25.6 vs 36.8/32.3, continuous scoring). Instead, training runs for at most one epoch at learning rate 5e-6, is checked every 50 steps against Topical-USR's human ratings, keeps the best checkpoint, and stops after 10 checks without improvement. Topical-USR never overlaps the test sets, but it shares USR's rating dimensions with Persona-usr.
9. **Scoring.** Both the paper's argmax → {1,2,3} (Eqs. 13-14) and continuous Σ y·p(y) scores are reported.
10. **Persona-usr "Relevance".** USR has no Relevance rating, though Table 15 lists one; that row is skipped.
11. **Representative dimensions.** The six dimensions the paper found are built in (`dimensions.REPRESENTATIVE`); `task_grouping` reproduces the selection procedure but needs human overall scores.
12. **Memory.** One 80-94 GB H100 fits the 1.3B model without gradient checkpointing.

Exact numbers in Tables 2-3 are not expected to reproduce: they depend on the unpublished generation prompts, the particular InstructGPT reproduction, and model-selection details the paper does not give.
