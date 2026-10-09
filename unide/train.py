"""Multitask training of UniDE on UniDE-data (Sec. 3.3, Eq. 6; hyper-parameters of Sec. 4.4).

  python -m unide.train --train data/unide_data/train.jsonl --val data/unide_data/val.jsonl \
      --backbone facebook/opt-1.3b --out runs/unide_opt13b --bf16 --grad_checkpointing

Ablations
  --grouping single         w/o multitask learning (one module on all data)
  --grouping level          original level groups instead of Flu/Coh/Eng (Discussion 1)
  --prompt score | mc       w/o template mapping (Score-prompt) / MC-prompt (Discussion 3)
  --two_level               response-level dims use the turn-level template (RQ3)
  --task_weights 0.29,0.36,0.35   best-weight setting of Discussion 2
  data from `build --no_filter --target 0`  w/o instance filtering
"""
import argparse
import json
import math
import os
import random
from collections import defaultdict

import numpy as np
import torch
from scipy.stats import spearmanr

from .benchmarks import SPECS, load_unified
from .dimensions import task_of, task_names
from .evaluate import aggregate, run as eval_run
from .model import UniDE
from .task_selector import TaskSelector


def load_examples(path: str, grouping: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                r["task"] = task_of(r["dimension"], grouping, r["level"])
                rows.append(r)
    return rows


class MultitaskBatches:
    """Mini-batches that equally cover every task subset (Flu-data / Coh-data / Eng-data)."""

    def __init__(self, examples, tasks, batch_size, seed=0):
        self.by_task = defaultdict(list)
        for i, ex in enumerate(examples):
            self.by_task[ex["task"]].append(i)
        self.tasks = [t for t in tasks if self.by_task[t]]
        self.per_task = max(1, batch_size // len(self.tasks))
        self.n_batches = math.ceil(len(examples) / (self.per_task * len(self.tasks)))
        self.rng = random.Random(seed)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        pools = {t: [] for t in self.tasks}
        for _ in range(self.n_batches):
            batch = []
            for t in self.tasks:
                for _ in range(self.per_task):
                    if not pools[t]:  # smaller subsets are cycled with a fresh shuffle
                        pools[t] = self.rng.sample(self.by_task[t], len(self.by_task[t]))
                    batch.append(pools[t].pop())
            self.rng.shuffle(batch)
            yield batch


@torch.no_grad()
def validate(model, examples, args, device):
    model.eval()
    preds, labels, tasks = [], [], []
    for i in range(0, len(examples), args.eval_batch_size):
        chunk = examples[i:i + args.eval_batch_size]
        batch = {k: v.to(device) for k, v in model.collate(chunk, args.max_len, args.two_level).items()}
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda"):
            exp, probs = model.score(batch, mode="expected")
        preds += exp.float().cpu().tolist()
        labels += [c["label"] for c in chunk]
        tasks += [c["task"] for c in chunk]
    model.train()
    preds, labels, tasks = np.array(preds), np.array(labels), np.array(tasks)
    report = {}
    for t in sorted(set(tasks)):
        m = tasks == t
        report[t] = {"acc": float((np.clip(np.rint(preds[m]), 1, 3) == labels[m]).mean()),
                     "spearman": float(spearmanr(preds[m], labels[m])[0])}
    report["avg_spearman"] = float(np.nanmean([v["spearman"] for v in report.values()]))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--backbone", default="facebook/opt-1.3b",
                    help="InstructGPT-1.3B is not public; use an open 1.3B decoder or your RLHF checkpoint")
    ap.add_argument("--kind", choices=["mlm", "seq2seq", "causal"], default=None)
    ap.add_argument("--tokenizer", default=None,
                    help="load the tokenizer from here instead of --backbone (for checkpoints that ship without "
                         "tokenizer files, e.g. DeepSpeed-Chat RLHF actors: use facebook/opt-1.3b)")
    ap.add_argument("--out", default="runs/unide")
    ap.add_argument("--grouping", choices=["task", "level", "single"], default="task")
    ap.add_argument("--prompt", choices=["mlm", "score", "mc"], default="mlm")
    ap.add_argument("--two_level", action="store_true")
    ap.add_argument("--task_weights", default=None, help="comma-separated, default 1/3 each")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--patience", type=int, default=5, help="early stop after N epochs w/o val improvement")
    ap.add_argument("--max_val", type=int, default=6000, help="subsample validation set for speed")
    ap.add_argument("--max_train", type=int, default=0, help="subsample training set (0 = all); for smoke tests")
    ap.add_argument("--dev_data", default=None,
                    help="human-rated development benchmark (unified JSONL, e.g. data/bench/tc_usr.jsonl); if given, "
                         "the checkpoint with the best correlation on it is kept instead of the best synthetic val")
    ap.add_argument("--dev_bench", default="topicalchat", choices=list(SPECS))
    ap.add_argument("--eval_every", type=int, default=250, help="dev evaluation every N optimizer steps (with --dev_data)")
    ap.add_argument("--dev_patience", type=int, default=8, help="stop after N dev evaluations without improvement")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--grad_checkpointing", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tasks = task_names(args.grouping)
    weights = None
    if args.task_weights:
        weights = dict(zip(tasks, map(float, args.task_weights.split(","))))

    train = load_examples(args.train, args.grouping)
    val = load_examples(args.val, args.grouping)
    if len(val) > args.max_val:
        val = random.Random(0).sample(val, args.max_val)
    if args.max_train and len(train) > args.max_train:
        train = random.Random(0).sample(train, args.max_train)
    print(f"train {len(train)} / val {len(val)} examples; tasks {tasks}")

    model = UniDE(args.backbone, tasks, args.prompt, args.kind, gradient_checkpointing=args.grad_checkpointing,
                  tokenizer_path=args.tokenizer)
    model.to(device)
    print(f"backbone kind: {model.kind}; label words {model.label_words} -> ids {model.label_ids}")

    batches = MultitaskBatches(train, tasks, args.batch_size, args.seed)
    steps = args.epochs * len(batches) // args.grad_accum
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    from transformers import get_linear_schedule_with_warmup
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup_ratio * steps), steps)

    os.makedirs(args.out, exist_ok=True)
    dev_rows = load_unified([args.dev_data]) if args.dev_data else None
    selector = TaskSelector(args.grouping)
    state = {"best": -1e9, "bad": 0}

    def save_best(info):
        model.save(args.out)
        with open(os.path.join(args.out, "train_args.json"), "w") as f:
            json.dump({**vars(args), **info}, f, indent=2)
        print(f"  saved best model to {args.out}", flush=True)

    def dev_eval(tag):
        """Correlation with human ratings on the development benchmark; returns True when it is time to stop."""
        per_dim, _ = eval_run(model, dev_rows, SPECS[args.dev_bench], selector, args.eval_batch_size, args.max_len,
                              "expected", args.two_level, device)
        model.train()
        avg = aggregate(per_dim)["avg"]
        score = (avg["pearson"] + avg["spearman"]) / 2
        print(f"{tag} dev ({args.dev_bench}) r/rho = {100 * avg['pearson']:.1f}/{100 * avg['spearman']:.1f}",
              flush=True)
        if score > state["best"]:
            state["best"], state["bad"] = score, 0
            save_best({"best_dev": avg, "selected_at": tag})
        else:
            state["bad"] += 1
        return state["bad"] >= args.dev_patience

    stop, global_step = False, 0
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, idx in enumerate(batches, 1):
            batch = {k: v.to(device) for k, v in model.collate([train[i] for i in idx], args.max_len,
                                                                args.two_level).items()}
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=args.bf16 and device.type == "cuda"):
                loss = model(**batch, task_weights=weights)["loss"] / args.grad_accum
            loss.backward()
            running += loss.item() * args.grad_accum
            if step % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                global_step += 1
                if dev_rows is not None and args.eval_every and global_step % args.eval_every == 0:
                    if dev_eval(f"epoch {epoch + 1} step {step}"):
                        stop = True
                        break
            if step % 200 == 0:
                print(f"epoch {epoch + 1} step {step}/{len(batches)} loss {running / 200:.4f}", flush=True)
                running = 0.0
        rep = validate(model, val, args, device)
        print(f"epoch {epoch + 1} validation: {json.dumps(rep)}", flush=True)
        if dev_rows is not None:
            if not stop and dev_eval(f"epoch {epoch + 1} end"):
                stop = True
        elif rep["avg_spearman"] > state["best"]:   # paper's setting: select on the synthetic validation set
            state["best"], state["bad"] = rep["avg_spearman"], 0
            save_best({"best_val": rep, "epoch": epoch + 1})
        else:
            state["bad"] += 1
            stop = state["bad"] >= args.patience
        if stop:
            print("early stopping")
            break


if __name__ == "__main__":
    main()
