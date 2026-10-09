"""Collect the evaluation results of all runs into one table (the paper's Tables 2-4 and 7-10 in one place).

  python -m unide.summarize                          # every runs/*/ folder that has evaluation files
  python -m unide.summarize --runs runs/final_* runs/vanilla_*
Prints a compact table (averages over levels) and writes every level to results/summary.csv.
"""
import argparse
import csv
import glob
import json
import os

# The paper's numbers (r/rho x 100, averages over levels) for the same configurations, for side-by-side comparison.
# Sources: Tables 2-4, 7-10 and Fig. 7a. "-" = not reported; two_level is the main result minus the reported drop
# (6.2 on FED, 7.4 on PersonaChat, Pearson only); Fig. 7a reports FED Pearson only.
PAPER = {
    "main":            ("52.1/53.4", "46.4/48.0"),
    "wo_filter":       ("42.4/42.3", "-"),
    "wo_multitask":    ("32.8/32.9", "-"),
    "score_prompt":    ("41.5/42.3", "-"),
    "mc_prompt":       ("46.7/47.0", "-"),
    "level_groups":    ("44.6/43.4", "38.4/38.1"),
    "best_weights":    ("52.5/53.8", "-"),
    "two_level":       ("45.9/-", "39.0/-"),
    "roberta_large":   ("35.1/-", "-"),
    "bart_large":      ("38.7/-", "-"),
    "vanilla_opt-1.3b":      ("35.0/- (InstructGPT)", "-"),
    "vanilla_chat-opt-1.3b-rlhf-actor-ema-deepspeed": ("35.0/- (InstructGPT)", "-"),
    "vanilla_chat-opt-1.3b-rlhf-actor-deepspeed":     ("35.0/- (InstructGPT)", "-"),
    "vanilla_roberta-large": ("18.4/-", "-"),
    "vanilla_bart-large":    ("19.5/-", "-"),
}


def paper_key(run: str) -> str:
    """final_main_s43 -> main; test_final_wo_filter -> wo_filter; final_main_rlhf (a TAG run) -> main;
    vanilla_* stays as is."""
    name = run[len("test_"):] if run.startswith("test_") else run
    name = name[len("final_"):] if name.startswith("final_") else name
    if "_s" in name and name.rsplit("_s", 1)[1].isdigit():
        name = name.rsplit("_s", 1)[0]
    if name not in PAPER and not name.startswith("vanilla_"):
        for key in sorted(PAPER, key=len, reverse=True):
            if name.startswith(key + "_"):
                return key
    return name


MODES = ("argmax", "expected")
BENCHES = ("fed", "personachat")
LEVELS = ("response", "turn", "dialogue", "avg")


def _load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _pair(res, level):
    if not res or level not in res.get("levels", {}):
        return "-"
    v = res["levels"][level]
    return f"{100 * v['pearson']:.1f}/{100 * v['spearman']:.1f}"


def collect(run_dirs):
    rows = []
    for d in sorted(run_dirs):
        res = {(m, b): _load(os.path.join(d, f"{b}_{m}.json")) for m in MODES for b in BENCHES}
        if not any(res.values()):
            continue
        args = _load(os.path.join(d, "train_args.json")) or {}
        backbone = args.get("backbone")
        if backbone is None and os.path.exists(os.path.join(d, "backbone.txt")):
            backbone = open(os.path.join(d, "backbone.txt")).read().strip() + " (untrained)"
        dev = args.get("best_dev")
        row = {"run": os.path.basename(os.path.normpath(d)), "backbone": backbone or "-",
               "dev r/rho": f"{100 * dev['pearson']:.1f}/{100 * dev['spearman']:.1f}" if dev else "-",
               "selected at": args.get("selected_at", "-")}
        for m in MODES:
            for b in BENCHES:
                for lvl in LEVELS:
                    row[f"{b} {lvl} ({m})"] = _pair(res[(m, b)], lvl)
        row["paper fed avg"], row["paper personachat avg"] = PAPER.get(paper_key(row["run"]), ("-", "-"))
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None, help="run folders (default: every folder in runs/)")
    ap.add_argument("--out", default="results/summary.csv")
    args = ap.parse_args()
    dirs = args.runs if args.runs else [d for d in glob.glob("runs/*") if os.path.isdir(d)]
    rows = collect(dirs)
    if not rows:
        print("no evaluation results found")
        return
    cols = ["run", "dev r/rho", "fed avg (argmax)", "fed avg (expected)", "paper fed avg",
            "personachat avg (argmax)", "personachat avg (expected)", "paper personachat avg"]
    heads = ["run", "dev (Topical-USR)", "FED (paper scoring)", "FED (continuous)", "FED (paper)",
             "PersonaChat (paper scoring)", "PersonaChat (continuous)", "PersonaChat (paper)"]
    print("| " + " | ".join(heads) + " |")
    print("|" + "---|" * len(heads))
    for r in rows:
        print("| " + " | ".join(r[c] for c in cols) + " |")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nall levels written to {args.out} (values are Pearson/Spearman x 100; the paper used "
          f"its own scoring, so compare the 'paper scoring' columns with the paper's)")


if __name__ == "__main__":
    main()
