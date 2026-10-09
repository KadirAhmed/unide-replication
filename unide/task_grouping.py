"""Task division and representative-dimension selection (Sec. 3.3, Fig. 3b/c, Fig. 4, Eqs. 2-5).

  1. annotate  - label a small sample (300 turn instances) with all 12 candidate dimensions using the LLM
  2. group     - Spearman correlation matrix between dimensions; split the 12 dimensions into 3 groups of 4
                 strongly-correlated dimensions (threshold 0.6)
  3. template  - export a CSV for humans to assign the three overall scores (Flu/Coh/Eng) to each instance
  4. weights   - fit OS_task = sum_k w_k * S_k per group, normalise the weights, keep the top-2 per group

  python -m unide.task_grouping annotate --inp data/verified.jsonl --n 300 --out data/sample12.jsonl
  python -m unide.task_grouping group    --inp data/sample12.jsonl --out_dir data/grouping
  python -m unide.task_grouping template --inp data/sample12.jsonl --out data/grouping/overall_scores.csv
  python -m unide.task_grouping weights  --inp data/sample12.jsonl --overall data/grouping/overall_scores.csv
"""
import argparse
import csv
import itertools
import json
import os
import random

import numpy as np
from scipy.stats import spearmanr

from .dimensions import STUDIED_DIMENSIONS, DIMENSIONS, description
from .llm import ChatLLM, parse_json
from .data_generation import read_jsonl


# ---------------------------------------------------------------- 1. annotation with all 12 candidates
def annotate_prompt(context: str, response: str) -> str:
    defs = "\n".join(f"- {d}: {description(d)}" for d in STUDIED_DIMENSIONS)
    keys = ", ".join(f'"{d}": s' for d in STUDIED_DIMENSIONS)
    return (f"Context: {context}\nResponse: {response}\n\nScore the response (and the short conversation it forms "
            f"with the context) on each aspect, 1 = bad, 2 = moderate, 3 = good:\n{defs}\n\nReturn JSON: {{{keys}}}")


def cmd_annotate(args, llm=None):
    from concurrent.futures import ThreadPoolExecutor
    llm = llm or ChatLLM(args.model, temperature=0.0)
    data = read_jsonl(args.inp)
    rng = random.Random(args.seed)
    items = []
    for inst in rng.sample(data, min(len(data), args.n)):
        t = rng.choice(inst["turns"])
        items.append({"context": f"Human: {t['human']}", "response": f"Chatbot: {t['chatbot']}"})

    def work(it):
        out = parse_json(llm("You are an expert dialogue-quality annotator. Reply with JSON only.",
                             annotate_prompt(it["context"], it["response"])))
        return {**it, "scores": {d: int(out[d]) for d in STUDIED_DIMENSIONS}}

    with ThreadPoolExecutor(args.workers) as ex:
        results = [r for r in ex.map(lambda it: _safe(work, it), items) if r]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(results):
            f.write(json.dumps({"id": i, **r}, ensure_ascii=False) + "\n")
    print(f"annotated {len(results)} instances -> {args.out}")


def _safe(fn, x):
    try:
        return fn(x)
    except Exception as e:
        print(f"[warn] {e}")
        return None


def score_matrix(rows, dims=STUDIED_DIMENSIONS) -> np.ndarray:
    return np.array([[r["scores"][d] for d in dims] for r in rows], dtype=float)


# ---------------------------------------------------------------- 2. correlation-based grouping
def spearman_matrix(X: np.ndarray) -> np.ndarray:
    """Eq. (2) for every pair of dimensions (columns of X)."""
    rho = spearmanr(X)[0]
    return np.nan_to_num(np.atleast_2d(rho), nan=0.0)


def best_partition(rho: np.ndarray, n_groups: int = 3, threshold: float = 0.6):
    """Exhaustively search partitions of the dimensions into equal-size groups (5775 for 12 -> 3x4) and return
    the one with the highest mean within-group correlation, plus whether every within-group pair >= threshold."""
    n = rho.shape[0]
    size = n // n_groups
    best, best_score = None, -np.inf

    def partitions(items):
        if not items:
            yield []
            return
        first, rest = items[0], items[1:]
        for comb in itertools.combinations(rest, size - 1):
            group = (first,) + comb
            remaining = [x for x in rest if x not in comb]
            for p in partitions(remaining):
                yield [group] + p

    for p in partitions(list(range(n))):
        score = np.mean([rho[i, j] for g in p for i, j in itertools.combinations(g, 2)])
        if score > best_score:
            best, best_score = p, score
    min_within = min(rho[i, j] for g in best for i, j in itertools.combinations(g, 2))
    return best, best_score, min_within >= threshold, min_within


def cmd_group(args):
    rows = read_jsonl(args.inp)
    rho = spearman_matrix(score_matrix(rows))
    os.makedirs(args.out_dir, exist_ok=True)
    np.savetxt(os.path.join(args.out_dir, "spearman.csv"), rho, delimiter=",", fmt="%.3f",
               header=",".join(STUDIED_DIMENSIONS), comments="")
    parts, score, ok, min_within = best_partition(rho, threshold=args.threshold)
    names = [[STUDIED_DIMENSIONS[i] for i in g] for g in parts]
    print(f"mean within-group rho = {score:.3f}; min within-group rho = {min_within:.3f} "
          f"({'>=' if ok else '<'} threshold {args.threshold})")
    for g in names:
        # name the group after its most representative (paper) task, by majority vote over Fig. 3(c) groups
        votes = [DIMENSIONS[d][3] for d in g]
        print(f"  {max(set(votes), key=votes.count):3s}: {g}")
    changed = [d for g in names for d in g if DIMENSIONS[d][2] != _majority_level(g)]
    print(f"dimensions whose group differs from their original level: {changed}")
    with open(os.path.join(args.out_dir, "groups.json"), "w") as f:
        json.dump({"groups": names, "mean_within_rho": score, "min_within_rho": min_within}, f, indent=2)
    if args.plot:
        _plot(rho, os.path.join(args.out_dir, "spearman.png"))


def _majority_level(group):
    lv = [DIMENSIONS[d][2] for d in group]
    return max(set(lv), key=lv.count)


def _plot(rho, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(rho, cmap="coolwarm", vmin=0.3, vmax=1.0)
    ax.set_xticks(range(len(STUDIED_DIMENSIONS)), STUDIED_DIMENSIONS, rotation=45, ha="right")
    ax.set_yticks(range(len(STUDIED_DIMENSIONS)), STUDIED_DIMENSIONS)
    for i in range(len(rho)):
        for j in range(len(rho)):
            ax.text(j, i, f"{rho[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


# ---------------------------------------------------------------- 3/4. representative dimensions
TASK_GROUPS = {t: [d for d in STUDIED_DIMENSIONS if DIMENSIONS[d][3] == t] for t in ("flu", "coh", "eng")}


def cmd_template(args):
    rows = read_jsonl(args.inp)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "context", "response", "overall_flu", "overall_coh", "overall_eng"])
        for r in rows:
            w.writerow([r["id"], r["context"], r["response"], "", "", ""])
    print(f"fill in overall_flu/coh/eng (1-3) for each row of {args.out}, then run `weights`")


def fit_group_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Least squares for Eqs. (3)-(5) (with intercept), weights normalised to sum to 1."""
    A = np.hstack([X, np.ones((len(X), 1))])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    w = coef[:-1]
    if (w < 0).any():
        print(f"  [note] negative raw weight(s) {np.round(w, 3)}; the paper reports all positive")
    wp = np.clip(w, 0, None)
    return wp / wp.sum() if wp.sum() > 0 else wp


def select_representatives(rows, overall: dict, top_k: int = 2):
    result = {}
    for task, dims in TASK_GROUPS.items():
        use = [r for r in rows if str(r["id"]) in overall and overall[str(r["id"])].get(task) is not None]
        X = score_matrix(use, dims)
        y = np.array([overall[str(r["id"])][task] for r in use], dtype=float)
        w = fit_group_weights(X, y)
        order = np.argsort(-w)
        result[task] = {"weights": {dims[i]: round(float(w[i]), 4) for i in order},
                        "representative": [dims[i] for i in order[:top_k]]}
    return result


def cmd_weights(args):
    rows = read_jsonl(args.inp)
    overall = {}
    with open(args.overall, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            vals = {t: r.get(f"overall_{t}") for t in ("flu", "coh", "eng")}
            overall[r["id"]] = {t: float(v) for t, v in vals.items() if v not in (None, "")}
    res = select_representatives(rows, overall)
    for t, r in res.items():
        print(f"{t}: weights={r['weights']}  -> representative {r['representative']}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("annotate")
    a.add_argument("--inp", default="data/verified.jsonl")
    a.add_argument("--out", default="data/sample12.jsonl")
    a.add_argument("--n", type=int, default=300)
    a.add_argument("--model", default="gpt-3.5-turbo-0125")
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("group")
    g.add_argument("--inp", default="data/sample12.jsonl")
    g.add_argument("--out_dir", default="data/grouping")
    g.add_argument("--threshold", type=float, default=0.6)
    g.add_argument("--plot", action="store_true")
    t = sub.add_parser("template")
    t.add_argument("--inp", default="data/sample12.jsonl")
    t.add_argument("--out", default="data/grouping/overall_scores.csv")
    w = sub.add_parser("weights")
    w.add_argument("--inp", default="data/sample12.jsonl")
    w.add_argument("--overall", default="data/grouping/overall_scores.csv")
    w.add_argument("--out", default="data/grouping/representative.json")
    args = ap.parse_args()
    {"annotate": cmd_annotate, "group": cmd_group, "template": cmd_template, "weights": cmd_weights}[args.cmd](args)


if __name__ == "__main__":
    main()
