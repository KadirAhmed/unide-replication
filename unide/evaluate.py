"""Meta-evaluation of UniDE against human ratings (Sec. 4.2, Tables 2/3, Sec. 5.2).

  python -m unide.evaluate --model runs/unide --benchmark fed --data data/bench/fed.jsonl
  python -m unide.evaluate --model runs/unide --benchmark personachat \
        --data data/bench/persona_usr.jsonl data/bench/persona_see.jsonl
  # vanilla (untrained) backbone, as in RQ1/RQ2:
  python -m unide.evaluate --vanilla facebook/opt-1.3b --benchmark fed --data data/bench/fed.jsonl
"""
import argparse
import json
import random

import numpy as np
from scipy.stats import pearsonr, spearmanr

from .benchmarks import SPECS, load_unified
from .dimensions import LEVELS
from .task_selector import TaskSelector


def minmax(x):
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)


ALIASES = {  # benchmark releases spell some qualities differently
    "fluency": ["fluent"], "listen": ["listening"], "interest": ["interesting", "interestingness"],
    "enjoy": ["enjoyment", "enjoyable"], "inquisitive": ["inquisitiveness"],
}


def _norm(k: str) -> str:
    return "".join(ch for ch in k.lower() if ch.isalnum())


def find_key(annotations: dict, key: str):
    if key in annotations:
        return key
    wanted = {_norm(key)} | {_norm(a) for a in ALIASES.get(key.lower(), [])}
    return next((k for k in annotations if _norm(k) in wanted), None)


def bare(utt: str) -> str:
    """Drop a leading speaker tag: response-level items are scored without any context or tag."""
    head, sep, rest = utt.partition(":")
    return rest.strip() if sep and head.strip() in ("Chatbot", "Human", "System", "User") else utt.strip()


def build_items(rows, spec, selector):
    """One list of scoring items per (level, dimension) of the spec."""
    jobs = []
    for level, dim, src_type, key in spec:
        items = []
        for r in rows:
            k = find_key(r["annotations"], key) if r["type"] == src_type else None
            if k is None:
                continue
            if level == "response":
                ctx, resp = [], bare(r["response"])
            elif level == "turn":
                ctx, resp = r["context"], r["response"]
            else:
                ctx, resp = r["context"] + ([r["response"]] if r.get("response") else []), None
            items.append({"id": r["id"], "level": level, "dimension": dim, "context": ctx, "response": resp,
                          "task": selector(dim, level), "human": r["annotations"][k]})
        if items:
            jobs.append((level, dim, items))
        else:
            print(f"[warn] no annotations for {dim} ({key}); skipped")
    return jobs


def pred_summary(pred, mode: str = "argmax") -> str:
    """How spread out the model's scores are: share of 1/2/3 (argmax) or mean and spread (expected)."""
    pred = np.asarray(pred, dtype=float)
    if mode == "argmax":
        n = max(1, len(pred))
        return "pred 1/2/3 = " + "/".join(f"{100 * np.sum(np.rint(pred) == q) / n:.0f}%" for q in (1, 2, 3))
    return f"pred mean {pred.mean():.2f} sd {pred.std():.2f}"


def correlations(pred, human):
    p, h = minmax(pred), minmax(human)  # Min-Max normalisation of both score scales
    if np.std(p) == 0 or np.std(h) == 0:
        return {"pearson": 0.0, "spearman": 0.0, "p_pearson": 1.0, "p_spearman": 1.0, "n": len(p)}
    r, pr = pearsonr(p, h)
    s, ps = spearmanr(p, h)
    return {"pearson": float(r), "spearman": float(s), "p_pearson": float(pr), "p_spearman": float(ps), "n": len(p)}


def aggregate(per_dim):
    """Average correlations of dimensions assigned to the same level, then over levels (the 'Avg.' column)."""
    table = {}
    for lvl in LEVELS:
        ds = [v for v in per_dim.values() if v["level"] == lvl]
        if ds:
            table[lvl] = {m: float(np.mean([d[m] for d in ds])) for m in ("pearson", "spearman")}
    table["avg"] = {m: float(np.mean([table[l][m] for l in LEVELS if l in table])) for m in ("pearson", "spearman")}
    return table


def mse_analysis(per_dim_scores, n=100, seed=0):
    """Sec. 5.2: per (instance, level) average of normalised predicted / human scores on a 0-100 scale; up to
    n instances per level; ACC = 1 - MSE (on [0,1])."""
    points = {}
    for (level, dim), (ids, pred, human) in per_dim_scores.items():
        for i, p, h in zip(ids, minmax(pred), minmax(human)):
            points.setdefault((i, level), []).append((p, h))
    rng = random.Random(seed)
    xs, ys, lv = [], [], []
    for level in LEVELS:
        keys = [k for k in points if k[1] == level]
        for k in rng.sample(keys, min(n, len(keys))):
            xs.append(100 * np.mean([p for p, _ in points[k]]))
            ys.append(100 * np.mean([h for _, h in points[k]]))
            lv.append(level)
    xs, ys = np.array(xs), np.array(ys)
    mse = float(np.mean((xs - ys) ** 2)) if len(xs) else float("nan")
    return {"mse_0_100": mse, "acc": 1 - mse / 1e4, "points": list(zip(xs.tolist(), ys.tolist(), lv))}


def print_table(name, table, per_dim):
    print(f"\n{name}: r / rho (%)")
    for d, v in per_dim.items():
        sig = "" if v["p_pearson"] < 0.05 else "*"
        extra = f"  {v['pred']}" if "pred" in v else ""
        print(f"  {d:32s} {100 * v['pearson']:6.1f}{sig}/{100 * v['spearman']:5.1f}  (n={v['n']}){extra}")
    print("  " + " | ".join(f"{k}: {100 * v['pearson']:.1f}/{100 * v['spearman']:.1f}" for k, v in table.items()))


def run(model, rows, spec, selector, batch_size=32, max_len=512, score_mode="argmax", two_level=False, device=None):
    import torch
    device = device or next(model.parameters()).device
    model.eval()
    per_dim, raw = {}, {}
    for level, dim, items in build_items(rows, spec, selector):
        preds = []
        for i in range(0, len(items), batch_size):
            batch = {k: v.to(device) for k, v in model.collate(items[i:i + batch_size], max_len, two_level).items()}
            with torch.no_grad():
                s, _ = model.score(batch, mode=score_mode)
            preds += s.float().cpu().tolist()
        human = [it["human"] for it in items]
        per_dim[f"{level}/{dim}"] = {"level": level, "task": items[0]["task"], **correlations(preds, human),
                                     "pred": pred_summary(preds, score_mode)}
        raw[(level, dim)] = ([it["id"] for it in items], preds, human)
    return per_dim, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="trained UniDE directory")
    ap.add_argument("--vanilla", help="evaluate an untrained backbone with verbaliser heads")
    ap.add_argument("--tokenizer", default=None, help="with --vanilla: load the tokenizer from here instead")
    ap.add_argument("--benchmark", choices=list(SPECS), required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--selector", choices=["manual", "auto"], default="manual")
    ap.add_argument("--score_mode", choices=["argmax", "expected"], default="argmax",
                    help="argmax = Eqs. 13-14 of the paper; expected = sum_y y*p(y) (fewer ties)")
    ap.add_argument("--two_level", action="store_true")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--out", default=None)
    ap.add_argument("--plot", default=None, help="save the Fig. 6 style scatter plot here")
    args = ap.parse_args()

    import torch
    from .model import UniDE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.model:
        model = UniDE.load(args.model)
    else:
        model = UniDE(args.vanilla, ["flu", "coh", "eng"], "mlm", tokenizer_path=args.tokenizer)
    model.to(device)
    grouping = {"flu": "task", "all": "single", "response": "level"}[model.tasks[0]]
    selector = TaskSelector(grouping, args.selector)

    rows = load_unified(args.data)
    per_dim, raw = run(model, rows, SPECS[args.benchmark], selector, args.batch_size, args.max_len,
                       args.score_mode, args.two_level, device)
    table = aggregate(per_dim)
    print_table(args.benchmark, table, per_dim)
    mse = mse_analysis(raw)
    print(f"\nSec. 5.2 scoring accuracy: MSE = {mse['mse_0_100']:.2f} (0-100 scale), ACC = {mse['acc']:.4f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"per_dimension": per_dim, "levels": table, "mse": mse}, f, indent=2)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        for lvl, mk in zip(LEVELS, "os^"):
            pts = [(x, y) for x, y, l in mse["points"] if l == lvl]
            if pts:
                ax.scatter(*zip(*pts), marker=mk, label=f"{lvl}-level", alpha=0.7)
        ax.plot([0, 100], [0, 100], "r--", label="Gold line")
        ax.set_xlabel("Metric Scores"); ax.set_ylabel("Human Annotations"); ax.legend()
        fig.savefig(args.plot, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    main()
