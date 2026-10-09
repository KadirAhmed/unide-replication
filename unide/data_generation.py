"""UniDE-data construction (Sec. 3.2, Fig. 2).

Step 1  Data generation   - a generative LLM writes 4-turn Human/Chatbot conversations that realise a preset
                            quality plan (bad / moderate / good per level and dimension); replies whose
                            response-level target is not good get a focused rewrite that adds the wording problems.
Step 2  Data validation   - a verifiable LLM scores the same conversations blind (1/2/3) with the SAME rubric;
                            conversations whose verified quality contradicts the plan are filtered out.
Build                     - each dialogue is split into 1 dialogue-level, 4 turn-level and 4 response-level
                            instances annotated with the 6 representative dimensions; train/val split by dialogue.
Stats / view              - agreement, confusion tables and removal rates; a readable dump for manual checks.

Usage
  python -m unide.data_generation generate --n 50 --out data/pilot_raw.jsonl
  python -m unide.data_generation verify   --inp data/pilot_raw.jsonl --out data/pilot_verified.jsonl
  python -m unide.data_generation verify   --inp data/pilot_raw.jsonl --out data/pilot_ver_unit.jsonl --mode unit
  python -m unide.data_generation stats    --inp data/pilot_verified.jsonl
  python -m unide.data_generation compare  --inp data/pilot_verified.jsonl data/pilot_ver_unit.jsonl
  python -m unide.data_generation view     --inp data/pilot_verified.jsonl --out data/pilot_view.txt
  python -m unide.data_generation build    --inp data/verified.jsonl --out_dir data/unide_data --n_val 500
  python -m unide.data_generation build    --inp data/verified.jsonl --out_dir data/unide_data_nofilter --filter none
"""
import argparse
import json
import os
import random
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from .dimensions import REPRESENTATIVE
from .llm import ChatLLM, parse_json

DEFAULT_MODEL = "gpt-3.5-turbo-0125"  # the paper's gpt-3.5-turbo-1106 was retired on 2026-09-28
N_TURNS = 4
TOPICS = {"chit-chat": 0.45, "personalized": 0.25, "knowledge-grounded": 0.30}
QUALITY_NAME = {1: "bad", 2: "moderate", 3: "good"}
DIALOGUE_Q = [0.31, 0.23, 0.46]   # P(bad), P(moderate), P(good) of a dialogue (Table 1)
RESPONSE_Q = [0.13, 0.22, 0.65]   # same for a response on its own (Table 1)
TURN_DIMS = REPRESENTATIVE["turn"]          # Relevance, Engagingness
RESP_DIMS = REPRESENTATIVE["response"]      # Fluency, Understandability
DIAL_DIMS = REPRESENTATIVE["dialogue"]      # Coherence, Likeability
ALL_DIMS = DIAL_DIMS + TURN_DIMS + RESP_DIMS

# One shared definition of every score, given to BOTH the generator and the verifier.
RUBRIC = {
    "Coherence": {
        3: "every Chatbot reply follows from what was said before; the conversation flows naturally",
        2: "mostly follows, but exactly one Chatbot reply is off-topic or makes an abrupt jump",
        1: "two or more Chatbot replies ignore what was said; the conversation is hard to follow",
    },
    "Likeability": {
        3: "the Chatbot is warm, friendly and polite throughout",
        2: "the Chatbot's tone is neutral or flat",
        1: "the Chatbot is rude, dismissive or annoying",
    },
    "Relevance": {
        3: "directly addresses what the Human just said",
        2: "on the topic but vague, generic or only partly answers the Human",
        1: "ignores the Human's message or talks about something unrelated",
    },
    "Engagingness": {
        3: "invites the conversation to continue, e.g. with a follow-up question, an interesting detail or enthusiasm",
        2: "acceptable but minimal; gives the Human nothing to build on",
        1: "dull, dismissive or conversation-ending, e.g. 'ok.'",
    },
    "Fluency": {
        3: "natural and grammatical",
        2: "understandable but with awkward phrasing or one small grammar mistake",
        1: "several grammar mistakes, missing words or broken word order",
    },
    "Understandability": {
        3: "its meaning is clear on its own, even without the conversation",
        2: "its meaning is somewhat unclear or ambiguous on its own",
        1: "hard to make sense of on its own (garbled or nonsensical)",
    },
}


# Concrete calibration examples, shown to the generator and the verifier alike.
EXAMPLES = (
    'Example Chatbot replies to the Human message "Can you suggest some resources to learn Python?":\n'
    '- Relevance good, Engagingness good: "Sure! Codecademy and the book Python Crash Course are great for '
    'beginners. Do you prefer videos or books?"\n'
    '- Relevance good, Engagingness moderate: "Codecademy and Python Crash Course are good."\n'
    '- Relevance moderate, Engagingness moderate: "There are lots of resources online."\n'
    '- Relevance bad, Engagingness bad: "My cat likes to sleep all day."\n'
    '- Likeability bad (rude or dismissive tone): "Just search it yourself, it is not hard."'
)
DEGRADE_EXAMPLES = (
    'Example rewrites of the reply "You can try Codecademy or Coursera, they have great beginner courses.":\n'
    '- Fluency moderate, Understandability good: "You can try Codecademy or Coursera, they has great beginner courses."\n'
    '- Fluency moderate, Understandability moderate: "You can try that Codecademy or the other one, they have '
    'courses good for it."\n'
    '- Fluency bad, Understandability moderate: "Codecademy you try or Coursera maybe, beginner great the courses '
    'they having."\n'
    '- Fluency bad, Understandability bad: "Try courses Coursera the beginner or great have Codecademy they for."'
)


# ----------------------------------------------------------------------------------------------------------------
# Quality plans: 18 targets per conversation, kept mutually consistent with the rubric
# ----------------------------------------------------------------------------------------------------------------
def _jitter(q: int, rng: random.Random, p_keep: float = 0.8) -> int:
    if rng.random() < p_keep:
        return q
    return max(1, min(3, q + rng.choice([-1, 1])))


def _turn_relevance(q_dial: int, rng: random.Random) -> list[int]:
    """Per-turn Relevance that matches the dialogue's Coherence rubric."""
    rel = [3] * N_TURNS
    order = list(range(N_TURNS))
    rng.shuffle(order)
    if q_dial == 3:                       # all replies follow; at most one is merely vague
        if rng.random() < 0.5:
            rel[order[0]] = 2
    elif q_dial == 2:                     # exactly one off-topic reply
        rel[order[0]] = 1
        for i in order[1:]:
            rel[i] = 2 if rng.random() < 0.4 else 3
    else:                                 # two or three off-topic replies
        k = rng.choice([2, 3])
        for i in order[:k]:
            rel[i] = 1
        for i in order[k:]:
            rel[i] = 2 if rng.random() < 0.5 else 3
    return rel


def sample_plan(rng: random.Random) -> dict:
    """Preset [level][dimension][quality] targets for one conversation."""
    topic = rng.choices(list(TOPICS), weights=list(TOPICS.values()))[0]
    q_dial = rng.choices([1, 2, 3], weights=DIALOGUE_Q)[0]
    like = _jitter(q_dial, rng)           # the Chatbot's tone across the whole conversation
    turns = []
    for rel in _turn_relevance(q_dial, rng):
        # response-level quality is sampled independently of the context
        flu = rng.choices([1, 2, 3], weights=RESPONSE_Q)[0]
        und = {1: rng.choices([1, 2], weights=[0.7, 0.3])[0], 2: rng.choice([2, 3]), 3: 3}[flu]
        if und == 1:
            rel = min(rel, 2)             # a garbled reply cannot clearly address the Human
        eng = {1: rng.choices([1, 2], weights=[0.8, 0.2])[0],
               2: rng.choices([1, 2, 3], weights=[0.2, 0.7, 0.1])[0],
               3: rng.choices([2, 3], weights=[0.4, 0.6])[0]}[rel]
        if like == 1 or und == 1:
            eng = min(eng, 2)             # a rude or garbled reply is not engaging
        turns.append({"Relevance": rel, "Engagingness": eng, "Fluency": flu, "Understandability": und})
    return {"topic": topic, "dialogue": {"Coherence": q_dial, "Likeability": like}, "turns": turns}


# ----------------------------------------------------------------------------------------------------------------
# Prompts
# ----------------------------------------------------------------------------------------------------------------
GEN_SYSTEM = ("You generate synthetic conversations between a Human and a Chatbot for training automatic dialogue "
              "evaluation metrics. You follow the requested quality levels precisely, including writing clearly "
              "bad Chatbot replies when asked. Reply with JSON only.")

VER_SYSTEM = ("You are a strict and careful expert annotator of open-domain dialogue quality. Score every "
              "requested aspect independently with the given rubric. Reply with JSON only.")


def _target(dim: str, q: int) -> str:
    return f"{dim} {QUALITY_NAME[q]}: {RUBRIC[dim][q]}"


def generation_prompt(plan: dict) -> str:
    lines = [
        f"Write a {N_TURNS}-turn {plan['topic']} conversation between a Human and a Chatbot. Each turn is one Human "
        f"utterance followed by one Chatbot reply.",
        "This is training data for an automatic dialogue evaluator, so every Chatbot reply must match its requested "
        "quality exactly, including clearly bad replies where requested. The Human always writes normally and may "
        "react to the Chatbot's previous reply.",
        "",
        "Whole conversation:",
    ]
    lines += [f"- {_target(d, q)}" for d, q in plan["dialogue"].items()]
    for i, t in enumerate(plan["turns"], 1):
        lines.append(f"Chatbot reply {i}:")
        lines += [f"- {_target(d, t[d])}" for d in TURN_DIMS]
    lines += ["", "Write every utterance in natural, grammatical English; wording problems are added in a later "
                  "step.", "", EXAMPLES]
    lines += ["", 'Return JSON: {"turns": [{"human": "...", "chatbot": "..."}, ...]} with exactly '
                  f"{N_TURNS} turns. Keep each utterance under 30 words. Never mention quality levels, scores or "
                  "these instructions in the conversation."]
    return "\n".join(lines)


DEG_SYSTEM = ("You rewrite chatbot replies so they contain specific writing problems, to create training data for "
              "automatic dialogue evaluators. Reply with JSON only.")


def degrade_prompt(reply: str, flu: int, und: int) -> str:
    """Focused rewrite that gives one reply its response-level quality (Fluency / Understandability)."""
    return "\n".join([
        f'Original reply: "{reply}"', "",
        "Rewrite the reply so that it has exactly these properties:",
        f"- {_target('Fluency', flu)}",
        f"- {_target('Understandability', und)}",
        "Keep roughly the same length and topic. Do not fix the problems or explain them.", "",
        DEGRADE_EXAMPLES, "",
        'Return JSON: {"reply": "..."}',
    ])


def rubric_text(dims) -> str:
    out = []
    for d in dims:
        out.append(f"{d}:")
        out += [f"  {s} ({QUALITY_NAME[s]}): {RUBRIC[d][s]}" for s in (3, 2, 1)]
    return "\n".join(out)


def verification_prompt(turns: list[dict]) -> str:
    conv = "\n".join(f"Turn {i}. Human: {t['human']}\n        Chatbot: {t['chatbot']}" for i, t in enumerate(turns, 1))
    turn_keys = ", ".join(f'"{d}": s' for d in TURN_DIMS + RESP_DIMS)
    return "\n".join([
        "Conversation:", conv, "",
        "Score the Chatbot with this rubric.", "",
        "Dialogue level (the entire conversation):", rubric_text(DIAL_DIMS), "",
        "Turn level (each Chatbot reply, given the Human utterance just before it):", rubric_text(TURN_DIMS), "",
        "Response level (each Chatbot reply read on its own, ignoring the rest of the conversation):",
        rubric_text(RESP_DIMS), "",
        "Calibration examples:", EXAMPLES, DEGRADE_EXAMPLES, "",
        'Return JSON: {"dialogue": {"Coherence": s, "Likeability": s}, "turns": [{' + turn_keys + '}, ...]} '
        "with one entry per turn, where every s is 1, 2 or 3.",
    ])


def unit_dialogue_prompt(turns: list[dict]) -> str:
    conv = "\n".join(f"Turn {i}. Human: {t['human']}\n        Chatbot: {t['chatbot']}" for i, t in enumerate(turns, 1))
    return "\n".join(["Dialogue to score:", conv, "",
                      "Score the Chatbot over the entire conversation with this rubric.", rubric_text(DIAL_DIMS), "",
                      "Calibration examples:", EXAMPLES, "",
                      'Return JSON: {"Coherence": s, "Likeability": s} where every s is 1, 2 or 3.'])


def unit_turn_prompt(turn: dict) -> str:
    return "\n".join(["Exchange to score:", f"Human: {turn['human']}", f"Chatbot: {turn['chatbot']}", "",
                      "Score the Chatbot's reply, given the Human utterance just before it, with this rubric.",
                      rubric_text(TURN_DIMS), "", "Calibration examples:", EXAMPLES, "",
                      'Return JSON: {"Relevance": s, "Engagingness": s} where every s is 1, 2 or 3.'])


def unit_response_prompt(turn: dict) -> str:
    return "\n".join(["Reply to score:", f'"{turn["chatbot"]}"', "",
                      "Score this reply on its own (there is no other context) with this rubric.",
                      rubric_text(RESP_DIMS), "", "Calibration examples:", DEGRADE_EXAMPLES, "",
                      'Return JSON: {"Fluency": s, "Understandability": s} where every s is 1, 2 or 3.'])


# ----------------------------------------------------------------------------------------------------------------
# Step 1 / Step 2
# ----------------------------------------------------------------------------------------------------------------
def generate_one(llm, plan: dict) -> dict | None:
    out = parse_json(llm(GEN_SYSTEM, generation_prompt(plan)))
    turns = out.get("turns", [])
    if len(turns) != N_TURNS or not all(t.get("human") and t.get("chatbot") for t in turns):
        return None
    turns = [{"human": t["human"].strip(), "chatbot": t["chatbot"].strip()} for t in turns]
    for t, target in zip(turns, plan["turns"]):
        flu, und = target["Fluency"], target["Understandability"]
        if (flu, und) != (3, 3):
            new = degrade_reply(llm, t["chatbot"], flu, und)
            if new is None:
                return None
            t["chatbot_original"], t["chatbot"] = t["chatbot"], new
    return {"plan": plan, "turns": turns}


def degrade_reply(llm, reply: str, flu: int, und: int, tries: int = 2) -> str | None:
    for _ in range(tries):
        out = parse_json(llm(DEG_SYSTEM, degrade_prompt(reply, flu, und), temperature=0.7))
        new = str(out.get("reply", "")).strip()
        if new and new.lower() != reply.lower():
            return new
    return None


def verify_one(llm, inst: dict) -> dict | None:
    out = parse_json(llm(VER_SYSTEM, verification_prompt(inst["turns"]), temperature=0.0))
    try:
        ver = {"dialogue": {d: int(out["dialogue"][d]) for d in DIAL_DIMS},
               "turns": [{d: int(t[d]) for d in TURN_DIMS + RESP_DIMS} for t in out["turns"]]}
    except (KeyError, TypeError, ValueError):
        return None
    if len(ver["turns"]) != N_TURNS or not all(s in (1, 2, 3) for s in _all_scores(ver)):
        return None
    return {**inst, "verified": ver}


def _ask_scores(llm, prompt: str, dims) -> dict | None:
    out = parse_json(llm(VER_SYSTEM, prompt, temperature=0.0))
    try:
        scores = {d: int(out[d]) for d in dims}
    except (KeyError, TypeError, ValueError):
        return None
    return scores if all(v in (1, 2, 3) for v in scores.values()) else None


def verify_units(llm, inst: dict) -> dict | None:
    """One focused call per unit: the dialogue, each exchange, each reply on its own (9 calls)."""
    dial = _ask_scores(llm, unit_dialogue_prompt(inst["turns"]), DIAL_DIMS)
    turns = []
    for t in inst["turns"]:
        a = _ask_scores(llm, unit_turn_prompt(t), TURN_DIMS)
        b = _ask_scores(llm, unit_response_prompt(t), RESP_DIMS)
        if a is None or b is None:
            return None
        turns.append({**a, **b})
    if dial is None:
        return None
    return {**inst, "verified": {"dialogue": dial, "turns": turns}}


def _all_scores(scores: dict) -> list[int]:
    return list(scores["dialogue"].values()) + [s for t in scores["turns"] for s in t.values()]


def label_gaps(inst: dict) -> list[int]:
    """|preset - verified| for all 2 + 4*4 = 18 labels of a conversation."""
    p, v = inst["plan"], inst["verified"]
    gaps = [abs(p["dialogue"][d] - v["dialogue"][d]) for d in DIAL_DIMS]
    for pt, vt in zip(p["turns"], v["turns"]):
        gaps += [abs(pt[d] - vt[d]) for d in TURN_DIMS + RESP_DIMS]
    return gaps


def is_valid(inst: dict, max_gap: int = 1, max_mismatches: int = 3) -> bool:
    """Instance filtering: drop instances whose verified quality contradicts the preset quality
    (e.g. preset bad but verified good, as S_n in Fig. 2a)."""
    gaps = label_gaps(inst)
    return max(gaps) <= max_gap and sum(g > 0 for g in gaps) <= max_mismatches


def _run_parallel(fn, items, out_path, workers):
    lock = threading.Lock()
    n_ok = 0
    with open(out_path, "a", encoding="utf-8") as f, ThreadPoolExecutor(workers) as ex:
        futures = [ex.submit(fn, it) for it in items]
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                res = fut.result()
            except Exception as e:
                print(f"[warn] {type(e).__name__}: {e}")
                res = None
            if res is not None:
                with lock:
                    f.write(json.dumps(res, ensure_ascii=False) + "\n")
                    f.flush()
                n_ok += 1
            if i % 100 == 0:
                print(f"  {i}/{len(items)} done, {n_ok} kept")
    return n_ok


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def cmd_generate(args, llm=None):
    llm = llm or ChatLLM(args.model, temperature=args.temperature)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    done = len(read_jsonl(args.out))  # resumable
    rng = random.Random(args.seed + done)
    plans = [sample_plan(rng) for _ in range(max(0, args.n - done))]
    print(f"generating {len(plans)} conversations ({done} already in {args.out})")
    _run_parallel(lambda p: generate_one(llm, p), plans, args.out, args.workers)


def cmd_verify(args, llm=None):
    llm = llm or ChatLLM(args.model, temperature=0.0)
    raw = read_jsonl(args.inp)
    for i, r in enumerate(raw):
        r.setdefault("id", i)
    done_ids = {r["id"] for r in read_jsonl(args.out)}
    todo = [r for r in raw if r["id"] not in done_ids]
    mode = getattr(args, "mode", "conversation")
    fn = verify_units if mode == "unit" else verify_one
    print(f"verifying {len(todo)} conversations ({mode} mode, {args.model})")

    def work(r):
        res = fn(llm, r)
        return None if res is None else {**res, "verifier": {"model": args.model, "mode": mode}}
    _run_parallel(work, todo, args.out, args.workers)


# ----------------------------------------------------------------------------------------------------------------
# Inspection: stats and view
# ----------------------------------------------------------------------------------------------------------------
def _pairs(data, d):
    if d in DIAL_DIMS:
        return [(x["plan"]["dialogue"][d], x["verified"]["dialogue"][d]) for x in data]
    return [(p[d], v[d]) for x in data for p, v in zip(x["plan"]["turns"], x["verified"]["turns"])]


def cmd_stats(args):
    data = read_jsonl(args.inp)
    if not data:
        print(f"no verified conversations in {args.inp}")
        return
    print(f"{len(data)} verified conversations\n\nexact agreement between plan and verifier:")
    for d in ALL_DIMS:
        pairs = _pairs(data, d)
        print(f"  {d:18s} {sum(a == b for a, b in pairs) / len(pairs):5.0%}")
    print("\nconfusion tables (rows = plan 1/2/3, columns = verifier said 1/2/3):")
    for d in ALL_DIMS:
        c = Counter(_pairs(data, d))
        print(f"  {d}")
        for t in (1, 2, 3):
            print(f"    plan {t}: " + " ".join(f"{c[(t, v)]:4d}" for v in (1, 2, 3)))
    flips = sum(max(label_gaps(x)) == 2 for x in data)
    print(f"\nconversations with at least one bad<->good flip: {flips} ({flips / len(data):.0%})")
    print("share removed by the filter (the paper removed 8.5%):")
    for gap, mis in [(0, 0), (1, 0), (1, 2), (1, 3), (1, 5), (1, 8), (1, 18)]:
        removed = 1 - sum(is_valid(x, gap, mis) for x in data) / len(data)
        print(f"  --max_gap {gap} --max_mismatches {mis:2d}  ->  {removed:6.1%}")


def cmd_compare(args):
    """Side-by-side comparison of verification runs of the SAME generated conversations."""
    runs = {os.path.basename(p): {x["id"]: x for x in read_jsonl(p)} for p in args.inp}
    names = list(runs)
    common = set.intersection(*(set(r) for r in runs.values()))
    if not common:
        print("no conversation ids in common")
        return
    ids = sorted(common)
    tag = {n: chr(65 + i) for i, n in enumerate(names)}
    for n in names:
        v = next(iter(runs[n].values())).get("verifier", {})
        print(f"  {tag[n]} = {n}  ({v.get('model', '?')}, {v.get('mode', 'conversation')} mode)")
    print(f"  {len(ids)} conversations in common\n")
    print(f"{'agreement with plan':22s}" + "".join(f"{tag[n]:>8s}" for n in names))
    for d in ALL_DIMS:
        row = [_pairs([runs[n][i] for i in ids], d) for n in names]
        print(f"  {d:20s}" + "".join(f"{sum(a == b for a, b in r) / len(r):8.0%}" for r in row))
    flips = [sum(max(label_gaps(runs[n][i])) == 2 for i in ids) / len(ids) for n in names]
    kept = [sum(is_valid(runs[n][i]) for i in ids) / len(ids) for n in names]
    print(f"  {'bad<->good flips':20s}" + "".join(f"{f:8.0%}" for f in flips))
    print(f"  {'removed by default':20s}" + "".join(f"{1 - k:8.0%}" for k in kept))
    if len(names) > 1:
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
        print(f"\n{'verifier vs verifier':22s}" + "".join(f"{tag[a] + '~' + tag[b]:>8s}" for a, b in pairs))
        for d in ALL_DIMS:
            vals = []
            for a, b in pairs:
                sa = [s for _, s in _pairs([runs[a][i] for i in ids], d)]
                sb = [s for _, s in _pairs([runs[b][i] for i in ids], d)]
                vals.append(sum(x == y for x, y in zip(sa, sb)) / len(sa))
            print(f"  {d:20s}" + "".join(f"{v:8.0%}" for v in vals))


def _fmt(plan_scores: dict, ver_scores: dict | None) -> str:
    parts = []
    for d, q in plan_scores.items():
        s = f"{d}={QUALITY_NAME[q]}"
        if ver_scores is not None and ver_scores.get(d) != q:
            s += f" [verifier: {QUALITY_NAME.get(ver_scores.get(d), '?')}]"
        parts.append(s)
    return ", ".join(parts)


def cmd_view(args):
    rows = read_jsonl(args.inp)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("Targets from the plan; [verifier: ...] marks where the verifier disagreed.\n")
        for i, x in enumerate(rows):
            p, v = x["plan"], x.get("verified")
            f.write(f"\n#{i} {p['topic']} | {_fmt(p['dialogue'], v['dialogue'] if v else None)}\n")
            for j, (t, q) in enumerate(zip(x["turns"], p["turns"])):
                f.write(f"  Human:   {t['human']}\n  Chatbot: {t['chatbot']}\n")
                if "chatbot_original" in t:
                    f.write(f"      (before rewrite: {t['chatbot_original']})\n")
                f.write(f"      {_fmt(q, v['turns'][j] if v else None)}\n")
    print(f"wrote {len(rows)} conversations to {args.out}")


# ----------------------------------------------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------------------------------------------
def flatten(inst: dict, label_source: str = "plan", keep: str = "all") -> list[dict]:
    """Template-ready instances of all three levels (Table 13).

    label_source: "plan" = preset quality (paper) or "verifier" = the verifier's blind scores.
    keep: "all"    = every label;
          "noflip" = drop labels where the verifier says the opposite of the plan (bad vs good), the
                     inconsistency shown in the paper's Fig. 2a; one-step disagreements are kept;
          "agree"  = only labels where plan and verifier give exactly the same score (strict).
    """
    did, plan, turns = inst["id"], inst["plan"], inst["turns"]
    lab = plan if label_source == "plan" else inst["verified"]
    ver = inst.get("verified")

    def ok(p_score, v_score):
        if keep == "all":
            return True
        return p_score == v_score if keep == "agree" else abs(p_score - v_score) <= 1

    utts = [u for t in turns for u in (f"Human: {t['human']}", f"Chatbot: {t['chatbot']}")]
    rows = [{"dialogue_id": did, "level": "dialogue", "dimension": d, "context": utts, "response": None,
             "label": lab["dialogue"][d], "topic": plan["topic"]}
            for d in DIAL_DIMS if ok(plan["dialogue"][d], ver and ver["dialogue"][d])]
    for i, (t, lt) in enumerate(zip(turns, lab["turns"])):
        pt, vt = plan["turns"][i], (ver["turns"][i] if ver else {})
        for d in TURN_DIMS:
            if ok(pt[d], vt.get(d)):
                rows.append({"dialogue_id": did, "turn": i, "level": "turn", "dimension": d,
                             "context": [f"Human: {t['human']}"], "response": f"Chatbot: {t['chatbot']}",
                             "label": lt[d], "topic": plan["topic"]})
        for d in RESP_DIMS:
            if ok(pt[d], vt.get(d)):
                rows.append({"dialogue_id": did, "turn": i, "level": "response", "dimension": d,
                             "context": [], "response": t["chatbot"], "label": lt[d]})
    return rows


def cmd_build(args):
    label_source = getattr(args, "label_source", "plan")
    mode = "none" if getattr(args, "no_filter", False) else getattr(args, "filter", "conversation")
    data = read_jsonl(args.inp)
    for i, r in enumerate(data):
        r.setdefault("id", i)
    if mode == "conversation":
        kept = [r for r in data if is_valid(r, args.max_gap, args.max_mismatches)]
        print(f"conversation filtering: {len(data)} -> {len(kept)} "
              f"(reduction {100 * (1 - len(kept) / max(1, len(data))):.2f}%)")
        data = kept
    if args.target and len(data) > args.target:
        data = data[: args.target]
    elif args.target and len(data) < args.target:
        print(f"[note] only {len(data)} conversations available (target {args.target}); generate more to reach it")
    keep = mode if mode in ("agree", "noflip") else "all"
    rng = random.Random(args.seed)
    rng.shuffle(data)
    val, train = data[: args.n_val], data[args.n_val:]
    os.makedirs(args.out_dir, exist_ok=True)
    for name, split in (("train", train), ("val", val)):
        rows = [row for inst in split for row in flatten(inst, label_source, keep)]
        with open(os.path.join(args.out_dir, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(split)} dialogues -> {len(rows)} labelled instances")
    _print_stats(train + val, label_source, keep)


def _print_stats(data, label_source="plan", keep="all"):
    kept = [r for inst in data for r in flatten(inst, label_source, keep)]
    total = Counter(r["dimension"] for inst in data for r in flatten(inst, label_source, "all"))
    if keep != "all":
        print(f"label filtering ({keep}) kept {len(kept)} of {sum(total.values())} labels "
              f"({len(kept) / max(1, sum(total.values())):.0%})")
    print(f"  {'':18s} {'kept':>6s}  {'share':>5s}  bad/moderate/good")
    for d in ALL_DIMS:
        labels = [r["label"] for r in kept if r["dimension"] == d]
        n = max(1, len(labels))
        dist = "/".join(f"{100 * labels.count(q) / n:.0f}%" for q in (1, 2, 3))
        print(f"  {d:18s} {len(labels):6d}  {len(labels) / max(1, total[d]):5.0%}  {dist}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--n", type=int, default=8200)
    g.add_argument("--out", default="data/raw_generated.jsonl")
    g.add_argument("--temperature", type=float, default=0.8)
    v = sub.add_parser("verify")
    v.add_argument("--inp", default="data/raw_generated.jsonl")
    v.add_argument("--out", default="data/verified.jsonl")
    v.add_argument("--mode", choices=["conversation", "unit"], default="conversation",
                   help="conversation = one call scores all 18 labels; unit = one call per dialogue/exchange/reply")
    for p in (g, v):
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("--inp", nargs="+", required=True, help="verified files of the same raw conversations")
    s = sub.add_parser("stats")
    s.add_argument("--inp", default="data/verified.jsonl")
    w = sub.add_parser("view")
    w.add_argument("--inp", default="data/verified.jsonl")
    w.add_argument("--out", default="data/view.txt")
    b = sub.add_parser("build")
    b.add_argument("--inp", default="data/verified.jsonl")
    b.add_argument("--out_dir", default="data/unide_data")
    b.add_argument("--n_val", type=int, default=500)
    b.add_argument("--target", type=int, default=0, help="number of dialogues to keep (0 = all)")
    b.add_argument("--filter", choices=["noflip", "agree", "conversation", "none"], default="noflip",
                   help="noflip = drop labels where the verifier says the opposite of the plan (default, closest to "
                        "the paper); agree = keep only labels where plan and verifier agree exactly; conversation = "
                        "drop whole conversations using --max_gap/--max_mismatches; none = keep everything")
    b.add_argument("--no_filter", action="store_true", help="same as --filter none (w/o instance filtering ablation)")
    b.add_argument("--max_gap", type=int, default=1)
    b.add_argument("--max_mismatches", type=int, default=3)
    b.add_argument("--label_source", choices=["plan", "verifier"], default="plan",
                   help="plan = preset quality (paper); verifier = the verifier's blind scores")
    b.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"generate": cmd_generate, "verify": cmd_verify, "stats": cmd_stats, "view": cmd_view,
     "compare": cmd_compare, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
