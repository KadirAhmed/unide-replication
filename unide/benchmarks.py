"""Test benchmarks (Sec. 4.1, Appendix C) converted to one unified JSONL format:

  {"id": str, "type": "turn" | "dialogue", "context": [utterances], "response": str | null,
   "annotations": {raw_annotation_key: mean human score}}

Sources
  FED          http://shikib.com/fed_data.json                   -> python -m unide.benchmarks fed  fed_data.json  data/bench/fed.jsonl
  Persona-usr  https://shikib.com/pc_usr_data.json (USR)          -> python -m unide.benchmarks usr  pc_usr_data.json data/bench/persona_usr.jsonl
  Topical-usr  https://shikib.com/tc_usr_data.json (USR, dev only) -> python -m unide.benchmarks usr  tc_usr_data.json data/bench/tc_usr.jsonl
  Persona-see  See et al. 2019 human-evaluation logs released with ParlAI (controllable_dialogue project):
               https://parl.ai/downloads/controllable_dialogue/evaluation_logs_reproducible_v1.tar.gz, extracted
                                                                 -> python -m unide.benchmarks parlai <extracted folder> data/bench/persona_see.jsonl
               or the "Persona-Chatlog" file of the DSTC10 package -> python -m unide.benchmarks dstc10 <file> data/bench/persona_see.jsonl
               (or a generic JSON list of {"dialog": [...], "fluency": x, ...} -> python -m unide.benchmarks see ...)
"""
import json
import os
import sys
from collections import Counter

import numpy as np

# (UniDE level, UniDE dimension, source instance type, raw annotation key)   -- Table 14
FED_SPEC = [
    ("response", "Fluency", "turn", "Fluent"),
    ("response", "Correctness", "turn", "Correct"),
    ("response", "Specificity", "turn", "Specific"),
    ("response", "Understandability", "turn", "Understandable"),
    ("turn", "Interestingness", "turn", "Interesting"),
    ("turn", "Relevance", "turn", "Relevant"),
    ("turn", "Appropriateness", "turn", "Semantically appropriate"),
    ("turn", "Engagingness", "turn", "Engaging"),
    ("dialogue", "Coherence", "dialogue", "Coherent"),
    ("dialogue", "Consistency", "dialogue", "Consistent"),
    ("dialogue", "Diversity", "dialogue", "Diverse"),
    ("dialogue", "Topic depth", "dialogue", "Depth"),
    ("dialogue", "Likeability", "dialogue", "Likeable"),
    ("dialogue", "Informativeness", "dialogue", "Informative"),
    ("dialogue", "Flexibility", "dialogue", "Flexible"),
    ("dialogue", "Inquisitiveness", "dialogue", "Inquisitive"),
]

# Table 15. USR does not ship a "Relevance" rating; the row is kept and skipped with a warning unless you add it.
PERSONA_SPEC = [
    ("response", "Understandability", "turn", "Understandable"),
    ("response", "Naturalness", "turn", "Natural"),
    ("turn", "Relevance", "turn", "Relevance"),
    ("turn", "Context maintenance", "turn", "Maintains Context"),
    ("turn", "Engagingness", "turn", "Engaging"),
    ("dialogue", "Fluency", "dialogue", "fluency"),
    ("dialogue", "Listening", "dialogue", "listen"),
    ("dialogue", "Interestingness", "dialogue", "interest"),
    ("dialogue", "Enjoyment", "dialogue", "enjoy"),
    ("dialogue", "Inquisitiveness", "dialogue", "inquisitive"),
]

# Topical-USR (https://shikib.com/tc_usr_data.json): used only as a DEVELOPMENT set for choosing how long to train,
# never as a test set. Turn-level ratings; "Uses Knowledge" and "Overall" have no UniDE counterpart.
TOPICAL_SPEC = [
    ("response", "Understandability", "turn", "Understandable"),
    ("response", "Naturalness", "turn", "Natural"),
    ("turn", "Context maintenance", "turn", "Maintains Context"),
    ("turn", "Engagingness", "turn", "Engaging"),
]

SPECS = {"fed": FED_SPEC, "personachat": PERSONA_SPEC, "topicalchat": TOPICAL_SPEC}


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None  # e.g. FED's "N/A (no errors)" free-text answers


def _mean(v):
    if isinstance(v, (int, float)):
        return float(v)
    vals = [n for n in (_num(x) for x in v) if n is not None]
    return float(np.mean(vals)) if vals else None


def _scores(d: dict, skip=()) -> dict:
    return {k: m for k, v in d.items() if k not in skip and isinstance(v, (list, int, float))
            and (m := _mean(v)) is not None}


# UniDE-data was generated with "Human:" / "Chatbot:" speaker tags; benchmarks are mapped onto the same tags
_TAGS = {"user": "Human", "human": "Human", "system": "Chatbot", "chatbot": "Chatbot", "bot": "Chatbot",
         "model": "Chatbot"}


def _retag(utt: str) -> str:
    head, sep, rest = utt.partition(":")
    if sep and head.strip().lower() in _TAGS:
        return f"{_TAGS[head.strip().lower()]}: {rest.strip()}"
    return utt.strip()


def _split(ctx):
    if isinstance(ctx, list):
        return [c.strip() for c in ctx if c.strip()]
    return [l.strip() for l in ctx.split("\n") if l.strip()]


def _alternate(utts: list[str], last_speaker: str = "Human") -> list[str]:
    """Add speaker tags to untagged utterances, counting back from the last one."""
    other = {"Human": "Chatbot", "Chatbot": "Human"}
    tags, cur = [], last_speaker
    for _ in utts:
        tags.append(cur)
        cur = other[cur]
    return [f"{t}: {u}" for t, u in zip(reversed(tags), utts)]


def convert_fed(entries):
    out = []
    for i, e in enumerate(entries):
        ann = _scores(e.get("annotations", {}))
        ctx = [_retag(u) for u in _split(e["context"])]
        if e.get("response"):
            out.append({"id": f"fed-turn-{i}", "type": "turn", "context": ctx,
                        "response": _retag(e["response"]), "annotations": ann})
        else:
            out.append({"id": f"fed-dial-{i}", "type": "dialogue", "context": ctx, "response": None,
                        "annotations": ann})
    return out


def convert_usr(entries):
    out = []
    for i, e in enumerate(entries):
        ctx = _alternate(_split(e["context"]), last_speaker="Human")
        for j, r in enumerate(e["responses"]):
            out.append({"id": f"usr-{i}-{j}", "type": "turn", "context": ctx,
                        "response": f"Chatbot: {r['response'].strip()}",
                        "annotations": _scores(r, skip=("model", "response"))})
    return out


def convert_see(entries):
    """Generic list of {"dialog": [utterances], "<quality>": score(s), ...}."""
    out = []
    for i, e in enumerate(entries):
        utts = [_retag(u) for u in _split(e["dialog"])]
        if not any(":" in u[:12] for u in utts):
            utts = _alternate(utts, last_speaker="Chatbot")
        out.append({"id": f"see-{i}", "type": "dialogue", "context": utts, "response": None,
                    "annotations": _scores(e, skip=("dialog",))})
    return out


def _speaker_role(name: str, human_like: set) -> str:
    return "Human" if name in human_like else "Chatbot"


def convert_dstc10(entries):
    """DSTC10 dialogue-level files (e.g. Persona-Chatlog = Persona-see):
    {"dialogue_id", "model", "dialogue": [{"speaker", "text"}], "annotations": {quality: [scores]}}."""
    speakers = Counter(t.get("speaker", "") for e in entries for t in e.get("dialogue", []))
    hints = ("human", "user", "person", "evaluator", "worker", "turker")
    human_like = {s for s in speakers if any(h in str(s).lower() for h in hints)}
    print(f"speakers found: {dict(speakers)} -> treated as Human: {sorted(human_like) or 'none'}")
    if not human_like:
        print("[warn] could not tell which speaker is the human; tags fall back to alternating turns")
    out = []
    for i, e in enumerate(entries):
        turns = [t for t in e.get("dialogue", []) if str(t.get("text", "")).strip()]
        if human_like:
            utts = [f"{_speaker_role(t.get('speaker', ''), human_like)}: {str(t['text']).strip()}" for t in turns]
        else:
            utts = _alternate([str(t["text"]).strip() for t in turns], last_speaker="Chatbot")
        out.append({"id": f"dstc10-{e.get('dialogue_id', i)}", "type": "dialogue", "context": utts,
                    "response": None, "annotations": _scores(e.get("annotations", {}))})
    return out


# Quality names used in the See et al. (2019) human evaluation (ParlAI controllable_dialogue logs)
_SEE_QUALITIES = {"fluency", "listen", "interest", "enjoy", "inquisitive", "turing", "make_sense", "avoid_rep",
                  "repetitive", "humanness", "engagingness", "persona_guess"}


def _iter_json_records(path):
    """Yield (file, record) for every JSON object in a file or a directory tree of .json / .jsonl files."""
    import glob
    import os
    files = [path] if os.path.isfile(path) else sorted(
        f for f in glob.glob(os.path.join(path, "**", "*"), recursive=True)
        if os.path.isfile(f) and (f.endswith(".json") or f.endswith(".jsonl") or "." not in os.path.basename(f)))
    for fpath in files:
        try:
            with open(fpath, encoding="utf-8") as f:
                text = f.read().strip()
        except (UnicodeDecodeError, OSError):
            continue
        if not text or text[0] not in "[{":
            continue
        try:
            obj = json.loads(text)
            items = obj if isinstance(obj, list) else [obj]
        except json.JSONDecodeError:
            items = []
            for line in text.splitlines():
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        for it in items:
            if isinstance(it, dict):
                yield fpath, it


def _find_ratings(rec: dict):
    if sum(k.lower() in _SEE_QUALITIES for k in rec) >= 2:
        return rec
    for v in rec.values():
        if isinstance(v, dict):
            found = _find_ratings(v)
            if found is not None:
                return found
    return None


def _find_turns(rec: dict):
    for key in ("dialog", "dialogue", "conversation", "turns", "messages"):
        if isinstance(rec.get(key), list) and rec[key]:
            return rec[key]
    return None


def _utterances(turns):
    """-> list of (speaker or None, text) from dicts, [a, b] pairs or plain strings."""
    out = []
    for t in turns:
        if isinstance(t, dict):
            spk = next((t[k] for k in ("speaker", "id", "agent", "role", "sender") if k in t), None)
            txt = next((t[k] for k in ("text", "utterance", "message", "content") if k in t), "")
            out.append((str(spk) if spk is not None else None, str(txt)))
        elif isinstance(t, (list, tuple)):
            out += [(None, str(x)) for x in t]
        else:
            out.append((None, str(t)))
    return [(s, x.strip()) for s, x in out if x.strip() and x.strip() != "__SILENCE__"]


def convert_parlai_logs(path):
    """See et al. (2019) human-evaluation logs (Persona-see): a folder (or file) of JSON/JSONL conversation records,
    each with a dialogue and a dict of quality ratings such as fluency / listen / interest / enjoy / inquisitive."""
    recs = list(_iter_json_records(path))
    files = {f for f, _ in recs}
    parsed, skipped = [], 0
    for fpath, rec in recs:
        turns, ratings = _find_turns(rec), _find_ratings(rec)
        utts = _utterances(turns) if turns else []
        if not utts or ratings is None:
            skipped += 1
            continue
        parsed.append((fpath, rec, utts, ratings))
    speakers = Counter(s for _, _, utts, _ in parsed for s, _ in utts if s is not None)
    hints = ("human", "user", "person", "evaluator", "worker", "turker")
    human_like = {s for s in speakers if any(h in s.lower() for h in hints)}
    print(f"read {len(recs)} records from {len(files)} files; kept {len(parsed)}, skipped {skipped} "
          f"(no dialogue or no ratings)")
    print(f"speakers found: {dict(speakers) or 'none (untagged)'} -> treated as Human: {sorted(human_like) or 'none'}")
    out = []
    for i, (fpath, rec, utts, ratings) in enumerate(parsed):
        if human_like and any(s is not None for s, _ in utts):
            tagged = [f"{'Human' if s in human_like else 'Chatbot'}: {x}" for s, x in utts]
        else:  # untagged record: the human evaluator opens the conversation
            tagged = _alternate([x for _, x in utts], last_speaker="Human" if len(utts) % 2 else "Chatbot")
        out.append({"id": f"see-{i}", "type": "dialogue", "context": tagged, "response": None,
                    "annotations": _scores(ratings), "source": os.path.basename(fpath)})
    return out


def summarize(rows):
    types = Counter(r["type"] for r in rows)
    keys = Counter(k for r in rows for k in r["annotations"])
    print(f"instances: {dict(types)}")
    print("annotation keys (instances having each): " + ", ".join(f"{k} ({n})" for k, n in sorted(keys.items())))


def load_unified(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


if __name__ == "__main__":
    kind, src, dst = sys.argv[1:4]
    if kind == "parlai":
        rows = convert_parlai_logs(src)
    else:
        with open(src, encoding="utf-8") as f:
            data = json.load(f)
        rows = {"fed": convert_fed, "usr": convert_usr, "see": convert_see, "dstc10": convert_dstc10}[kind](data)
    with open(dst, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summarize(rows)
    print(f"wrote {len(rows)} instances to {dst}")
