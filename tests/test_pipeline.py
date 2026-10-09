"""Torch-free tests: data generation (mock LLM), filtering, flattening, grouping, weights, selector, templates,
benchmark conversion and metric aggregation.   Run: python -m pytest tests -q"""
import argparse
import json
import random

import numpy as np

from unide import data_generation as dg
from unide import task_grouping as tg
from unide.benchmarks import convert_fed, convert_usr, convert_dstc10, convert_parlai_logs, FED_SPEC, PERSONA_SPEC
from unide.dimensions import STUDIED_DIMENSIONS, DIMENSIONS
from unide.evaluate import build_items, correlations, aggregate, mse_analysis, pred_summary, print_table
from unide.task_selector import TaskSelector
from unide.templates import build_prompt, SLOT


class MockLLM:
    """Returns a conversation for generation prompts and (mostly agreeing) scores for verification prompts."""

    def __init__(self, seed=0, flip_rate=0.1):
        self.rng, self.flip = random.Random(seed), flip_rate

    def __call__(self, system, user, temperature=None):
        if "Return JSON: {\"turns\": [{\"human\"" in user:
            return json.dumps({"turns": [{"human": f"question {i}?", "chatbot": f"answer {i}."} for i in range(4)]})
        if 'Return JSON: {"reply"' in user:
            self.n_rewrites = getattr(self, "n_rewrites", 0) + 1
            return json.dumps({"reply": "broken " + user.split('"')[1]})
        for head, dims in (("Dialogue to score:", ("Coherence", "Likeability")),
                           ("Exchange to score:", ("Relevance", "Engagingness")),
                           ("Reply to score:", ("Fluency", "Understandability"))):
            if user.startswith(head):
                self.unit_calls = getattr(self, "unit_calls", 0) + 1
                return json.dumps({d: 3 for d in dims})
        if user.startswith("Conversation:"):
            # the mock verifier cannot see the plan, so it echoes a hidden plan registered by the test
            plan = self.current_plan
            f = lambda q: 4 - q if self.rng.random() < self.flip else q  # occasional quality inversion
            return json.dumps({"dialogue": {d: f(q) for d, q in plan["dialogue"].items()},
                               "turns": [{d: f(q) for d, q in t.items()} for t in plan["turns"]]})
        raise AssertionError(user[:80])


def test_generation_verification_build(tmp_path):
    llm, rng = MockLLM(), random.Random(1)
    insts = []
    for i in range(60):
        plan = dg.sample_plan(rng)
        inst = dg.generate_one(llm, plan)
        llm.current_plan = plan
        inst = dg.verify_one(llm, {**inst, "id": i})
        insts.append(inst)
    kept = [x for x in insts if dg.is_valid(x)]
    assert 0 < len(kept) < len(insts)             # inversions are filtered out
    assert all(max(dg.label_gaps(x)) <= 1 for x in kept)
    rows = dg.flatten(kept[0])
    assert len(rows) == 2 + 4 * 2 + 4 * 2          # 2 dialogue + 8 turn + 8 response labels
    assert {r["level"] for r in rows} == {"dialogue", "turn", "response"}
    p = tmp_path / "verified.jsonl"
    p.write_text("\n".join(json.dumps(x) for x in insts))
    args = argparse.Namespace(inp=str(p), out_dir=str(tmp_path / "out"), n_val=5, target=0, no_filter=False,
                              filter="conversation", max_gap=1, max_mismatches=3, seed=0, label_source="plan")
    dg.cmd_build(args)
    train = [json.loads(l) for l in open(tmp_path / "out" / "train.jsonl")]
    val_ids = {json.loads(l)["dialogue_id"] for l in open(tmp_path / "out" / "val.jsonl")}
    assert not val_ids & {r["dialogue_id"] for r in train}   # split by dialogue


def test_prompt_mentions_every_target():
    plan = dg.sample_plan(random.Random(3))
    text = dg.generation_prompt(plan)
    assert text.count("Relevance good") + text.count("Relevance moderate") + text.count("Relevance bad") >= 4
    assert "Whole conversation:" in text and dg.EXAMPLES in text
    for t in plan["turns"]:  # turn-level targets are spelled out; response-level ones go to the rewrite step
        assert all(dg.RUBRIC[d][t[d]] in text for d in dg.TURN_DIMS)
    deg = dg.degrade_prompt("Hello there.", 1, 2)
    assert dg.RUBRIC["Fluency"][1] in deg and dg.RUBRIC["Understandability"][2] in deg and '"Hello there."' in deg
    ver = dg.verification_prompt([{"human": "hi", "chatbot": "hello"}] * 4)
    assert ver.startswith("Conversation:") and all(dg.RUBRIC[d][s] in ver for d in dg.ALL_DIMS for s in (1, 2, 3))


def test_plans_follow_the_rubric():
    rng = random.Random(0)
    plans = [dg.sample_plan(rng) for _ in range(3000)]
    for p in plans:
        coh, rel = p["dialogue"]["Coherence"], [t["Relevance"] for t in p["turns"]]
        assert {3: rel.count(1) == 0, 2: rel.count(1) == 1, 1: rel.count(1) >= 2}[coh]
        for t in p["turns"]:
            assert t["Fluency"] < 3 or t["Understandability"] == 3
            assert p["dialogue"]["Likeability"] > 1 or t["Engagingness"] <= 2
            assert t["Understandability"] > 1 or (t["Relevance"] <= 2 and t["Engagingness"] <= 2)
    share = lambda lvl_vals, q: sum(v == q for v in lvl_vals) / len(lvl_vals)
    coh = [p["dialogue"]["Coherence"] for p in plans]
    assert abs(share(coh, 1) - 0.31) < 0.03 and abs(share(coh, 3) - 0.46) < 0.03   # Table 1 dialogue mix
    rel = [t["Relevance"] for p in plans for t in p["turns"]]
    assert 0.18 < share(rel, 1) < 0.30 and share(rel, 3) > 0.45


def test_stats_view_and_label_source(tmp_path, capsys):
    llm, rng = MockLLM(seed=2), random.Random(5)
    rows = []
    for i in range(20):
        plan = dg.sample_plan(rng)
        llm.current_plan = plan
        rows.append(dg.verify_one(llm, {**dg.generate_one(llm, plan), "id": i}))
    p = tmp_path / "v.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    dg.cmd_stats(argparse.Namespace(inp=str(p)))
    out = capsys.readouterr().out
    assert "20 verified conversations" in out and "confusion tables" in out and "max_mismatches" in out
    dg.cmd_view(argparse.Namespace(inp=str(p), out=str(tmp_path / "view.txt")))
    assert (tmp_path / "view.txt").read_text().count("Chatbot:") == 80
    flipped = next(r for r in rows if max(dg.label_gaps(r)) > 0)
    plan_lab = [x["label"] for x in dg.flatten(flipped, "plan")]
    ver_lab = [x["label"] for x in dg.flatten(flipped, "verifier")]
    assert plan_lab != ver_lab and len(plan_lab) == len(ver_lab) == 18


def _synthetic_scores(n=300, seed=0):
    rng = np.random.default_rng(seed)
    latent = {t: rng.normal(size=n) for t in ("flu", "coh", "eng")}
    rows = []
    for i in range(n):
        s = {d: int(np.clip(np.rint(2 + latent[DIMENSIONS[d][3]][i] + 0.35 * rng.normal()), 1, 3))
             for d in STUDIED_DIMENSIONS}
        rows.append({"id": i, "scores": s})
    return rows, latent


def test_grouping_recovers_three_tasks():
    rows, _ = _synthetic_scores()
    rho = tg.spearman_matrix(tg.score_matrix(rows))
    assert rho.shape == (12, 12) and np.allclose(np.diag(rho), 1)
    parts, score, ok, _ = tg.best_partition(rho, threshold=0.6)
    groups = [{STUDIED_DIMENSIONS[i] for i in g} for g in parts]
    expected = [{d for d in STUDIED_DIMENSIONS if DIMENSIONS[d][3] == t} for t in ("flu", "coh", "eng")]
    assert sorted(map(sorted, groups)) == sorted(map(sorted, expected))


def test_representative_weights():
    rows, latent = _synthetic_scores()
    overall = {}
    for r in rows:  # overall score driven mostly by two dimensions per group
        s = r["scores"]
        overall[str(r["id"])] = {"flu": 0.5 * s["Fluency"] + 0.4 * s["Understandability"] + 0.1 * s["Specificity"],
                                 "coh": 0.5 * s["Coherence"] + 0.4 * s["Relevance"] + 0.1 * s["Consistency"],
                                 "eng": 0.5 * s["Engagingness"] + 0.4 * s["Likeability"] + 0.1 * s["Diversity"]}
    res = tg.select_representatives(rows, overall)
    assert set(res["flu"]["representative"]) == {"Fluency", "Understandability"}
    assert set(res["coh"]["representative"]) == {"Coherence", "Relevance"}
    assert set(res["eng"]["representative"]) == {"Engagingness", "Likeability"}
    assert abs(sum(res["flu"]["weights"].values()) - 1) < 1e-6


def test_task_selector_matches_tables():
    sel = TaskSelector("task", "manual")
    for level, dim, _, _ in FED_SPEC + PERSONA_SPEC:
        assert sel(dim, level) == DIMENSIONS[dim][3], dim
    auto = TaskSelector("task", "auto")
    assert auto("Grammaticality", "response", "The response is grammatical and fluent.") == "flu"
    assert auto("Topic flow", "dialogue", "The conversation is coherent and stays on topic.") == "coh"
    assert TaskSelector("level")("Relevance", "turn") == "turn"
    assert TaskSelector("single")("Relevance", "turn") == "all"


def test_templates():
    pre, suf = build_prompt("dialogue", ["Human: hi", "Chatbot: hello"], None, "Coherence")
    assert suf == f" The entire conversation is {SLOT} coherent."
    _, suf = build_prompt("response", [], "hi", "Fluency", backbone_kind="causal")
    assert suf.endswith(SLOT) and "fluent" in suf
    _, suf = build_prompt("response", [], "hi", "Fluency", two_level=True)
    assert "dialogue turn" in suf
    _, suf = build_prompt("turn", ["a"], "b", "Relevance", prompt_style="mc")
    assert suf.endswith(SLOT) and "(C) 3" in suf


def test_benchmarks_and_metrics():
    fed = convert_fed([  # shapes copied from the real fed_data.json, including free-text "N/A" answers
        {"context": "User: Hi!\nSystem: Hi! What's up?\nUser: Can't say", "response": "System: It's probably boring?",
         "system": "Meena", "annotations": {"Fluent": [2, 1, 2], "Relevant": [1, "N/A (unclear)", 1]}},
        {"context": "User: Hi\nSystem: Yo", "system": "Meena",
         "annotations": {"Coherent": [2, 1], "Error recovery": ["N/A (no errors)", 1]}},
    ])
    assert fed[0]["context"] == ["Human: Hi!", "Chatbot: Hi! What's up?", "Human: Can't say"]
    assert fed[0]["response"] == "Chatbot: It's probably boring?" and fed[0]["annotations"]["Relevant"] == 1.0
    assert fed[1]["type"] == "dialogue" and fed[1]["annotations"]["Error recovery"] == 1.0
    usr = convert_usr([{"context": "hi there\nhello , how are you ?\nfine , you ?\n", "fact": "your persona: x",
                        "annotators": ["a", "b"], "responses": [
                            {"response": "good .\n", "model": "Seq2Seq", "Understandable": [1, 0], "Engaging": [2, 3]}]}])
    assert usr[0]["context"] == ["Human: hi there", "Chatbot: hello , how are you ?", "Human: fine , you ?"]
    assert usr[0]["response"] == "Chatbot: good ." and usr[0]["annotations"] == {"Understandable": 0.5, "Engaging": 2.5}
    jobs = build_items(fed, FED_SPEC, TaskSelector())
    assert {(l, d) for l, d, _ in jobs} == {("response", "Fluency"), ("turn", "Relevance"), ("dialogue", "Coherence")}
    resp_item = next(items for l, d, items in jobs if l == "response")[0]
    assert resp_item["response"] == "It's probably boring?" and resp_item["context"] == []
    c = correlations([1, 2, 3, 3], [1.5, 2.0, 2.9, 3.0])
    assert c["pearson"] > 0.9
    table = aggregate({"a": {"level": "turn", "pearson": 0.4, "spearman": 0.5},
                       "b": {"level": "turn", "pearson": 0.6, "spearman": 0.5},
                       "c": {"level": "dialogue", "pearson": 0.2, "spearman": 0.1}})
    assert abs(table["turn"]["pearson"] - 0.5) < 1e-9 and abs(table["avg"]["pearson"] - 0.35) < 1e-9
    m = mse_analysis({("turn", "Relevance"): (["x", "y", "z"], [1, 2, 3], [1, 2, 3])})
    assert m["mse_0_100"] == 0 and m["acc"] == 1


def test_rewrites_only_replies_that_need_it():
    llm, rng = MockLLM(), random.Random(9)
    for _ in range(30):
        plan = dg.sample_plan(rng)
        inst = dg.generate_one(llm, plan)
        for t, q in zip(inst["turns"], plan["turns"]):
            needs = (q["Fluency"], q["Understandability"]) != (3, 3)
            assert ("chatbot_original" in t) == needs
            assert not needs or t["chatbot"].startswith("broken ")


def test_unit_verification_and_compare(tmp_path, capsys):
    llm, rng = MockLLM(seed=4), random.Random(6)
    raw = []
    for i in range(10):
        plan = dg.sample_plan(rng)
        raw.append({**dg.generate_one(llm, plan), "id": i})
    unit = [dg.verify_units(llm, r) for r in raw]
    assert llm.unit_calls == 10 * 9 and all(len(u["verified"]["turns"]) == 4 for u in unit)
    conv = []
    for r in raw:
        llm.current_plan = r["plan"]
        conv.append(dg.verify_one(llm, r))
    for name, rows in (("conv.jsonl", conv), ("unit.jsonl", unit)):
        (tmp_path / name).write_text("\n".join(json.dumps(x) for x in rows))
    dg.cmd_compare(argparse.Namespace(inp=[str(tmp_path / "conv.jsonl"), str(tmp_path / "unit.jsonl")]))
    out = capsys.readouterr().out
    assert "10 conversations in common" in out and "verifier vs verifier" in out and "A~B" in out


def test_label_level_filter(tmp_path):
    llm, rng = MockLLM(seed=3, flip_rate=0.3), random.Random(8)
    rows = []
    for i in range(20):
        plan = dg.sample_plan(rng)
        llm.current_plan = plan
        rows.append(dg.verify_one(llm, {**dg.generate_one(llm, plan), "id": i}))
    for r in rows:
        kept = dg.flatten(r, keep="agree")
        assert len(kept) == sum(g == 0 for g in dg.label_gaps(r))
    p = tmp_path / "v.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows))
    args = argparse.Namespace(inp=str(p), out_dir=str(tmp_path / "out"), n_val=4, target=0, no_filter=False,
                              filter="agree", max_gap=1, max_mismatches=3, seed=0, label_source="plan")
    dg.cmd_build(args)
    built = [json.loads(l) for f in ("train", "val") for l in open(tmp_path / "out" / f"{f}.jsonl")]
    by_id = {r["id"]: r for r in rows}
    for b in built:  # every kept label is confirmed by the verifier
        v = by_id[b["dialogue_id"]]["verified"]
        vs = v["dialogue"][b["dimension"]] if b["level"] == "dialogue" else v["turns"][b["turn"]][b["dimension"]]
        assert vs == b["label"]
    assert len(built) == sum(len(dg.flatten(r, keep="agree")) for r in rows)


def test_noflip_filter_keeps_one_step_disagreements():
    llm, rng = MockLLM(seed=5, flip_rate=0.3), random.Random(12)
    for i in range(20):
        plan = dg.sample_plan(rng)
        llm.current_plan = plan
        r = dg.verify_one(llm, {**dg.generate_one(llm, plan), "id": i})
        gaps = dg.label_gaps(r)
        assert len(dg.flatten(r, keep="noflip")) == sum(g <= 1 for g in gaps)
        assert len(dg.flatten(r, keep="agree")) == sum(g == 0 for g in gaps) <= sum(g <= 1 for g in gaps)


def test_dstc10_persona_see_and_aliases(capsys):
    rows = convert_dstc10([{"dialogue_id": "d1", "model": "m", "dialogue": [
        {"speaker": "human_evaluator", "text": "hi !"}, {"speaker": "model", "text": "hello , how are you ?"}],
        "annotations": {"Fluency": [3, 4], "Listening": [2, 2], "Interest": [1, 3], "Enjoy": [4], "Inquisitive": [2]}}])
    assert "human_evaluator" in capsys.readouterr().out
    assert rows[0]["context"] == ["Human: hi !", "Chatbot: hello , how are you ?"]
    jobs = build_items(rows, PERSONA_SPEC, TaskSelector())
    assert {d for _, d, _ in jobs} == {"Fluency", "Listening", "Interestingness", "Enjoyment", "Inquisitiveness"}


def test_parlai_log_folder(tmp_path, capsys):
    d = tmp_path / "evaluation_logs_reproducible"
    d.mkdir()
    rec = {"model_persona": "your persona: i like dogs.",
           "dialog": [{"speaker": "human_evaluator", "text": "hi ! how are you ?"},
                      {"speaker": "model", "text": "i am good . i like dogs ."}],
           "evaluation_results": {"enjoy": 3, "interest": 2, "listen": 4, "fluency": 3, "inquisitive": 2,
                                  "turing": 1, "make_sense": 4, "avoid_rep": 3, "persona_guess": True}}
    (d / "repetition_model_setting05").write_text("\n".join(json.dumps(rec) for _ in range(3)))
    (d / "baseline_model.jsonl").write_text(json.dumps({**rec, "dialog": [["hello there", "hi , what is up ?"]]}))
    (d / "README.txt").write_text("not json")
    rows = convert_parlai_logs(str(d))
    out = capsys.readouterr().out
    assert len(rows) == 4 and "kept 4" in out and "human_evaluator" in out
    tagged = next(r for r in rows if r["source"] == "repetition_model_setting05")
    assert tagged["context"] == ["Human: hi ! how are you ?", "Chatbot: i am good . i like dogs ."]
    assert tagged["annotations"]["enjoy"] == 3 and tagged["annotations"]["listen"] == 4
    untagged = next(r for r in rows if r["source"] == "baseline_model.jsonl")
    assert untagged["context"] == ["Human: hello there", "Chatbot: hi , what is up ?"]
    jobs = build_items(rows, PERSONA_SPEC, TaskSelector())
    assert {d for _, d, _ in jobs} == {"Fluency", "Listening", "Interestingness", "Enjoyment", "Inquisitiveness"}


def test_prediction_summary_in_table(capsys):
    assert pred_summary([3, 3, 3, 2], "argmax") == "pred 1/2/3 = 0%/25%/75%"
    assert pred_summary([1.5, 2.5], "expected") == "pred mean 2.00 sd 0.50"
    per_dim = {"response/Fluency": {"level": "response", "pearson": 0.1, "spearman": 0.1, "p_pearson": 0.5,
                                    "n": 4, "pred": "pred 1/2/3 = 0%/0%/100%"}}
    print_table("fed", aggregate(per_dim), per_dim)
    assert "pred 1/2/3 = 0%/0%/100%" in capsys.readouterr().out


def test_topical_dev_spec():
    from unide.benchmarks import TOPICAL_SPEC
    tc = convert_usr([{"context": "do you like music ?\nyes , jazz mostly .\n", "fact": "jazz started in new orleans",
                       "responses": [{"response": "jazz began in new orleans !", "model": "Original Ground Truth",
                                      "Understandable": [1, 1, 1], "Natural": [3, 2, 3], "Maintains Context": [3, 3, 2],
                                      "Engaging": [2, 2, 3], "Uses Knowledge": [1, 1, 1], "Overall": [4, 4, 5]}]}])
    jobs = build_items(tc, TOPICAL_SPEC, TaskSelector())
    assert {d for _, d, _ in jobs} == {"Understandability", "Naturalness", "Context maintenance", "Engagingness"}


def test_summarize(tmp_path, capsys):
    from unide import summarize
    lv = lambda p, s: {"pearson": p, "spearman": s}
    res = {"levels": {"response": lv(0.2, 0.1), "turn": lv(0.4, 0.3), "dialogue": lv(0.5, 0.4), "avg": lv(0.368, 0.323)}}
    run = tmp_path / "final_main"; run.mkdir()
    for m in ("argmax", "expected"):
        for b in ("fed", "personachat"):
            (run / f"{b}_{m}.json").write_text(json.dumps(res))
    (run / "train_args.json").write_text(json.dumps({"backbone": "facebook/opt-1.3b", "best_dev": lv(0.273, 0.278),
                                                     "selected_at": "epoch 1 step 850"}))
    van = tmp_path / "vanilla_opt-1.3b"; van.mkdir()
    (van / "fed_expected.json").write_text(json.dumps(res)); (van / "backbone.txt").write_text("facebook/opt-1.3b")
    (tmp_path / "empty").mkdir()
    rows = summarize.collect([str(run), str(van), str(tmp_path / "empty")])
    assert [r["run"] for r in rows] == ["final_main", "vanilla_opt-1.3b"]
    assert rows[0]["fed avg (expected)"] == "36.8/32.3" and rows[0]["dev r/rho"] == "27.3/27.8"
    assert rows[1]["backbone"].endswith("(untrained)") and rows[1]["fed avg (argmax)"] == "-"
    assert rows[0]["paper fed avg"] == "52.1/53.4" and rows[1]["paper fed avg"].startswith("35.0")
    assert [summarize.paper_key(r) for r in ("final_main_s43", "test_final_wo_filter", "final_two_level",
                                              "final_main_rlhf", "final_wo_filter_rlhf_s43")] == \
        ["main", "wo_filter", "two_level", "main", "wo_filter"]
    assert summarize.PAPER[summarize.paper_key("vanilla_chat-opt-1.3b-rlhf-actor-ema-deepspeed")][0].startswith("35.0")
