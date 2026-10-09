"""Level-specific prompt templates (Sec. 3.4, Fig. 5a) and the verbalizer Z -> Y.

Every template is split into
  * prefix: the dialogue instance [d] (may be truncated from the left if too long)
  * suffix: the template text containing the answer slot SLOT
so that the collator can truncate the instance while always keeping the answer slot.

Backbone kinds:
  * "mlm"     (RoBERTa / BERT)  - SLOT becomes the tokenizer's mask token, exactly as in the paper.
  * "seq2seq" (BART)            - same as mlm; the decoder state at the mask position is used.
  * "causal"  (OPT / GPT / InstructGPT-style decoders) - a decoder cannot look to the right of the slot, so
    the dimension is moved in front of the slot and the slot is the last position (next-token prediction):
    "... Is the entire conversation coherent? The entire conversation is [SLOT]".
"""
from .dimensions import adjective, DIMENSIONS

SLOT = "<<ANSWER_SLOT>>"

# Z = {not, moderate, very} -> Y = {1, 2, 3}
VERBALIZERS = {
    "mlm": ["not", "moderate", "very"],   # MLM-prompt (main method)
    "score": ["1", "2", "3"],             # Score-prompt (w/o template mapping ablation)
    "mc": ["A", "B", "C"],                # MC-prompt (Discussion 3)
}

_OBJECT = {"response": "the response", "turn": "the dialogue turn", "dialogue": "the entire conversation"}
_OBJECT_MC = {"response": "response", "turn": "dialogue turn", "dialogue": "conversation"}


def format_utterances(utts: list[str]) -> str:
    """Join utterances. Utterances that already carry a speaker tag ('Human: ...') are kept as-is."""
    out = []
    for i, u in enumerate(utts):
        u = u.strip()
        if ":" in u[:20]:
            out.append(u)
        else:
            out.append(f"{'A' if i % 2 == 0 else 'B'}: {u}")
    return " ".join(out)


def instance_text(level: str, context: list[str] | None, response: str | None) -> str:
    """[d^res], [d^turn], [d^dial] instance slots."""
    if level == "response":
        return response.strip()
    if level == "turn":
        return f"Context: {format_utterances(context or [])} Response: {response.strip()}"
    if level == "dialogue":
        utts = list(context or []) + ([response] if response else [])
        return format_utterances(utts)
    raise ValueError(level)


def build_prompt(level: str, context, response, dimension: str, prompt_style: str = "mlm",
                 backbone_kind: str = "mlm", two_level: bool = False) -> tuple[str, str]:
    """Return (prefix, suffix). `two_level=True` reproduces the RQ3 ablation where response-level
    dimensions are wrapped with the turn-level template."""
    prefix = instance_text(level, context, response)
    tmpl_level = "turn" if (two_level and level == "response") else level
    obj = _OBJECT[tmpl_level]
    adj = adjective(dimension)

    if prompt_style == "mlm":
        if backbone_kind == "causal":
            suffix = f" Is {obj} {adj}? {obj[0].upper() + obj[1:]} is {SLOT}"
        else:
            suffix = f" {obj[0].upper() + obj[1:]} is {SLOT} {adj}."
    elif prompt_style == "score":
        # "[X] the entire conversation is [Coherence]. [Score]"
        suffix = f" {obj[0].upper() + obj[1:]} is {adj}. Score: {SLOT}"
    elif prompt_style == "mc":
        name = dimension.lower() if dimension in DIMENSIONS else dimension
        suffix = (f" Which is the most appropriate score for the {_OBJECT_MC[tmpl_level]} if evaluated in terms of "
                  f"{name}? (A) 1; (B) 2; (C) 3. Answer: {SLOT}")
    else:
        raise ValueError(prompt_style)
    return prefix, suffix


def training_target_text(label: int, prompt_style: str = "mlm") -> str:
    """Natural-language rendering of a numeric label (Fig. 2b), used only for inspection / export."""
    return VERBALIZERS[prompt_style][label - 1]
