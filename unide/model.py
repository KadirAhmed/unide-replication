"""UniDE evaluator: hard-parameter-sharing backbone T + one linear head per task module (Sec. 3.3, Fig. 5c).

Each head maps the hidden state at the answer slot to a distribution over Z = {not, moderate, very}
(Eq. 12). Heads are initialised from the backbone's word embeddings of the label words, so at step 0 each head
equals the pretrained LM's masked-token prediction restricted to the verbaliser (the MLM objective on polarity
words of Sec. 3.2); training then specialises each head on its own task data.
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .templates import SLOT, VERBALIZERS, build_prompt


def detect_kind(name_or_path: str) -> str:
    cfg = AutoConfig.from_pretrained(name_or_path)
    if getattr(cfg, "is_encoder_decoder", False):
        return "seq2seq"
    archs = " ".join(cfg.architectures or []).lower()
    if cfg.model_type in {"bert", "roberta", "deberta", "deberta-v2", "electra", "xlm-roberta", "albert"} \
            or "maskedlm" in archs:
        return "mlm"
    return "causal"


class UniDE(nn.Module):
    def __init__(self, backbone: str, tasks: list[str], prompt_style: str = "mlm", kind: str | None = None,
                 head_init: str = "verbalizer", gradient_checkpointing: bool = False, tokenizer_path: str | None = None):
        super().__init__()
        self.backbone_name, self.tasks, self.prompt_style = backbone, list(tasks), prompt_style
        self.kind = kind or detect_kind(backbone)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path or backbone)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.encoder = AutoModel.from_pretrained(backbone)
        if gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()
        hidden = self.encoder.config.hidden_size
        self.label_words = VERBALIZERS[prompt_style]
        self.label_ids = self._label_token_ids()
        self.heads = nn.ModuleDict({t: nn.Linear(hidden, len(self.label_words)) for t in self.tasks})
        if head_init == "verbalizer":
            self._init_heads_from_embeddings()

    # ------------------------------------------------------------------ verbaliser
    def _label_token_ids(self) -> list[int]:
        ids = []
        for w in self.label_words:
            toks = self.tokenizer.encode(" " + w, add_special_tokens=False)
            ids.append(toks[0])  # first sub-token of " not" / " moderate" / " very"
        if len(set(ids)) != len(ids):
            raise ValueError(f"label words {self.label_words} collide after tokenisation: {ids}")
        return ids

    @torch.no_grad()
    def _init_heads_from_embeddings(self):
        emb = self.encoder.get_input_embeddings().weight
        if emb.shape[1] != self.encoder.config.hidden_size:
            return  # e.g. projected embeddings; keep random init
        w = emb[self.label_ids].detach().clone().float()
        for head in self.heads.values():
            head.weight.copy_(w)
            head.bias.zero_()

    # ------------------------------------------------------------------ encoding
    def encode(self, prefix: str, suffix: str, max_len: int = 512) -> tuple[list[int], int]:
        """Tokenise prefix+suffix, truncating the instance (prefix) from the left, and return (ids, slot_pos).

        mlm / seq2seq : SLOT -> mask token, slot_pos = position of the mask token
        causal        : suffix ends with SLOT, which is removed; slot_pos = last position (next-token prediction)
        """
        tok = self.tokenizer
        before, after = suffix.split(SLOT)
        pre_ids = tok.encode(prefix, add_special_tokens=False)
        bef_ids = tok.encode(before.rstrip(), add_special_tokens=False)  # label words carry their own space
        aft_ids = tok.encode(after, add_special_tokens=False) if after.strip() else []
        if self.kind == "causal":
            head = [tok.bos_token_id] if tok.bos_token_id is not None else []
            tail = []
            mid = []
        else:
            head = [tok.cls_token_id if tok.cls_token_id is not None else tok.bos_token_id]
            tail = [tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id]
            mid = [tok.mask_token_id]
        budget = max_len - len(head) - len(bef_ids) - len(mid) - len(aft_ids) - len(tail)
        pre_ids = pre_ids[-budget:] if budget > 0 else []
        ids = head + pre_ids + bef_ids + mid + aft_ids + tail
        slot = len(head) + len(pre_ids) + len(bef_ids) - (1 if self.kind == "causal" else 0)
        return ids, slot

    def collate(self, items: list[dict], max_len: int = 512, two_level: bool = False) -> dict:
        """items: dicts with level, context, response, dimension, task and (optionally) label."""
        enc = [self.encode(*build_prompt(it["level"], it["context"], it["response"], it["dimension"],
                                         self.prompt_style, self.kind, two_level), max_len) for it in items]
        L = max(len(ids) for ids, _ in enc)
        pad = self.tokenizer.pad_token_id
        input_ids = torch.full((len(enc), L), pad, dtype=torch.long)
        attn = torch.zeros((len(enc), L), dtype=torch.long)
        for i, (ids, _) in enumerate(enc):
            input_ids[i, :len(ids)] = torch.tensor(ids)
            attn[i, :len(ids)] = 1
        batch = {"input_ids": input_ids, "attention_mask": attn,
                 "positions": torch.tensor([p for _, p in enc]),
                 "task_ids": torch.tensor([self.tasks.index(it["task"]) for it in items])}
        if all("label" in it for it in items):
            batch["labels"] = torch.tensor([int(it["label"]) - 1 for it in items])  # Y={1,2,3} -> {0,1,2}
        return batch

    # ------------------------------------------------------------------ forward
    def forward(self, input_ids, attention_mask, positions, task_ids, labels=None, task_weights=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # decoder states for seq2seq (decoder input = shifted input ids)
        h_slot = h[torch.arange(h.size(0), device=h.device), positions]  # E_[MASK]
        all_logits = torch.stack([self.heads[t](h_slot.float()) for t in self.tasks], dim=1)  # B x T x 3
        logits = all_logits[torch.arange(h.size(0), device=h.device), task_ids]
        result = {"logits": logits}
        if labels is not None:
            # Eq. (6): Loss = sum_t w_t * L_t, each L_t the cross entropy on that task's examples in the batch
            loss = logits.new_zeros(())
            for t_idx, t in enumerate(self.tasks):
                m = task_ids == t_idx
                if m.any():
                    w = 1.0 / len(self.tasks) if task_weights is None else task_weights[t]
                    loss = loss + w * F.cross_entropy(logits[m], labels[m])
            result["loss"] = loss
        return result

    @torch.no_grad()
    def score(self, batch, mode: str = "argmax"):
        """Eqs. (13)-(14): argmax over Z mapped to Y={1,2,3}; mode='expected' returns E[y] instead."""
        probs = self.forward(**{k: v for k, v in batch.items() if k != "labels"})["logits"].softmax(-1)
        if mode == "expected":
            return (probs * torch.arange(1, probs.size(-1) + 1, device=probs.device)).sum(-1), probs
        return probs.argmax(-1) + 1, probs

    # ------------------------------------------------------------------ io
    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.encoder.save_pretrained(os.path.join(path, "backbone"))
        self.tokenizer.save_pretrained(os.path.join(path, "backbone"))
        torch.save(self.heads.state_dict(), os.path.join(path, "heads.pt"))
        with open(os.path.join(path, "unide_config.json"), "w") as f:
            json.dump({"backbone": self.backbone_name, "tasks": self.tasks, "prompt_style": self.prompt_style,
                       "kind": self.kind}, f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(os.path.join(path, "unide_config.json")) as f:
            cfg = json.load(f)
        model = cls(os.path.join(path, "backbone"), cfg["tasks"], cfg["prompt_style"], cfg["kind"], head_init="none")
        model.backbone_name = cfg["backbone"]
        model.heads.load_state_dict(torch.load(os.path.join(path, "heads.pt"), map_location="cpu"))
        return model
