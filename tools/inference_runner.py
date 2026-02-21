"""Real bertsum-korean scorer using Espresso-AI checkpoint."""

from __future__ import annotations

import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, BatchEncoding


def _resolve_repo() -> str:
    repo = os.getenv("BERTSUM_KOREAN_REPO", "vendor/bertsum-korean")
    if not os.path.isdir(repo):
        raise RuntimeError(f"bertsum repo not found: {repo}")
    return repo


def _resolve_checkpoint(repo: str) -> str:
    checkpoint = os.getenv("BERTSUM_KOREAN_CHECKPOINT", os.path.join(repo, "epoch.1-step.17141.ckpt"))
    if not os.path.isfile(checkpoint):
        raise RuntimeError(f"bertsum checkpoint not found: {checkpoint}")
    return checkpoint


def _load_sum_encoder(repo: str):
    encoder_path = Path(repo) / "src" / "model" / "encoder.py"
    spec = importlib.util.spec_from_file_location("bertsum_encoder", encoder_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load encoder module from {encoder_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SumEncoder


class _BertSumExt(torch.nn.Module):
    def __init__(self, repo: str) -> None:
        super().__init__()
        SumEncoder = _load_sum_encoder(repo)

        base_ckpt = os.getenv("BERTSUM_BASE_CHECKPOINT", "klue/bert-base")
        cfg = AutoConfig.from_pretrained(base_ckpt)
        self.base_model = AutoModel.from_config(cfg)
        self.head = SumEncoder(
            num_layers=2,
            hidden_size=cfg.hidden_size,
            intermediate_size=2048,
            num_attention_heads=8,
            dropout_prob=0.1,
        )

    def forward(self, encodings: BatchEncoding, cls_token_ids: torch.Tensor) -> torch.Tensor:
        token_embeds = self.base_model(**encodings).last_hidden_state
        out = self.head(token_embeds, cls_token_ids)
        return torch.sigmoid(out["logits"]) * out["cls_token_mask"]


@lru_cache(maxsize=1)
def _load_model_and_tokenizer() -> Tuple[_BertSumExt, AutoTokenizer]:
    repo = _resolve_repo()
    checkpoint_path = _resolve_checkpoint(repo)
    model = _BertSumExt(repo)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model_state = {k.replace("model.", "", 1): v for k, v in state_dict.items() if k.startswith("model.")}
    model.load_state_dict(model_state, strict=False)
    model.eval()

    base_ckpt = os.getenv("BERTSUM_BASE_CHECKPOINT", "klue/bert-base")
    tokenizer = AutoTokenizer.from_pretrained(base_ckpt)
    return model, tokenizer


def _build_document_inputs(sentences: List[str], tokenizer: AutoTokenizer, max_seq_len: int = 512) -> tuple[BatchEncoding, torch.Tensor]:
    input_ids: List[int] = []
    token_type_ids: List[int] = []
    attention_mask: List[int] = []
    cls_token_ids: List[int] = []
    seq_id = 0

    for sent in sentences:
        ids = tokenizer(sent, add_special_tokens=True)["input_ids"]
        if len(input_ids) + len(ids) > max_seq_len:
            remaining = max_seq_len - len(input_ids)
            if remaining <= 1:
                break
            ids = ids[: remaining - 1] + [ids[-1]]

        cls_token_ids.append(len(input_ids))
        input_ids.extend(ids)
        token_type_ids.extend([seq_id] * len(ids))
        attention_mask.extend([1] * len(ids))
        seq_id = 1 - seq_id
        if len(input_ids) >= max_seq_len:
            break

    if not cls_token_ids:
        return BatchEncoding(), torch.zeros((1, max_seq_len), dtype=torch.long)

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    pad_len = max_seq_len - len(input_ids)
    if pad_len > 0:
        input_ids.extend([pad_id] * pad_len)
        token_type_ids.extend([0] * pad_len)
        attention_mask.extend([0] * pad_len)

    cls_ids_padded = cls_token_ids + [-1] * (max_seq_len - len(cls_token_ids))
    encodings = BatchEncoding(
        {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "token_type_ids": torch.tensor([token_type_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
        }
    )
    return encodings, torch.tensor([cls_ids_padded], dtype=torch.long)


def score_sentences(sentences: List[str]) -> List[float]:
    if not sentences:
        return []
    model, tokenizer = _load_model_and_tokenizer()
    encodings, cls_token_ids = _build_document_inputs(sentences, tokenizer)

    with torch.no_grad():
        scores = model(encodings, cls_token_ids)[0]

    valid = int((cls_token_ids[0] != -1).sum().item())
    out = [float(v) for v in scores[:valid].tolist()]
    if valid < len(sentences):
        out.extend([0.0] * (len(sentences) - valid))
    return out
