"""Real bertsum-korean inference adapter."""

from __future__ import annotations

import importlib.util
import os
import sys
import types
import urllib.request
from functools import lru_cache
from typing import List, Sequence, Tuple

import torch
from transformers import AutoTokenizer

DEFAULT_REPO = "/workspace/bertsum-korean"
DEFAULT_CHECKPOINT_URL = (
    "https://github.com/Espresso-AI/bertsum-korean/releases/download/checkpoints/"
    "epoch.1-step.17141.ckpt"
)
DEFAULT_CHECKPOINT_PATH = "/workspace/codex/models/epoch.1-step.17141.ckpt"
DEFAULT_BASE_CHECKPOINT = "klue/bert-base"


def _ensure_checkpoint(path: str) -> str:
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    urllib.request.urlretrieve(DEFAULT_CHECKPOINT_URL, path)
    return path


def _load_module(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec: {module_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=1)
def _load_bertsum_class():
    repo = os.getenv("BERTSUM_KOREAN_REPO", DEFAULT_REPO)
    model_dir = os.path.join(repo, "src", "model")

    pkg_name = "_bertsum_local"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [model_dir]  # type: ignore[attr-defined]
    sys.modules[pkg_name] = pkg

    _load_module(f"{pkg_name}.encoder", os.path.join(model_dir, "encoder.py"))
    bertsum_mod = _load_module(f"{pkg_name}.bertsum", os.path.join(model_dir, "bertsum.py"))
    return bertsum_mod.BertSum_Ext


@lru_cache(maxsize=1)
def _load_artifacts() -> Tuple[object, object]:
    checkpoint_path = os.getenv("BERTSUM_KOREAN_CHECKPOINT", DEFAULT_CHECKPOINT_PATH)
    base_checkpoint = os.getenv("BERTSUM_KOREAN_BASE_CHECKPOINT", DEFAULT_BASE_CHECKPOINT)

    BertSum_Ext = _load_bertsum_class()
    checkpoint_path = _ensure_checkpoint(checkpoint_path)

    model = BertSum_Ext(base_checkpoint=base_checkpoint)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
    model.load_state_dict(state, strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)
    return model, tokenizer


def _build_inputs(sentences: Sequence[str], tokenizer: object, max_seq_len: int):
    input_ids: List[int] = []
    token_type_ids: List[int] = []
    attention_mask: List[int] = []
    cls_token_ids: List[int] = []

    kept = 0
    seq_id = 0
    for sent in sentences:
        enc = tokenizer(sent, add_special_tokens=True)
        sent_ids = enc["input_ids"]
        if len(input_ids) + len(sent_ids) > max_seq_len:
            remain = max_seq_len - len(input_ids)
            if remain <= 1:
                break
            sent_ids = sent_ids[: remain - 1] + [sent_ids[-1]]

        cls_token_ids.append(len(input_ids))
        input_ids.extend(sent_ids)
        token_type_ids.extend([seq_id] * len(sent_ids))
        attention_mask.extend([1] * len(sent_ids))
        kept += 1
        seq_id = 1 - seq_id
        if len(input_ids) >= max_seq_len:
            break

    if len(input_ids) < max_seq_len:
        pad_len = max_seq_len - len(input_ids)
        input_ids.extend([tokenizer.pad_token_id] * pad_len)
        token_type_ids.extend([0] * pad_len)
        attention_mask.extend([0] * pad_len)

    if len(cls_token_ids) < max_seq_len:
        cls_token_ids.extend([-1] * (max_seq_len - len(cls_token_ids)))

    encodings = {
        "input_ids": torch.tensor([input_ids], dtype=torch.long),
        "token_type_ids": torch.tensor([token_type_ids], dtype=torch.long),
        "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
    }
    cls = torch.tensor([cls_token_ids], dtype=torch.long)
    return encodings, cls, kept


def score_sentences(sentences: Sequence[str]) -> List[float]:
    if not sentences:
        return []
    model, tokenizer = _load_artifacts()
    max_seq_len = int(os.getenv("BERTSUM_MAX_SEQ_LEN", "512"))

    encodings, cls_token_ids, kept = _build_inputs(sentences, tokenizer, max_seq_len)
    if kept == 0:
        return [0.0] * len(sentences)

    with torch.no_grad():
        outputs = model(encodings=encodings, cls_token_ids=cls_token_ids)

    probs = torch.sigmoid(outputs["logits"][0][:kept]).tolist()
    if kept < len(sentences):
        probs += [0.0] * (len(sentences) - kept)
    return [float(v) for v in probs]
