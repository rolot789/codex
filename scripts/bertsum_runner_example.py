"""bertsum-korean runner.

This script satisfies the runner contract expected by `src.extractive_summarizer`:
- input args: --input-json / --output-json
- input JSON: {"sentences": ["...", ...]}
- output JSON: {"scores": [float, ...]}

Two modes are supported:
1) real mode (default): uses a local bertsum-korean checkout if available.
2) fallback mode: uses sentence-length scoring for environments without model deps.

Environment variables for real mode:
- BERTSUM_KOREAN_REPO: path to local Espresso-AI/bertsum-korean checkout
- BERTSUM_KOREAN_SCORER: import path exposing `score_sentences(sentences)->List[float]`
  (default: tools.inference_runner)
- BERTSUM_USE_FALLBACK=1 to force fallback heuristic
"""

import argparse
import importlib
import json
import os
import sys
from typing import List


DEFAULT_SCORER_MODULE = "tools.bertsum_real_inference"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def score_with_fallback(sentences: List[str]) -> List[float]:
    return [float(len(s)) for s in sentences]


def score_with_real_bertsum(sentences: List[str]) -> List[float]:
    if os.getenv("BERTSUM_USE_FALLBACK") == "1":
        return score_with_fallback(sentences)

    repo = os.getenv("BERTSUM_KOREAN_REPO")
    if not repo:
        return score_with_fallback(sentences)

    if not os.path.isdir(repo):
        return score_with_fallback(sentences)

    if repo not in sys.path:
        sys.path.insert(0, repo)

    module_path = os.getenv("BERTSUM_KOREAN_SCORER", DEFAULT_SCORER_MODULE)
    module = importlib.import_module(module_path)

    if not hasattr(module, "score_sentences"):
        raise RuntimeError(
            f"Module '{module_path}' must expose score_sentences(sentences)->List[float]"
        )

    scores = module.score_sentences(sentences)
    if not isinstance(scores, list) or len(scores) != len(sentences):
        raise RuntimeError("score_sentences returned invalid result")

    return [float(v) for v in scores]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    sentences = payload.get("sentences", [])
    scores = score_with_real_bertsum(sentences)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"scores": scores}, f, ensure_ascii=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
