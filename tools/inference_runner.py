"""Local scorer stub for bertsum integration.

Replace this module with true Espresso-AI/bertsum-korean inference code in an
environment where torch/transformers and model checkpoints are installed.
"""

from typing import List


def score_sentences(sentences: List[str]) -> List[float]:
    # Temporary deterministic baseline: longer sentences get slightly higher weight.
    return [float(len(s)) for s in sentences]
