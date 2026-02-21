import argparse
import json
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[가-힣A-Za-z0-9]+")


@dataclass
class SentenceScore:
    idx: int
    sentence: str
    score: float


def split_sentences(text: str) -> List[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []
    parts = _SENT_SPLIT_RE.split(compact)
    return [p.strip() for p in parts if p.strip()]


def _tokens(text: str) -> List[str]:
    return [tok.lower() for tok in _WORD_RE.findall(text)]


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(_tokens(a)), set(_tokens(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _score_sentences_local(sentences: Sequence[str]) -> List[float]:
    doc_tokens = _tokens(" ".join(sentences))
    tf = {}
    for token in doc_tokens:
        tf[token] = tf.get(token, 0) + 1

    scores: List[float] = []
    for sent in sentences:
        toks = _tokens(sent)
        if not toks:
            scores.append(0.0)
            continue
        content_score = sum(tf.get(t, 0) for t in toks) / len(toks)
        length_penalty = 1.0 if 8 <= len(toks) <= 35 else 0.85
        scores.append(content_score * length_penalty)
    return scores


def _run_bertsum_runner(sentences: Sequence[str], runner_path: str) -> List[float]:
    with tempfile.TemporaryDirectory(prefix="bertsum_runner_") as tmpdir:
        in_path = os.path.join(tmpdir, "input.json")
        out_path = os.path.join(tmpdir, "output.json")

        with open(in_path, "w", encoding="utf-8") as f:
            json.dump({"sentences": list(sentences)}, f, ensure_ascii=False)

        subprocess.run([
            "python",
            runner_path,
            "--input-json",
            in_path,
            "--output-json",
            out_path,
        ], check=True)

        with open(out_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

    scores = payload.get("scores")
    if not isinstance(scores, list) or len(scores) != len(sentences):
        raise RuntimeError("bertsum runner returned invalid scores payload")
    return [float(v) for v in scores]


def _score_sentences_bertsum_korean(
    sentences: Sequence[str],
    runner_path: str | None,
) -> List[float]:
    """Run sentence scoring through a bertsum-korean inference runner.

    Runner contract:
    - executable Python script
    - args: `--input-json <path> --output-json <path>`
    - input JSON: {"sentences": ["...", ...]}
    - output JSON: {"scores": [float, ...]}  # same length as input sentences

    You can point this to a wrapper script inside a local bertsum-korean checkout
    where actual model weights are loaded and inference is performed.
    """
    resolved_runner = runner_path or os.getenv("BERTSUM_KOREAN_RUNNER")
    if not resolved_runner:
        raise RuntimeError(
            "bertsum-korean backend requires --bertsum-runner or BERTSUM_KOREAN_RUNNER."
        )
    if not os.path.exists(resolved_runner):
        raise RuntimeError(f"bertsum runner not found: {resolved_runner}")

    return _run_bertsum_runner(sentences, resolved_runner)


def score_sentences(
    sentences: Sequence[str],
    backend: str,
    bertsum_runner: str | None = None,
) -> List[float]:
    if backend == "local":
        return _score_sentences_local(sentences)
    if backend == "bertsum-korean":
        return _score_sentences_bertsum_korean(sentences, runner_path=bertsum_runner)
    raise ValueError(f"Unsupported backend: {backend}")


def summarize_extractive(
    context: str,
    max_chars: int = 150,
    min_chars: int = 90,
    similarity_threshold: float = 0.72,
    backend: str = "local",
    bertsum_runner: str | None = None,
) -> str:
    sentences = split_sentences(context)
    if not sentences:
        return ""

    scores = score_sentences(sentences, backend=backend, bertsum_runner=bertsum_runner)
    ranked = sorted(
        (SentenceScore(i, s, sc) for i, (s, sc) in enumerate(zip(sentences, scores))),
        key=lambda x: x.score,
        reverse=True,
    )

    selected: List[Tuple[int, str]] = []
    current = ""

    for item in ranked:
        if any(_jaccard(item.sentence, picked) >= similarity_threshold for _, picked in selected):
            continue
        candidate = f"{current} {item.sentence}".strip() if current else item.sentence
        if len(candidate) <= max_chars:
            selected.append((item.idx, item.sentence))
            current = candidate
        if len(current) >= max_chars - 10:
            break

    selected.sort(key=lambda x: x[0])
    summary = " ".join(s for _, s in selected).strip()

    if len(summary) < min_chars:
        used = {i for i, _ in selected}
        for item in ranked:
            if item.idx in used:
                continue
            candidate = f"{summary} {item.sentence}".strip() if summary else item.sentence
            if len(candidate) <= max_chars:
                summary = candidate
                break

    if len(summary) > max_chars:
        segments = split_sentences(summary)
        while segments and len(" ".join(segments)) > max_chars:
            segments.pop()
        summary = " ".join(segments).strip()

    return summary


def _read_input(text: str | None, path: str | None) -> str:
    if text:
        return text
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    raise ValueError("Provide --text or --input-file")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extractive summarizer (<=150 chars)")
    parser.add_argument("--text", type=str, help="Raw context text")
    parser.add_argument("--input-file", type=str, help="Path to UTF-8 text file")
    parser.add_argument("--max-chars", type=int, default=150)
    parser.add_argument(
        "--backend",
        choices=["local", "bertsum-korean"],
        default="local",
        help="Sentence scoring backend",
    )
    parser.add_argument(
        "--bertsum-runner",
        type=str,
        default=None,
        help="Path to bertsum-korean inference runner script",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    context = _read_input(args.text, args.input_file)
    summary = summarize_extractive(
        context=context,
        max_chars=args.max_chars,
        backend=args.backend,
        bertsum_runner=args.bertsum_runner,
    )
    print(summary)
    print(f"[chars={len(summary)}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
