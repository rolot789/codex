#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: $0 <input-file> [max-chars]" >&2
  exit 1
fi

INPUT_FILE="$1"
MAX_CHARS="${2:-150}"

: "${BERTSUM_KOREAN_REPO:=/workspace/bertsum-korean}"
: "${BERTSUM_KOREAN_SCORER:=tools.bertsum_real_inference}"

python -m src.extractive_summarizer \
  --input-file "$INPUT_FILE" \
  --max-chars "$MAX_CHARS" \
  --backend bertsum-korean \
  --bertsum-runner scripts/bertsum_runner_example.py
