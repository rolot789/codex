#!/usr/bin/env bash
set -euo pipefail

BERTSUM_REPO_DIR="${BERTSUM_REPO_DIR:-/workspace/bertsum-korean}"

if [ ! -d "$BERTSUM_REPO_DIR/.git" ]; then
  git clone https://github.com/Espresso-AI/bertsum-korean.git "$BERTSUM_REPO_DIR"
else
  echo "[bootstrap] using existing repo: $BERTSUM_REPO_DIR"
fi

python -m pip install torch transformers==4.30.2 sentencepiece omegaconf pandas

echo "[bootstrap] complete"
echo "export BERTSUM_KOREAN_REPO=$BERTSUM_REPO_DIR"
echo "export BERTSUM_KOREAN_SCORER=tools.bertsum_real_inference"
