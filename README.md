# codex

텍스트 입력을 받아 150자 이내 추출 요약을 반환하는 CLI입니다.

## 실행

```bash
python -m src.extractive_summarizer --text "긴 문맥 텍스트를 여기에 입력"
```

또는 파일 입력:

```bash
python -m src.extractive_summarizer --input-file sample.txt
```

## 백엔드

- `--backend local` (기본): 로컬 점수화 기반
- `--backend bertsum-korean`: 외부 bertsum-korean 러너 스크립트를 통해 모델 점수 사용

```bash
python -m src.extractive_summarizer \
  --input-file sample.txt \
  --backend bertsum-korean \
  --bertsum-runner scripts/bertsum_runner_example.py
```

`bertsum_runner.py` 계약:
- 입력 인자: `--input-json <path> --output-json <path>`
- 입력 JSON: `{"sentences": ["...", ...]}`
- 출력 JSON: `{"scores": [float, ...]}` (문장 개수와 동일)

## 실제 모델 추론 연결(다음 단계)

`bertsum_runner_example.py`는 기본적으로 실제 bertsum 모듈 import를 시도하고, 설정이 없으면 길이 기반 fallback으로 동작합니다.

필수 환경 변수:
- `BERTSUM_KOREAN_REPO`: 로컬 `Espresso-AI/bertsum-korean` 체크아웃 경로
- `BERTSUM_KOREAN_SCORER` (옵션): `score_sentences(sentences)`를 제공하는 모듈 경로
  - 기본값: `tools.inference_runner`

예시:

```bash
export BERTSUM_KOREAN_REPO=/path/to/bertsum-korean
export BERTSUM_KOREAN_SCORER=tools.inference_runner
python -m src.extractive_summarizer \
  --input-file sample.txt \
  --backend bertsum-korean \
  --bertsum-runner scripts/bertsum_runner_example.py
```

임시/개발 환경에서 휴리스틱 강제:

```bash
export BERTSUM_USE_FALLBACK=1
```

출력은 요약문과 글자 수(`[chars=N]`)를 함께 표시합니다.

## 문장 분리기 선택

`split_sentences`는 전략 기반으로 추상화되어 있으며, 요약 시 문장 분리기를 선택할 수 있습니다.

- `default`: 정규식 기반 기본 분리기
- `korean-advanced`: 한국어 특화 라이브러리(`kss`) 기반 분리기

```bash
python -m src.extractive_summarizer \
  --input-file sample.txt \
  --sentence-splitter korean-advanced
```

`korean-advanced` 사용 시:

```bash
pip install kss
```

## 문장 분리기 품질 확인 절차

문장 분리 결과가 달라지는 샘플에서 요약 품질을 비교하려면 아래 순서를 사용합니다.

1) 비교용 입력 샘플을 준비합니다 (인용부호/괄호/줄바꿈/특수문자 포함 권장).
2) 기본 분리기와 고급 분리기로 각각 요약을 생성합니다.
3) 결과 문장 경계와 핵심 정보 보존 여부를 비교합니다.

예시:

```bash
cat > /tmp/splitter_sample.txt <<'EOF'
그는 말했다. "정말인가?" (확실해!)
다음 줄은 특수문자 테스트 #요약 @품질 입니다.
EOF

python -m src.extractive_summarizer \
  --input-file /tmp/splitter_sample.txt \
  --sentence-splitter default

python -m src.extractive_summarizer \
  --input-file /tmp/splitter_sample.txt \
  --sentence-splitter korean-advanced
```

검토 포인트:
- 인용부호/괄호 내부 문장이 과분리 혹은 미분리되지 않는지
- 줄바꿈이 문맥 손실 없이 자연스럽게 연결되는지
- 특수문자가 포함된 문장에서도 핵심 문장이 선택되는지


실제 체크포인트 다운로드 예시:

```bash
curl -L -o vendor/bertsum-korean/epoch.1-step.17141.ckpt \
  https://github.com/Espresso-AI/bertsum-korean/releases/download/checkpoints/epoch.1-step.17141.ckpt
```

추가 환경 변수:
- `BERTSUM_KOREAN_CHECKPOINT` (옵션): `.ckpt` 파일 경로
- `BERTSUM_BASE_CHECKPOINT` (옵션): 기본값 `klue/bert-base`
