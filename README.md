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


실제 체크포인트 다운로드 예시:

```bash
curl -L -o vendor/bertsum-korean/epoch.1-step.17141.ckpt \
  https://github.com/Espresso-AI/bertsum-korean/releases/download/checkpoints/epoch.1-step.17141.ckpt
```

추가 환경 변수:
- `BERTSUM_KOREAN_CHECKPOINT` (옵션): `.ckpt` 파일 경로
- `BERTSUM_BASE_CHECKPOINT` (옵션): 기본값 `klue/bert-base`
