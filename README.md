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

## 실제 모델 추론 연결

기본 스코어러는 `tools.bertsum_real_inference`로 설정되어 있으며,
아래 2가지를 준비하면 **실제 Espresso-AI/bertsum-korean 체크포인트**를 사용합니다.

1) bertsum-korean 소스 체크아웃
```bash
git clone https://github.com/Espresso-AI/bertsum-korean.git /workspace/bertsum-korean
```

2) 필요 패키지 설치
```bash
python -m pip install torch transformers==4.30.2 sentencepiece omegaconf pandas
```

실행 예시:

```bash
BERTSUM_KOREAN_REPO=/workspace/bertsum-korean \
BERTSUM_KOREAN_SCORER=tools.bertsum_real_inference \
python -m src.extractive_summarizer \
  --input-file sample.txt \
  --backend bertsum-korean \
  --bertsum-runner scripts/bertsum_runner_example.py
```

환경 변수:
- `BERTSUM_KOREAN_REPO`: 로컬 bertsum-korean 체크아웃 경로
- `BERTSUM_KOREAN_SCORER`: 스코어러 모듈 경로 (기본: `tools.bertsum_real_inference`)
- `BERTSUM_KOREAN_CHECKPOINT`: 체크포인트 경로 (기본: `models/epoch.1-step.17141.ckpt`)
- `BERTSUM_USE_FALLBACK=1`: 모델 대신 휴리스틱 점수 강제

출력은 요약문과 글자 수(`[chars=N]`)를 함께 표시합니다.


## 한 번에 세팅 + 실행 (충돌 방지용)

여러 커밋/여러 셸에서 따로 작업하면 환경 변수·의존성 차이로 충돌이 나기 쉬워서,
아래처럼 스크립트 2개로 **한 번에 동일 절차**를 실행할 수 있습니다.

1) 부트스트랩
```bash
./scripts/bootstrap_bertsum_korean.sh
```

2) 요약 실행
```bash
./scripts/run_bertsum_summary.sh /path/to/input.txt
```

필요하면 글자 수 제한도 전달 가능:
```bash
./scripts/run_bertsum_summary.sh /path/to/input.txt 150
```

