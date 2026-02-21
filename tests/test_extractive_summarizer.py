import os
import subprocess

import pytest

from src.extractive_summarizer import split_sentences, summarize_extractive


def test_summary_respects_max_chars():
    text = (
        "이 프로젝트는 긴 문서를 입력받아 핵심 정보를 빠르게 전달하는 요약 서비스를 목표로 한다. "
        "초기 버전에서는 추출 요약을 적용해 중요한 문장을 선별한다. "
        "최종 출력은 150자 이내로 제한하여 모바일 환경에서도 읽기 쉽도록 설계한다."
    )
    result = summarize_extractive(text, max_chars=150)
    assert len(result) <= 150
    assert result


def test_bertsum_backend_requires_runner():
    with pytest.raises(RuntimeError):
        summarize_extractive("테스트 문장입니다.", backend="bertsum-korean")


def test_bertsum_backend_with_runner_script_fallback_mode():
    text = "첫 문장입니다. 두 번째 문장입니다. 세 번째 문장입니다."
    result = summarize_extractive(
        text,
        backend="bertsum-korean",
        bertsum_runner="scripts/bertsum_runner_example.py",
        max_chars=150,
    )
    assert result
    assert len(result) <= 150


def test_cli_bertsum_backend_with_runner_script_env_config():
    cmd = [
        "python",
        "-m",
        "src.extractive_summarizer",
        "--text",
        "첫 문장입니다. 두 번째 문장입니다. 세 번째 문장입니다.",
        "--backend",
        "bertsum-korean",
        "--bertsum-runner",
        "scripts/bertsum_runner_example.py",
    ]
    env = {
        **os.environ,
        "BERTSUM_KOREAN_REPO": ".",
        "BERTSUM_KOREAN_SCORER": "tools.inference_runner",
        "BERTSUM_USE_FALLBACK": "1",
    }
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
    assert "[chars=" in completed.stdout


def test_split_sentences_handles_quotes_and_parentheses():
    text = '그는 말했다. "정말인가?" (확실해!) 그리고 끝.'
    assert split_sentences(text) == ["그는 말했다.", '"정말인가?"', "(확실해!)", "그리고 끝."]


def test_split_sentences_handles_newlines_and_special_characters():
    text = "첫 줄입니다.\n둘째 줄입니다?!\n\n#해시태그 테스트. 마지막🙂문장입니다!"
    assert split_sentences(text) == [
        "첫 줄입니다.",
        "둘째 줄입니다?!",
        "#해시태그 테스트.",
        "마지막🙂문장입니다!",
    ]


def test_split_sentences_accepts_custom_splitter():
    class DummySplitter:
        def split(self, text: str):
            return ["A", "B"]

    assert split_sentences("무시되는 입력", splitter=DummySplitter()) == ["A", "B"]


def test_korean_advanced_splitter_available_or_reports_missing_dependency():
    try:
        result = split_sentences("문장입니다. 다음 문장입니다.", splitter="korean-advanced")
    except RuntimeError as exc:
        assert "requires `kss` package" in str(exc)
    else:
        assert result
