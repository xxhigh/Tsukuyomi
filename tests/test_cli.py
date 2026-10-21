import sys
from types import SimpleNamespace

import pytest

import app
from agents.docs import CorpusChangedError


def test_cli_reports_corpus_change_without_printing_question_or_provider_payload(monkeypatch, capsys):
    monkeypatch.setenv("OPENAI_API_KEY", "test-provider-secret")
    monkeypatch.setattr(sys, "argv", ["app.py", "private-question"])
    monkeypatch.setattr(app, "create_llm", lambda *args: object())
    def fail(*args, **kwargs):
        raise CorpusChangedError("private failure payload")
    monkeypatch.setattr(app, "build_graph", lambda *args: SimpleNamespace(invoke=fail))
    assert app.main() == 1
    output = capsys.readouterr().out
    assert "지식이 변경" in output
    assert "다시 보내" in output
    assert "private" not in output
    assert "test-provider-secret" not in output


@pytest.mark.parametrize("status, exit_code", [("done", 0), ("insufficient", 0), ("invalid_citations", 1)])
def test_cli_single_question_exit_code_reflects_citation_failure(monkeypatch, capsys, status, exit_code):
    monkeypatch.setenv("OPENAI_API_KEY", "test-provider-secret")
    monkeypatch.setattr(sys, "argv", ["app.py", "질문"])
    monkeypatch.setattr(app, "create_llm", lambda *args: object())
    monkeypatch.setattr(app, "build_graph", lambda *args: SimpleNamespace(invoke=lambda *args, **kwargs: {
        "status": status, "answer": "안전한 결과 안내", "sources": [],
    }))

    assert app.main() == exit_code
    assert "안전한 결과 안내" in capsys.readouterr().out


@pytest.mark.parametrize("engine_version", [None, "5.5"])
def test_cli_interactive_excludes_invalid_citations_and_keeps_other_turns(monkeypatch, capsys, engine_version):
    monkeypatch.setenv("OPENAI_API_KEY", "test-provider-secret")
    monkeypatch.setattr(sys, "argv", ["app.py", *(["--engine-version", engine_version] if engine_version else [])])
    monkeypatch.setattr(app, "create_llm", lambda *args: object())
    questions = iter(["정상 질문", "검증 실패 질문", "근거 없는 질문", "후속 질문", "exit"])
    statuses = iter(["done", "invalid_citations", "insufficient", "done"])
    calls = []

    def invoke(state, **kwargs):
        calls.append(state)
        status = next(statuses)
        return {"status": status, "answer": status + " 안내", "sources": []}

    monkeypatch.setattr(app, "build_graph", lambda *args: SimpleNamespace(invoke=invoke))
    monkeypatch.setattr("builtins.input", lambda *args: next(questions))

    assert app.main() == 0
    valid_turn = [{"role": "user", "content": "정상 질문"}, {"role": "assistant", "content": "done 안내"}]
    assert calls[0]["history"] == []
    assert calls[1]["history"] == valid_turn
    assert calls[2]["history"] == valid_turn
    assert calls[3]["history"] == [*valid_turn, {"role": "user", "content": "근거 없는 질문"},
                                   {"role": "assistant", "content": "insufficient 안내"}]
    assert {call["engine_version"] for call in calls} == {engine_version}
    assert "invalid_citations 안내" in capsys.readouterr().out


@pytest.mark.parametrize("arguments, question, version", [
    (["shader", "오류"], "shader 오류", None),
    (["shader", "오류", "--engine-version", " 5.5 "], "shader 오류", "5.5"),
    (["--engine-version", "v" * 50, "shader"], "shader", "v" * 50),
    (["--", "--question-text"], "--question-text", None),
])
def test_cli_question_and_version_arguments(monkeypatch, arguments, question, version):
    monkeypatch.setenv("OPENAI_API_KEY", "test-provider-secret")
    monkeypatch.setattr(sys, "argv", ["app.py", *arguments])
    monkeypatch.setattr(app, "create_llm", lambda *args: object())
    calls = []

    def invoke(state, **kwargs):
        calls.append(state)
        return {"status": "insufficient", "answer": "근거 없음", "sources": []}

    monkeypatch.setattr(app, "build_graph", lambda *args: SimpleNamespace(invoke=invoke))
    assert app.main() == 0
    assert calls == [{"question": question, "history": [], "engine_version": version}]


@pytest.mark.parametrize("arguments, code", [
    (["--help"], 0), (["--engine-version", ""], 2),
    (["--engine-version", "   "], 2), (["--engine-version", "v" * 51], 2),
])
def test_cli_help_and_invalid_versions_precede_key_prompt_and_llm(monkeypatch, capsys, arguments, code):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["app.py", *arguments])

    def unexpected(*args):
        pytest.fail("도움말·입력 오류에서 키 입력이나 LLM 생성에 도달하면 안 됩니다")

    monkeypatch.setattr(app, "getpass", unexpected)
    monkeypatch.setattr(app, "create_llm", unexpected)
    with pytest.raises(SystemExit) as error:
        app.main()
    assert error.value.code == code
    captured = capsys.readouterr()
    assert "--engine-version" in captured.out + captured.err


@pytest.mark.parametrize("version", ["5.5", "5.6"])
def test_cli_selected_version_reaches_retriever_without_cross_version_answer(monkeypatch, capsys, version):
    from agents.graph import build_graph
    from agents.state import Route
    from tests.test_graph import EVIDENCE, FakeLLM

    monkeypatch.setenv("OPENAI_API_KEY", "test-provider-secret")
    monkeypatch.setattr(sys, "argv", ["app.py", "--engine-version", version, "shader 오류"])
    model = FakeLLM([Route(worker="troubleshooting", reason="검색"), Route(worker="answer", reason="완료")])
    calls = []

    def retrieve(query, **options):
        calls.append(options)
        return [EVIDENCE]

    monkeypatch.setattr(app, "create_llm", lambda *args: model)
    monkeypatch.setattr(app, "build_graph", lambda llm: build_graph(llm, retrieve))
    assert app.main() == 0
    assert calls == [{"k": 5, "kind": "error", "engine_version": version}]
    output = capsys.readouterr().out
    if version == "5.5":
        assert "[1] Shader 사례 | UE 5.5" in output
        assert model.writer_calls == 1
    else:
        assert "선택한 엔진 버전(5.6)" in output
        assert "Shader 사례" not in output
        assert model.writer_calls == 0
