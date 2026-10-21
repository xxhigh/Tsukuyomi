import json
from types import SimpleNamespace

import pytest

from agents.graph import MAX_RESEARCH_ROUNDS, MAX_SOURCE_EXCERPT, build_graph
from agents.state import Route

EVIDENCE = {"doc_id": "note1", "chunk_id": 0, "title": "Shader 사례", "kind": "error",
            "engine_version": "5.5", "content": "확인된 해결 절차", "url": "", "score": 0.9, "page": 0}


class FakeLLM:
    def __init__(self, routes, answer="등록된 해결 절차를 확인하세요. [1]"):
        self.routes = iter(routes)
        self.answer = answer
        self.writer_calls = 0
        self.router_messages = []
        self.writer_messages = []

    def with_structured_output(self, *args, **kwargs):
        owner = self
        class Router:
            def invoke(self, messages):
                owner.router_messages.append(messages)
                return next(owner.routes)
        return Router()

    def invoke(self, messages):
        self.writer_calls += 1
        self.writer_messages.append(messages)
        return SimpleNamespace(content=self.answer)


def test_supervisor_routes_errors_researches_and_preserves_followup_context():
    model = FakeLLM([Route(worker="troubleshooting", query="UE 5.5 shader", reason="오류 검색"),
                     Route(worker="answer", reason="근거 확보")])
    calls = []
    def retrieve(query, **kwargs):
        calls.append((query, kwargs))
        return [EVIDENCE]
    result = build_graph(model, retrieve).invoke({"question": "그 오류 해결법은?", "history": [
        {"role": "user", "content": "UE 5.5 shader 오류"}]})
    assert calls == [("UE 5.5 shader", {"k": 5, "kind": "error"})]
    assert "UE 5.5 shader 오류" in str(model.router_messages[0])
    assert result["status"] == "done"
    assert result["sources"][0]["number"] == 1
    assert "content" not in result["sources"][0]


def test_search_is_bounded_and_deduplicates_sources():
    model = FakeLLM([Route(worker="technique", query="Nanite", reason="검색") for _ in range(2)])
    result = build_graph(model, lambda *args, **kwargs: [EVIDENCE]).invoke({"question": "기술 질문"})
    assert result["rounds"] == MAX_RESEARCH_ROUNDS
    assert len(result["sources"]) == 1
    assert result["decisions"][-1]["worker"] == "answer"


def test_supervisor_cannot_skip_search_and_empty_evidence_never_calls_writer():
    model = FakeLLM([Route(worker="answer", reason="완료"), Route(worker="answer", reason="자료 없음")])
    result = build_graph(model, lambda *args, **kwargs: []).invoke({"question": "모르는 기술"})
    assert result["rounds"] == 1
    assert result["status"] == "insufficient"
    assert model.writer_calls == 0
    assert result["sources"] == []


@pytest.mark.parametrize("length", [1000, 1001, 8000])
@pytest.mark.parametrize("has_evidence", [False, True])
def test_forced_first_search_preserves_long_question(length, has_evidence):
    question = "오" * (length - 8) + "LOG_END!"
    model = FakeLLM([Route(worker="answer", query="사용하지 않을 검색어", reason="완료"),
                     Route(worker="answer", reason="검색 완료")])
    calls = []

    def retrieve(query, **options):
        calls.append((query, options))
        return [EVIDENCE] if has_evidence else []

    result = build_graph(model, retrieve).invoke({"question": question, "engine_version": "5.5"})

    assert calls == [(question, {"k": 5, "kind": "technique", "engine_version": "5.5"})]
    assert result["rounds"] == 1
    assert [decision["worker"] for decision in result["decisions"]] == ["technique", "answer"]
    assert len(model.router_messages) == 2
    assert json.loads(model.router_messages[0][-1].content)["question"] == question
    assert result["status"] == ("done" if has_evidence else "insufficient")
    assert model.writer_calls == int(has_evidence)
    if has_evidence:
        assert json.loads(model.writer_messages[0][-1].content)["question"] == question
        assert result["sources"][0]["doc_id"] == EVIDENCE["doc_id"]
    else:
        assert result["sources"] == []


@pytest.mark.parametrize("answer", ["근거 없는 답변", "잘못된 출처 [2]", "혼합 [1] [99]"])
def test_invalid_citations_do_not_reach_user(answer):
    model = FakeLLM([Route(worker="technique", reason="검색"), Route(worker="answer", reason="완료")], answer)
    result = build_graph(model, lambda *args, **kwargs: [EVIDENCE]).invoke({"question": "질문"})
    assert result["status"] == "invalid_citations"
    assert result["answer"] != answer
    assert result["sources"] == []


def test_engine_version_reaches_supervisor_retriever_and_writer():
    model = FakeLLM([Route(worker="troubleshooting", reason="검색"), Route(worker="answer", reason="완료")])
    calls = []
    def retrieve(query, **options):
        calls.append(options)
        return [EVIDENCE, {**EVIDENCE, "doc_id": "other", "engine_version": "5.6", "content": "wrong-version-body"}]
    result = build_graph(model, retrieve).invoke({"question": "shader 오류", "engine_version": "5.5"})
    assert calls == [{"k": 5, "kind": "error", "engine_version": "5.5"}]
    assert '"engine_version": "5.5"' in str(model.router_messages[0])
    assert '"engine_version": "5.5"' in str(model.writer_messages[0])
    assert "wrong-version-body" not in str(model.writer_messages[0])
    assert {source["engine_version"] for source in result["sources"]} == {"5.5"}


def test_missing_version_returns_insufficient_without_cross_version_answer():
    model = FakeLLM([Route(worker="technique", reason="검색"), Route(worker="answer", reason="완료")])
    result = build_graph(model, lambda *args, **kwargs: [EVIDENCE]).invoke(
        {"question": "shader", "engine_version": "5.7"})
    assert result["status"] == "insufficient"
    assert "5.7" in result["answer"]
    assert result["sources"] == []
    assert model.writer_calls == 0


@pytest.mark.parametrize("content", ["", "코드\n<script>window.bad = true</script>", "가" * MAX_SOURCE_EXCERPT,
                                     "가" * MAX_SOURCE_EXCERPT + "DO_NOT_RETURN_TAIL"])
def test_source_excerpts_are_exact_bounded_snapshots_of_cited_evidence(content):
    model = FakeLLM([Route(worker="technique", reason="검색"), Route(worker="answer", reason="완료")])
    evidence = {**EVIDENCE, "content": content}
    result = build_graph(model, lambda *args, **kwargs: [evidence]).invoke({"question": "기술 질문"})
    source = result["sources"][0]
    assert source["excerpt"] == content[:MAX_SOURCE_EXCERPT]
    assert source["excerpt_truncated"] is (len(content) > MAX_SOURCE_EXCERPT)
    assert "content" not in source
    assert {key: source[key] for key in EVIDENCE if key != "content"} == {
        key: value for key, value in evidence.items() if key != "content"}
    # 표시 길이와 관계없이 답변 모델에는 검색된 원문을 그대로 전달한다.
    import json
    writer_input = json.loads(model.writer_messages[0][-1].content)
    assert writer_input["evidence"][0]["content"] == content
    assert "excerpt" not in evidence
    evidence["content"] = "수정된 문서"
    assert source["excerpt"] == content[:MAX_SOURCE_EXCERPT]


def test_uncited_sources_are_excluded_and_citation_numbers_are_preserved():
    first = {**EVIDENCE, "doc_id": "unused", "chunk_id": 0, "content": "UNCITED_PRIVATE_CONTENT"}
    second = {**EVIDENCE, "doc_id": "cited", "chunk_id": 1, "content": "인용된 검색 원문"}
    model = FakeLLM([Route(worker="technique", reason="검색"), Route(worker="answer", reason="완료")], "확인하세요. [2]")
    result = build_graph(model, lambda *args, **kwargs: [first, second]).invoke({"question": "질문"})
    assert [source["number"] for source in result["sources"]] == [2]
    assert result["sources"][0]["excerpt"] == second["content"]
    assert "UNCITED_PRIVATE_CONTENT" not in str(result["sources"])
