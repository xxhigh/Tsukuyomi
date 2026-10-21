import json
from threading import Barrier
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_openai import ChatOpenAI

import api
from agents import docs, orchestration, web
from agents.graph import build_graph, run_config
from agents.state import AnswerAssessment, EvidenceAssessment, ResearchPlan, ResearchTask, Route, TaskAssessment
from tests.test_graph import EVIDENCE


class AgentLLM:
    def __init__(self, plans, assessments=(), reviews=(), answers=("확인된 내용입니다. [1]",), routes=None):
        self.outputs = {Route: iter(routes or [Route(worker="orchestrator", reason="복합 조사")]),
                        ResearchPlan: iter(plans), EvidenceAssessment: iter(assessments),
                        AnswerAssessment: iter(reviews)}
        self.answers = iter(answers)
        self.calls = []

    def with_structured_output(self, schema, **kwargs):
        def invoke(messages):
            self.calls.append((schema, json.loads(messages[-1].content)))
            value = next(self.outputs[schema])
            return value(messages) if callable(value) else value
        return SimpleNamespace(invoke=invoke)

    def invoke(self, messages):
        self.calls.append(("writer", json.loads(messages[-1].content)))
        value = next(self.answers)
        return SimpleNamespace(content=value(messages) if callable(value) else value)


def task(number=1, query="nanite", worker="technique", depends_on=()):
    return ResearchTask(id=number, question=f"조사 질문 {number}", query=query, worker=worker, depends_on=list(depends_on))


def plan(*tasks):
    return ResearchPlan(tasks=list(tasks), reason="다음 조사")


def assessment(number=1, sources=(1,), sufficient=True, finding="확인된 내용", missing=""):
    return TaskAssessment(task_id=number, relevant_sources=list(sources), sufficient=sufficient,
                          finding=finding, missing=missing)


def assessed(*tasks):
    return EvidenceAssessment(tasks=list(tasks))


def reviewed(supported=True, addresses=True, feedback="근거 확인"):
    return AnswerAssessment(supported=supported, addresses_question=addresses, feedback=feedback)


def test_decomposition_parallel_search_and_stable_source_attribution():
    barrier = Barrier(2)
    calls = []

    def retrieve(query, **options):
        calls.append((query, options))
        barrier.wait(timeout=3)  # 순차 실행이면 실패한다.
        return [{**EVIDENCE, "doc_id": query, "content": f"{query} 본문"}]

    model = AgentLLM([plan(task(1, "원인"), task(2, "우회책", "troubleshooting")), plan()],
                     [assessed(assessment(1, (1,)), assessment(2, (2,)))], [reviewed()],
                     ["원인입니다. [1] 우회책입니다. [2]"])
    result = build_graph(model, retrieve).invoke({"question": "원인과 우회책을 조사해줘", "engine_version": "5.5"},
                                                 config=run_config("test"))
    assert result["status"] == "done"
    assert result["rounds"] == 2
    assert {item[0] for item in calls} == {"원인", "우회책"}
    assert {item[1]["kind"] for item in calls} == {"technique", "error"}
    assert all(item[1]["engine_version"] == "5.5" for item in calls)
    assert [item["doc_id"] for item in result["sources"]] == ["원인", "우회책"]
    assert all(item["status"] == "done" for item in result["research_tasks"])
    assert [schema for schema, _ in model.calls] == [Route, ResearchPlan, EvidenceAssessment, ResearchPlan, "writer", AnswerAssessment]


def test_multi_hop_reads_body_then_rewrites_followup_using_prior_task():
    first = {**EVIDENCE, "content": "오류 원인 후보는 RenderModuleX다. 수정 여부는 별도 확인이 필요하다."}
    second = {**EVIDENCE, "doc_id": "patch", "content": "RenderModuleX는 5.5.2에서 수정됐다."}

    def followup(messages):
        payload = json.loads(messages[-1].content)
        assert "RenderModuleX" in payload["evidence"][0]["content"]
        assert payload["tasks"][0]["missing"] == "패치 수정 여부"
        return plan(task(2, "RenderModuleX 5.5.2 수정 기록", depends_on=(1,)))

    queries = []
    def retrieve(query, **options):
        queries.append(query)
        return [first if len(queries) == 1 else second]

    model = AgentLLM([plan(task(1, "오류 원인")), followup, plan()],
                     [assessed(assessment(sufficient=False, finding="RenderModuleX 원인 후보", missing="패치 수정 여부")),
                      assessed(assessment(2, (2,)))], [reviewed()], ["원인 후보와 수정 기록입니다. [1][2]"])
    result = build_graph(model, retrieve).invoke({"question": "이 오류 원인과 수정 버전을 알려줘"})
    assert result["status"] == "done"
    assert queries == ["오류 원인", "RenderModuleX 5.5.2 수정 기록"]
    assert result["research_tasks"][1]["depends_on"] == [1]
    assert result["research_tasks"][1]["dependency_sources"] == [1]
    assert result["accepted_sources"] == [1, 2]


def test_rewrite_after_empty_search_passes_failure_to_planner():
    def rewrite(messages):
        payload = json.loads(messages[-1].content)
        assert payload["tasks"][0]["status"] == "insufficient"
        return plan(task(2, "ENQUEUE_RENDER_COMMAND crash", depends_on=(1,)))

    queries = []
    def retrieve(query, **kwargs):
        queries.append(query)
        return [] if len(queries) == 1 else [EVIDENCE]

    model = AgentLLM([plan(task(query="렌더 명령 오류")), rewrite, plan()],
                     [assessed(assessment(2))], [reviewed()])
    result = build_graph(model, retrieve).invoke({"question": "오류 원인과 해결 기록을 찾아줘"})
    assert result["status"] == "done"
    assert queries == ["렌더 명령 오류", "ENQUEUE_RENDER_COMMAND crash"]


@pytest.mark.parametrize("bad_task", [task(1, "다른 검색"), task(2, " NANITE  "),
                                      task(2, "후속", depends_on=(2,)), task(2, "후속", depends_on=(6,))])
def test_duplicate_queries_ids_and_invalid_dependencies_stop_without_more_search(bad_task):
    model = AgentLLM([plan(task()), plan(bad_task)], [assessed(assessment())], [reviewed()])
    calls = []
    result = build_graph(model, lambda query, **kwargs: calls.append(query) or [EVIDENCE]).invoke({"question": "복합 질문"})
    assert calls == ["nanite"]
    assert result["status"] == "done"
    assert result["warnings"]


def test_six_search_budget_and_source_deduplication():
    model = AgentLLM([plan(*[task(i, f"query {i}") for i in range(1, 4)]),
                      plan(*[task(i, f"query {i}") for i in range(4, 7)])],
                     [assessed(*[assessment(i) for i in range(1, 4)]),
                      assessed(*[assessment(i) for i in range(4, 7)])], [reviewed()])
    calls = []
    result = build_graph(model, lambda query, **kwargs: calls.append(query) or [EVIDENCE]).invoke({"question": "복합 질문"},
                                                                                             config=run_config("test"))
    assert len(calls) == result["rounds"] == orchestration.MAX_COMPLEX_SEARCHES
    assert result["planning_rounds"] == 2
    assert len(result["sources"]) == len(result["evidence"]) == 1
    assert "횟수 한도" in str(result["warnings"])


def test_four_planning_rounds_bound_sequential_chain():
    model = AgentLLM([plan(task(i, f"query {i}", depends_on=(i - 1,) if i > 1 else ())) for i in range(1, 5)],
                     [assessed(assessment(i)) for i in range(1, 5)], [reviewed()])
    result = build_graph(model, lambda *args, **kwargs: [EVIDENCE]).invoke({"question": "복합 질문"}, config=run_config("test"))
    assert result["rounds"] == result["planning_rounds"] == 4
    assert result["status"] == "done"


@pytest.mark.parametrize("phase", ["plan", "search", "draft"])
def test_time_budget_stops_new_calls_without_releasing_unreviewed_answer(monkeypatch, phase):
    clock = [0]
    monkeypatch.setattr(orchestration, "monotonic", lambda: clock[0])

    def expire(value):
        def action(*args, **kwargs):
            clock[0] = orchestration.MAX_RESEARCH_SECONDS
            return value
        return action

    model = AgentLLM([expire(plan(task())) if phase == "plan" else plan(task()), plan()],
                     [assessed(assessment())], [], [expire("검증 전 초안 [1]")])
    retrieve = expire([EVIDENCE]) if phase == "search" else lambda *args, **kwargs: [EVIDENCE]
    result = build_graph(model, retrieve).invoke({"question": "복합 질문"})
    assert result["status"] == "insufficient"
    assert result["sources"] == []
    assert AnswerAssessment not in [schema for schema, _ in model.calls]
    assert result.get("rounds", 0) == (0 if phase == "plan" else 1)
    if phase in {"plan", "search"}:
        assert "writer" not in [schema for schema, _ in model.calls]


@pytest.mark.parametrize("reviews,expected", [([reviewed(False), reviewed()], "done"),
                                            ([reviewed(False), reviewed(False)], "insufficient"),
                                            ([reviewed(True, False), reviewed(True, False)], "insufficient")])
def test_answer_review_revision_and_final_rejection(reviews, expected):
    model = AgentLLM([plan(task()), plan()], [assessed(assessment())], reviews,
                     ["근거보다 과한 주장 [1]", "수정된 답변 [1]"])
    result = build_graph(model, lambda *args, **kwargs: [EVIDENCE]).invoke({"question": "복합 질문"})
    assert result["status"] == expected
    assert result["revision_count"] == 1
    writers = [payload for schema, payload in model.calls if schema == "writer"]
    assert len(writers) == 2
    assert writers[1]["previous_draft"] == "근거보다 과한 주장 [1]"
    assert writers[1]["revision_feedback"] == "근거 확인"
    if expected == "done":
        assert result["answer"] == "수정된 답변 [1]"
    else:
        assert result["sources"] == []
        assert "과한 주장" not in result["answer"]


@pytest.mark.parametrize("reviews", [assessed(assessment(sources=(99,))), assessed(),
                                    assessed(assessment(), assessment())])
def test_invalid_or_missing_evidence_assessment_never_reaches_writer(reviews):
    model = AgentLLM([plan(task()), plan()], [reviews])
    result = build_graph(model, lambda *args, **kwargs: [EVIDENCE]).invoke({"question": "복합 질문"})
    assert result["status"] == "insufficient"
    assert result["accepted_sources"] == result["sources"] == []
    assert "writer" not in [schema for schema, _ in model.calls]


@pytest.mark.parametrize("answer,status", [("사용한 자료입니다. [2]", "done"),
                                         ("제외된 자료입니다. [1]", "invalid_citations")])
def test_irrelevant_sources_are_excluded_without_renumbering(answer, status):
    model = AgentLLM([plan(task()), plan()], [assessed(assessment(sources=(2,)))], [reviewed()], [answer])
    irrelevant = {**EVIDENCE, "doc_id": "irrelevant", "content": "IRRELEVANT_BODY"}
    result = build_graph(model, lambda *args, **kwargs: [irrelevant, EVIDENCE]).invoke({"question": "복합 질문"})
    assert result["status"] == status
    writer = next(payload for schema, payload in model.calls if schema == "writer")
    assert [item["number"] for item in writer["evidence"]] == [2]
    assert "IRRELEVANT_BODY" not in str(writer)
    if status == "done":
        assert result["sources"][0]["number"] == 2


def test_assessment_cannot_attribute_another_independent_tasks_sources():
    def retrieve(query, **kwargs):
        return [{**EVIDENCE, "doc_id": query}]
    model = AgentLLM([plan(task(1, "first"), task(2, "second")), plan()],
                     [assessed(assessment(1, (2,), finding="MISATTRIBUTED_FINDING"), assessment(2, (2,)))],
                     [reviewed()], ["확인된 범위입니다. [2]"])
    result = build_graph(model, retrieve).invoke({"question": "복합 질문"})
    assert result["status"] == "done"
    assert result["accepted_sources"] == [2]
    assert result["research_tasks"][0]["status"] == "insufficient"
    assert result["research_tasks"][0]["finding"] == ""
    assert "MISATTRIBUTED_FINDING" not in str(result["research_tasks"])


def test_dependent_task_is_not_dispatched_with_its_unfinished_predecessor():
    calls = []
    model = AgentLLM([plan(task(1, "first"), task(2, "dependent", depends_on=(1,))), plan()],
                     [assessed(assessment())], [reviewed()])
    result = build_graph(model, lambda query, **kwargs: calls.append(query) or [EVIDENCE]).invoke({"question": "복합 질문"})
    assert calls == ["first"]
    assert result["warnings"]


def test_empty_plan_or_empty_results_return_insufficient_without_generation():
    for plans in ([plan()], [plan(task()), plan()]):
        model = AgentLLM(plans)
        result = build_graph(model, lambda *args, **kwargs: []).invoke({"question": "복합 질문"})
        assert result["status"] == "insufficient"
        assert result["sources"] == []
        assert "writer" not in [schema for schema, _ in model.calls]


def test_web_only_workers_do_not_load_local_index_or_send_raw_question(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("웹 전용 조사에서 로컬 인덱스에 접근했습니다.")
    monkeypatch.setattr(docs, "_fingerprint", forbidden)
    monkeypatch.setattr(docs, "retrieve", forbidden)
    calls = []
    def search(query, **kwargs):
        calls.append((query, kwargs))
        return [{**EVIDENCE, "source_type": "web", "engine_version": "미지정", "url": "https://example.com"}]
    model = AgentLLM([plan(task(query="Nanite 5.4.2 hotfix", worker="web_search")), plan()],
                     [assessed(assessment())], [reviewed()])
    result = build_graph(model, web_search=search).invoke({"question": "PRIVATE_INTERNAL_QUESTION", "engine_version": "5.4"})
    assert result["status"] == "done"
    assert calls == [("Nanite 5.4.2 hotfix", {"engine_version": "5.4"})]
    assert result["sources"][0]["engine_version"] == "미지정"


def test_web_failure_preserves_local_results_and_task_failure():
    def search(*args, **kwargs):
        raise web.WebSearchError("rate_limited")
    model = AgentLLM([plan(task(), task(2, "Nanite hotfix", "web_search")), plan()],
                     [assessed(assessment(), assessment(2, (), False, "", "웹 검색 실패"))], [reviewed()])
    result = build_graph(model, lambda *args, **kwargs: [EVIDENCE], web_search=search).invoke({"question": "복합 질문"})
    assert result["status"] == "done"
    assert result["research_tasks"][1]["status"] == "failed"
    assert result["warnings"]
    assert len(result["sources"]) == 1


def test_unavailable_web_falls_back_and_version_filter_remains_fixed():
    calls = []
    def retrieve(query, **options):
        calls.append(options)
        return [EVIDENCE, {**EVIDENCE, "doc_id": "wrong", "engine_version": "5.6"}]
    model = AgentLLM([plan(task(worker="web_search")), plan()], [assessed(assessment())], [reviewed()])
    result = build_graph(model, retrieve).invoke({"question": "복합 질문", "engine_version": "5.5"})
    assert calls == [{"k": 5, "kind": "technique", "engine_version": "5.5"}]
    assert result["research_tasks"][0]["worker"] == "technique"
    assert [item["doc_id"] for item in result["evidence"]] == [EVIDENCE["doc_id"]]
    assert result["warnings"]


def test_api_complex_response_keeps_tasks_out_of_response_and_trace(workspace, monkeypatch):
    monkeypatch.delenv("INTERNAL_API_TOKEN", raising=False)
    monkeypatch.setattr(api.trace, "TRACE_DIR", workspace / "traces")
    model = AgentLLM([plan(task(query="PRIVATE_TASK_QUERY")), plan()], [assessed(assessment(finding="PRIVATE_FINDING"))], [reviewed()])
    monkeypatch.setattr(api, "create_llm", lambda *args: model)
    monkeypatch.setattr(api, "build_graph", lambda llm: build_graph(llm, lambda *args, **kwargs: [EVIDENCE]))
    response = TestClient(api.app).post("/api/chat", headers={"X-LLM-API-Key": "SECRET_KEY"}, json={"question": "복합 질문"})
    assert response.status_code == 200
    assert response.json()["status"] == "done"
    assert response.json()["sources"][0]["doc_id"] == EVIDENCE["doc_id"]
    assert "research_tasks" not in response.json()
    trace = "".join(path.read_text() for path in (workspace / "traces").glob("*.jsonl"))
    for private in ("PRIVATE_TASK_QUERY", "PRIVATE_FINDING", "SECRET_KEY"):
        assert private not in response.text
        assert private not in trace


def test_new_schemas_work_with_real_sdk_and_mock_http():
    outputs = {"Route": iter([Route(worker="orchestrator", reason="복합 질문")]),
               "ResearchPlan": iter([plan(task()), plan()]),
               "EvidenceAssessment": iter([assessed(assessment())]),
               "AnswerAssessment": iter([reviewed()])}
    seen = []
    def respond(request):
        body = json.loads(request.content)
        tools = body.get("tools", [])
        if tools:
            name = tools[0]["function"]["name"]
            seen.append(name)
            message = {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call-test", "type": "function", "function": {
                    "name": name, "arguments": next(outputs[name]).model_dump_json()}}]}
        else:
            message = {"role": "assistant", "content": "확인된 내용입니다. [1]"}
        return httpx.Response(200, json={"id": "test", "object": "chat.completion", "created": 0, "model": "test",
                                        "choices": [{"index": 0, "finish_reason": "tool_calls" if tools else "stop", "message": message}]})
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        llm = ChatOpenAI(api_key="test-only", model="test", http_client=transport)
        result = build_graph(llm, lambda *args, **kwargs: [EVIDENCE]).invoke({"question": "복합 질문"})
    assert result["status"] == "done"
    assert seen == ["Route", "ResearchPlan", "EvidenceAssessment", "ResearchPlan", "AnswerAssessment"]


@pytest.mark.parametrize("phase", ["assessor", "planner", "writer", "reviewer"])
def test_local_corpus_change_during_complex_llm_calls_aborts(workspace, monkeypatch, phase):
    from agents import knowledge
    from tests.test_retrieval import LocalEmbeddings, add

    monkeypatch.setattr(docs, "_embeddings", LocalEmbeddings)
    add("nanite")
    reference = knowledge.DOCS_DIR / "reference.md"
    reference.write_text("nanite original reference", encoding="utf-8")

    def change_then(value):
        def action(messages):
            reference.write_text("nanite changed reference", encoding="utf-8")
            return value
        return action

    model = AgentLLM([plan(task()), change_then(plan()) if phase == "planner" else plan()],
                     [change_then(assessed(assessment())) if phase == "assessor" else assessed(assessment())],
                     [change_then(reviewed()) if phase == "reviewer" else reviewed()],
                     [change_then("확인된 내용 [1]") if phase == "writer" else "확인된 내용 [1]"])
    with pytest.raises(docs.CorpusChangedError):
        build_graph(model).invoke({"question": "nanite 복합 질문", "engine_version": "5.5"})


def test_mixed_web_change_after_local_batch_is_detected(workspace, monkeypatch):
    from agents import knowledge
    from tests.test_retrieval import LocalEmbeddings, add

    monkeypatch.setattr(docs, "_embeddings", LocalEmbeddings)
    add("nanite")
    reference = knowledge.DOCS_DIR / "reference.md"
    reference.write_text("nanite original reference", encoding="utf-8")
    def search(*args, **kwargs):
        reference.write_text("changed reference", encoding="utf-8")
        return [{**EVIDENCE, "doc_id": "web", "source_type": "web", "engine_version": "미지정"}]
    model = AgentLLM([plan(task()), plan(task(2, "Nanite hotfix", "web_search", depends_on=(1,)))],
                     [assessed(assessment())])
    with pytest.raises(docs.CorpusChangedError):
        build_graph(model, web_search=search).invoke({"question": "nanite 복합 질문"})


def test_reused_graph_has_independent_research_state():
    model = AgentLLM([plan(task()), plan(), plan(task()), plan()],
                     [assessed(assessment()), assessed(assessment())], [reviewed(), reviewed()],
                     ["첫 답변 [1]", "다음 답변 [1]"],
                     [Route(worker="orchestrator", reason="복합 질문")] * 2)
    graph = build_graph(model, lambda *args, **kwargs: [EVIDENCE])
    first = graph.invoke({"question": "첫 질문"})
    second = graph.invoke({"question": "다음 질문"})
    assert first["rounds"] == second["rounds"] == 1
    assert len(first["research_tasks"]) == len(second["research_tasks"]) == 1
    assert first["answer"] == "첫 답변 [1]"
    assert second["answer"] == "다음 답변 [1]"


def test_cli_complex_answer_uses_existing_output(monkeypatch, capsys):
    import app
    import sys

    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setattr(sys, "argv", ["app.py", "복합 질문"])
    model = AgentLLM([plan(task()), plan()], [assessed(assessment())], [reviewed()])
    monkeypatch.setattr(app, "create_llm", lambda *args: model)
    monkeypatch.setattr(app, "build_graph", lambda llm: build_graph(llm, lambda *args, **kwargs: [EVIDENCE]))
    assert app.main() == 0
    output = capsys.readouterr().out
    assert "확인된 내용입니다. [1]" in output
    assert EVIDENCE["title"] in output
