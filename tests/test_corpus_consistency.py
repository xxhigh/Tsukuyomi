from concurrent.futures import ThreadPoolExecutor

import pytest

from agents import docs, documents, knowledge
from agents.graph import build_graph
from agents.state import Route
from tests.test_graph import FakeLLM
from tests.test_retrieval import LocalEmbeddings, add


@pytest.fixture
def local_rag(workspace, monkeypatch):
    monkeypatch.setattr(docs, "_embeddings", LocalEmbeddings)
    monkeypatch.setattr(knowledge, "LOCK_TIMEOUT", 0.1)
    record = add("nanite")
    (knowledge.DOCS_DIR / "reference.md").write_text("nanite raw reference", encoding="utf-8")
    return record


def update(record, version="5.6"):
    return knowledge.update(record["id"], knowledge.KnowledgeUpdate(title="shader", kind="technique",
                            engine_version=version, content="shader new solution", expected_revision=1))


def in_another_thread(action):
    # 실제 요청처럼 다른 스레드에서 변경한다. LLM 호출 중 잠금이 남으면 이 작업이 실패한다.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(action).result(timeout=2)


@pytest.mark.parametrize("phase", ["research", "answer", "writer"])
@pytest.mark.parametrize("mutation", ["edit", "delete", "metadata"])
def test_changes_between_retrieval_and_final_validation_abort(local_rag, phase, mutation, monkeypatch):
    def change():
        if mutation == "edit":
            update(local_rag)
        elif mutation == "delete":
            knowledge.delete(local_rag["id"], 1)
        else:
            documents.update("reference.md", documents.DocumentUpdate(title="Changed reference",
                             engine_version="5.6", expected_revision=0))

    def routes():
        yield Route(worker="technique", reason="검색")
        if phase != "writer":
            in_another_thread(change)
        yield Route(worker="technique" if phase == "research" else "answer", reason="다음 단계")

    model = FakeLLM(routes())
    if phase == "writer":
        original_invoke = model.invoke
        def invoke(messages):
            response = original_invoke(messages)
            in_another_thread(change)
            return response
        monkeypatch.setattr(model, "invoke", invoke)
    with pytest.raises(docs.CorpusChangedError):
        build_graph(model).invoke({"question": "nanite", "engine_version": "5.5"})
    assert model.writer_calls == (1 if phase == "writer" else 0)


def test_unchanged_corpus_allows_two_rounds_and_valid_answer(local_rag):
    model = FakeLLM([Route(worker="technique", reason="검색") for _ in range(2)])
    result = build_graph(model).invoke({"question": "nanite", "engine_version": "5.5"})
    assert result["status"] == "done"
    assert result["rounds"] == 2
    assert model.writer_calls == 1
    assert result["corpus_sha256"] == docs._fingerprint()
    assert {source["engine_version"] for source in result["sources"]} == {"5.5"}
    assert all("corpus_sha256" not in source for source in result["sources"])


def test_empty_corpus_changed_before_insufficient_answer_is_detected(workspace, monkeypatch):
    def fail():
        pytest.fail("빈 저장소에서 임베딩을 로드하면 안 됩니다.")
    monkeypatch.setattr(docs, "_embeddings", fail)
    def routes():
        yield Route(worker="technique", reason="검색")
        add("nanite")
        yield Route(worker="answer", reason="자료 없음")
    model = FakeLLM(routes())
    with pytest.raises(docs.CorpusChangedError):
        build_graph(model).invoke({"question": "nanite"})
    assert model.writer_calls == 0


def test_updates_before_first_search_use_latest_corpus(local_rag):
    def routes():
        in_another_thread(lambda: update(local_rag))
        yield Route(worker="technique", reason="검색")
        yield Route(worker="answer", reason="완료")
    result = build_graph(FakeLLM(routes())).invoke({"question": "shader", "engine_version": "5.6"})
    assert result["status"] == "done"
    assert "shader new solution" in result["sources"][0]["excerpt"]
    assert result["sources"][0]["engine_version"] == "5.6"


def test_reused_graph_takes_a_new_snapshot_for_each_question(local_rag):
    model = FakeLLM([Route(worker="technique", reason="검색"), Route(worker="answer", reason="완료")] * 2)
    graph = build_graph(model)
    first = graph.invoke({"question": "nanite", "engine_version": "5.5"})
    update(local_rag)
    second = graph.invoke({"question": "shader", "engine_version": "5.6"})
    assert first["status"] == second["status"] == "done"
    assert first["corpus_sha256"] != second["corpus_sha256"]
    assert "shader new solution" not in first["sources"][0]["excerpt"]
    assert "shader new solution" in second["sources"][0]["excerpt"]


def test_change_during_retrieval_stops_before_another_model_call(local_rag, monkeypatch):
    retrieve = docs.retrieve
    def changing_retrieval(*args, **kwargs):
        found = retrieve(*args, **kwargs)
        (knowledge.DOCS_DIR / "reference.md").write_text("changed raw reference", encoding="utf-8")
        return found
    monkeypatch.setattr(docs, "retrieve", changing_retrieval)
    model = FakeLLM([Route(worker="technique", reason="검색")])
    with pytest.raises(docs.CorpusChangedError):
        build_graph(model).invoke({"question": "nanite", "engine_version": "5.5"})
    assert len(model.router_messages) == 1
    assert model.writer_calls == 0
