import pytest
import httpx
from fastapi.testclient import TestClient

import api
from tests.test_graph import EVIDENCE, FakeLLM
from agents.graph import build_graph
from agents.state import Route


@pytest.fixture
def client(workspace, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("INTERNAL_API_TOKEN", raising=False)
    monkeypatch.setattr(api.trace, "TRACE_DIR", workspace / "traces")
    return TestClient(api.app)


def test_health_key_configuration_and_missing_key(client):
    assert client.get("/api/health").json() == {"status": "ok"}
    assert client.get("/api/config").json()["has_server_key"] is False
    assert client.post("/api/chat", json={"question": "Nanite"}).status_code == 400


def test_save_and_list_knowledge(client):
    response = client.post("/api/knowledge", json={"title": "Nanite", "kind": "technique",
                                                  "engine_version": "5.5", "content": "팀 기술 기록"})
    assert response.status_code == 201
    assert client.get("/api/knowledge").json()["entries"] == [response.json()["entry"]]
    assert client.post("/api/knowledge", json={"title": "Error", "kind": "error",
                                              "engine_version": "5.5", "content": "오류"}).status_code == 422


@pytest.mark.parametrize("version", ["5.4", "5.4.1"])
def test_chat_searches_minor_family_and_returns_original_patch_source(client, monkeypatch, version):
    from agents import docs
    from tests.test_retrieval import LocalEmbeddings

    saved = client.post("/api/knowledge", json={"title": "nanite shader patch", "kind": "technique",
                        "engine_version": "5.4.2", "content": "nanite shader verification in 5.4.2"})
    assert saved.status_code == 201
    monkeypatch.setattr(docs, "_embeddings", LocalEmbeddings)
    monkeypatch.setattr(api, "create_llm", lambda *args: FakeLLM([
        Route(worker="technique", query="nanite shader", reason="검색"), Route(worker="answer", reason="완료")]))
    response = client.post("/api/chat", headers={"X-LLM-API-Key": "test-secret"}, json={
        "question": "5.4.1에서 nanite shader를 사용하려면?", "engine_version": version,
    })
    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "done"
    assert result["engine_version"] == version
    assert result["sources"][0]["engine_version"] == "5.4.2"
    assert result["sources"][0]["doc_id"] == saved.json()["entry"]["id"]


def test_validation_does_not_echo_key_or_arbitrary_body(client):
    response = client.post("/api/chat", headers={"X-LLM-API-Key": "test-secret"},
                           json={"question": " ", "api_key": "test-secret"})
    assert response.status_code == 422
    assert "test-secret" not in response.text


def test_request_key_is_isolated_and_trace_does_not_store_content(client, workspace, monkeypatch):
    seen = []
    def create_llm(key, model):
        seen.append(key)
        return FakeLLM([Route(worker="technique", reason="검색"), Route(worker="answer", reason="완료")])
    monkeypatch.setattr(api, "create_llm", create_llm)
    monkeypatch.setattr(api, "build_graph", lambda model: build_graph(model, lambda *args, **kwargs: [EVIDENCE]))
    for key in ("first-secret", "second-secret"):
        result = client.post("/api/chat", headers={"X-LLM-API-Key": key}, json={"question": "private-question"})
        assert result.status_code == 200
        assert result.json()["status"] == "done"
        assert result.json()["sources"][0]["excerpt"] == EVIDENCE["content"]
        assert result.json()["sources"][0]["excerpt_truncated"] is False
        assert "content" not in result.json()["sources"][0]
        assert key not in result.text
    assert seen == ["first-secret", "second-secret"]
    trace = "".join(path.read_text() for path in (workspace / "traces").glob("*.jsonl"))
    assert "private-question" not in trace
    assert "secret" not in trace
    assert "확인된 해결 절차" not in trace


def test_provider_or_embedding_failure_does_not_echo_exception(client, monkeypatch):
    def fail(*args):
        raise RuntimeError("secret-key sensitive provider payload")
    monkeypatch.setattr(api, "create_llm", fail)
    response = client.post("/api/chat", headers={"X-LLM-API-Key": "test"}, json={"question": "Nanite"})
    assert response.status_code == 503
    assert "secret-key" not in response.text


def test_chat_code_display_preserves_answer_citations_and_trace_privacy(client, workspace, monkeypatch):
    answer = "코드를 확인하세요. [1]\n```cpp\nValues[0] = 1;\n```"
    model = FakeLLM([Route(worker="technique", reason="검색"), Route(worker="answer", reason="완료")], answer)
    monkeypatch.setattr(api, "create_llm", lambda *args: model)
    monkeypatch.setattr(api, "build_graph", lambda llm: build_graph(llm, lambda *args, **kwargs: [EVIDENCE]))
    response = client.post("/api/chat", headers={"X-LLM-API-Key": "provider-secret"}, json={"question": "코드 질문"})

    assert response.status_code == 200
    result = response.json()
    assert result["answer"] == answer
    assert result["status"] == "done"
    assert result["sources"][0]["number"] == 1
    assert result["answer_blocks"] == [
        {"type": "text", "content": "코드를 확인하세요. [1]\n"},
        {"type": "code", "content": "Values[0] = 1;\n", "language": "cpp"},
    ]
    traces = "".join(path.read_text() for path in (workspace / "traces").glob("*.jsonl"))
    assert "Values[0]" not in traces
    assert "코드 질문" not in traces
    assert "provider-secret" not in response.text + traces


def test_document_id_conflict_returns_actionable_error_without_source_content(client, monkeypatch):
    from agents import docs
    from tests.test_document_ids import references

    references(docs.DOCS_DIR)
    model = FakeLLM([Route(worker="technique", reason="검색")])
    monkeypatch.setattr(api, "create_llm", lambda *args: model)

    def unexpected():
        pytest.fail("문서 ID가 중복되면 임베딩 모델을 호출하면 안 됩니다")

    monkeypatch.setattr(docs, "_embeddings", unexpected)
    response = client.post("/api/chat", headers={"X-LLM-API-Key": "provider-secret"},
                           json={"question": "private-question"})

    assert response.status_code == 503
    assert "문서 ID가 중복" in response.json()["detail"]
    assert "파일명" in response.json()["detail"]
    assert model.writer_calls == 0
    for private in ("Guide", "Nanite", "private-question", "provider-secret"):
        assert private not in response.text


def test_history_rejects_system_role_and_limits_messages(client):
    base = {"question": "질문", "history": [{"role": "system", "content": "지시"}]}
    assert client.post("/api/chat", json=base).status_code == 422
    base["history"] = [{"role": "user", "content": "질문"}] * 11
    assert client.post("/api/chat", json=base).status_code == 422


@pytest.mark.parametrize("error_type, expected_status", [
    (api.AuthenticationError, 401), (api.RateLimitError, 429), (api.APIStatusError, 502),
])
def test_provider_errors_have_actionable_redacted_responses(client, monkeypatch, error_type, expected_status):
    response = httpx.Response(expected_status, request=httpx.Request("POST", "https://provider.example"))
    def fail(*args):
        raise error_type("sensitive-provider-payload", response=response, body={"key": "secret"})
    monkeypatch.setattr(api, "create_llm", fail)
    result = client.post("/api/chat", headers={"X-LLM-API-Key": "test"}, json={"question": "Nanite"})
    assert result.status_code == expected_status
    assert "sensitive-provider-payload" not in result.text
    assert "secret" not in result.text


def test_corrupt_knowledge_returns_recoverable_error(client, workspace):
    from agents import knowledge
    knowledge.DOCS_DIR.mkdir()
    (knowledge.DOCS_DIR / "bad.json").write_text("{}", encoding="utf-8")
    assert client.get("/api/knowledge").status_code == 503


def test_knowledge_update_delete_and_conflicts(client):
    payload = {"title": "Nanite", "kind": "technique", "engine_version": "5.5", "content": "팀 기록"}
    original = client.post("/api/knowledge", json=payload).json()["entry"]
    path = f"/api/knowledge/{original['id']}"
    changed = client.put(path, json={**payload, "title": "변경 제목", "expected_revision": 1})
    assert changed.status_code == 200
    assert changed.json()["entry"]["revision"] == 2
    assert client.put(path, json={**payload, "expected_revision": 1}).status_code == 409
    assert client.delete(path, params={"expected_revision": 1}).status_code == 409
    assert client.put(path, json=payload).status_code == 422
    assert client.delete(path).status_code == 422
    assert client.get("/api/knowledge").json()["entries"][0]["title"] == "변경 제목"
    deleted = client.delete(path, params={"expected_revision": 2})
    assert deleted.status_code == 204
    assert deleted.content == b""
    assert client.get("/api/knowledge").json()["entries"] == []
    assert client.put(path, json={**payload, "expected_revision": 2}).status_code == 404
    assert client.delete(path, params={"expected_revision": 2}).status_code == 404


def test_invalid_error_edit_leaves_original_knowledge(client):
    payload = {"title": "Nanite", "kind": "technique", "engine_version": "5.5", "content": "팀 기록"}
    original = client.post("/api/knowledge", json=payload).json()["entry"]
    response = client.put(f"/api/knowledge/{original['id']}",
                          json={**payload, "kind": "error", "expected_revision": 1})
    assert response.status_code == 422
    assert client.get("/api/knowledge").json()["entries"] == [original]


def test_chat_passes_trimmed_version_to_graph_and_returns_effective_filter(client, monkeypatch):
    model = FakeLLM([Route(worker="technique", reason="검색"), Route(worker="answer", reason="완료")])
    calls = []
    def retrieve(query, **options):
        calls.append(options)
        return [EVIDENCE]
    monkeypatch.setattr(api, "create_llm", lambda *args: model)
    monkeypatch.setattr(api, "build_graph", lambda model: build_graph(model, retrieve))
    result = client.post("/api/chat", headers={"X-LLM-API-Key": "test"},
                         json={"question": "shader", "engine_version": " 5.5 "})
    assert result.status_code == 200
    assert result.json()["engine_version"] == "5.5"
    assert calls[0]["engine_version"] == "5.5"


@pytest.mark.parametrize("length", [1001, 8000])
def test_long_question_succeeds_when_supervisor_initially_skips_search(client, workspace, monkeypatch, length):
    question = "오" * (length - 8) + "LOG_END!"
    model = FakeLLM([Route(worker="answer", reason="완료"), Route(worker="answer", reason="검색 완료")])
    calls = []

    def retrieve(query, **options):
        calls.append(query)
        return [EVIDENCE]

    monkeypatch.setattr(api, "create_llm", lambda *args: model)
    monkeypatch.setattr(api, "build_graph", lambda llm: build_graph(llm, retrieve))
    response = client.post("/api/chat", headers={"X-LLM-API-Key": "test"},
                           json={"question": question, "engine_version": "5.5"})

    assert response.status_code == 200
    assert response.json()["status"] == "done"
    assert response.json()["engine_version"] == "5.5"
    assert calls == [question]
    traces = "".join(path.read_text() for path in (workspace / "traces").glob("*.jsonl"))
    assert "LOG_END!" not in traces


def test_question_over_api_limit_is_rejected_before_llm(client, monkeypatch):
    question = "오" * 8001

    def unexpected(*args):
        pytest.fail("질문 길이 검증 실패 후 LLM을 호출하면 안 됩니다")

    monkeypatch.setattr(api, "create_llm", unexpected)
    response = client.post("/api/chat", headers={"X-LLM-API-Key": "test"}, json={"question": question})

    assert response.status_code == 422
    assert "body.question" in response.json()["fields"]
    assert question not in response.text


@pytest.mark.parametrize("version", ["", "  ", "v" * 51])
def test_invalid_version_filter_is_rejected(client, version):
    result = client.post("/api/chat", json={"question": "shader", "engine_version": version})
    assert result.status_code == 422


@pytest.mark.parametrize("authorization", [None, "Bearer wrong-token"])
@pytest.mark.parametrize("method,path,payload", [
    ("GET", "/api/health", None),
    ("GET", "/api/config", None),
    ("POST", "/api/llm/check", {"model": "test-model"}),
    ("GET", "/api/documents", None),
    ("PUT", "/api/documents/note.md", {"title": "Title", "engine_version": "5.5", "expected_revision": 0}),
    ("GET", "/api/knowledge", None),
    ("POST", "/api/knowledge", {"title": "Test", "kind": "technique", "engine_version": "5.5", "content": "note"}),
    ("PUT", "/api/knowledge/" + "a" * 32, {"title": "Test", "kind": "technique", "engine_version": "5.5", "content": "note", "expected_revision": 1}),
    ("DELETE", "/api/knowledge/" + "a" * 32 + "?expected_revision=1", None),
    ("POST", "/api/chat", {"question": "private question"}),
])
def test_internal_access_denial_precedes_file_and_llm_operations(client, monkeypatch, authorization, method, path, payload):
    monkeypatch.setenv("INTERNAL_API_TOKEN", "server-internal-token")
    def forbidden(*args, **kwargs):
        pytest.fail("거부된 요청이 지식 저장소 또는 LLM에 도달했습니다.")
    for name in ("list_entries", "save", "update", "delete"):
        monkeypatch.setattr(api.knowledge, name, forbidden)
    for name in ("list_documents", "update"):
        monkeypatch.setattr(api.documents, name, forbidden)
    monkeypatch.setattr(api, "create_llm", forbidden)
    headers = {"X-LLM-API-Key": "provider-secret"}
    if authorization:
        headers["Authorization"] = authorization
    response = client.request(method, path, headers=headers, json=payload)
    assert response.status_code == 401
    assert response.json()["detail"]["code"] == "internal_access_required"
    assert response.headers["www-authenticate"] == "Bearer"
    assert "server-internal-token" not in response.text
    assert "wrong-token" not in response.text
    assert "provider-secret" not in response.text


@pytest.mark.parametrize("authorization", ["Basic server-internal-token", "Bearer ", "Bearer " + "x" * 4097])
def test_malformed_access_headers_are_rejected(client, monkeypatch, authorization):
    monkeypatch.setenv("INTERNAL_API_TOKEN", "server-internal-token")
    assert client.get("/api/config", headers={"Authorization": authorization}).status_code == 401


def test_valid_access_token_authorizes_knowledge_crud(client, monkeypatch):
    monkeypatch.setenv("INTERNAL_API_TOKEN", "server-internal-token")
    headers = {"Authorization": "Bearer server-internal-token"}
    assert client.get("/api/health", headers=headers).status_code == 200
    assert "server-internal-token" not in client.get("/api/config", headers=headers).text
    payload = {"title": "Private note", "kind": "technique", "engine_version": "5.5", "content": "team knowledge"}
    saved = client.post("/api/knowledge", headers=headers, json=payload)
    assert saved.status_code == 201
    record = saved.json()["entry"]
    path = f"/api/knowledge/{record['id']}"
    assert client.put(path, headers=headers, json={**payload, "expected_revision": 1}).status_code == 200
    assert client.get("/api/knowledge", headers=headers).json()["count"] == 1
    from agents import knowledge
    stored = (knowledge.DOCS_DIR / f"{record['id']}.json").read_text()
    assert "server-internal-token" not in stored
    assert client.delete(path, headers=headers, params={"expected_revision": 2}).status_code == 204


def test_access_token_is_separate_from_llm_key_and_never_reaches_model_or_trace(client, workspace, monkeypatch):
    monkeypatch.setenv("INTERNAL_API_TOKEN", "server-internal-token")
    model = FakeLLM([Route(worker="technique", reason="검색"), Route(worker="answer", reason="완료")])
    received = []
    def create(key, selected_model):
        received.append(key)
        return model
    monkeypatch.setattr(api, "create_llm", create)
    monkeypatch.setattr(api, "build_graph", lambda llm: build_graph(llm, lambda *args, **kwargs: [EVIDENCE]))
    result = client.post("/api/chat", headers={"Authorization": "Bearer server-internal-token",
                         "X-LLM-API-Key": "provider-secret"}, json={"question": "shader"})
    assert result.status_code == 200
    assert received == ["provider-secret"]
    assert "server-internal-token" not in str(model.router_messages + model.writer_messages)
    assert "server-internal-token" not in result.text
    assert "provider-secret" not in result.text
    trace = "".join(path.read_text() for path in (workspace / "traces").glob("*.jsonl"))
    assert "server-internal-token" not in trace
    assert "provider-secret" not in trace


def test_chat_excerpts_only_include_valid_citations_for_the_selected_version(client, monkeypatch):
    selected = {**EVIDENCE, "doc_id": "cited", "page": 4}
    uncited = {**EVIDENCE, "doc_id": "unused", "chunk_id": 1, "content": "UNCITED_SECRET_BODY"}
    other = {**EVIDENCE, "doc_id": "other-version", "engine_version": "5.6", "content": "OTHER_VERSION_BODY"}
    def model(*args):
        return FakeLLM([Route(worker="technique", reason="검색"), Route(worker="answer", reason="완료")])
    monkeypatch.setattr(api, "create_llm", model)
    monkeypatch.setattr(api, "build_graph", lambda llm: build_graph(llm, lambda *args, **kwargs: [selected, uncited, other]))
    response = client.post("/api/chat", headers={"X-LLM-API-Key": "provider-secret"},
                           json={"question": "shader", "engine_version": "5.5"})
    assert response.status_code == 200
    assert len(response.json()["sources"]) == 1
    source = response.json()["sources"][0]
    assert source["excerpt"] == EVIDENCE["content"]
    assert source["page"] == 4
    assert source["engine_version"] == "5.5"
    assert "UNCITED_SECRET_BODY" not in response.text
    assert "OTHER_VERSION_BODY" not in response.text


def test_llm_key_or_query_parameter_cannot_replace_internal_authorization(client, monkeypatch):
    monkeypatch.setenv("INTERNAL_API_TOKEN", "server-internal-token")
    assert client.get("/api/config?token=server-internal-token",
                      headers={"X-LLM-API-Key": "server-internal-token"}).status_code == 401


def test_unset_access_token_preserves_local_development(client):
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/config").status_code == 200


def test_document_metadata_api_conflict_and_original_preservation(client, monkeypatch):
    from urllib.parse import quote
    monkeypatch.setenv("INTERNAL_API_TOKEN", "server-internal-token")
    headers = {"Authorization": "Bearer server-internal-token"}
    api.knowledge.DOCS_DIR.mkdir()
    path = api.knowledge.DOCS_DIR / "한글 + source.md"
    path.write_text("Original reference", encoding="utf-8")
    listed = client.get("/api/documents", headers=headers).json()
    assert listed["count"] == 1
    assert listed["entries"][0]["revision"] == 0
    url = f"/api/documents/{quote(path.name, safe='')}"
    payload = {"title": "변경 제목", "engine_version": "5.5", "expected_revision": 0}
    saved = client.put(url, json=payload, headers=headers)
    assert saved.status_code == 200
    assert saved.json()["entry"]["revision"] == 1
    assert client.put(url, json=payload, headers=headers).status_code == 409
    assert path.read_text() == "Original reference"
    registry = (api.knowledge.DOCS_DIR / "metadata.json").read_text()
    assert "server-internal-token" not in registry
    path.unlink()
    assert client.put(url, json=payload, headers=headers).status_code == 404


def test_document_metadata_api_validation_and_storage_failure(client, monkeypatch):
    api.knowledge.DOCS_DIR.mkdir()
    (api.knowledge.DOCS_DIR / "note.md").write_text("Original reference")
    payload = {"title": "Title", "engine_version": "5.5", "expected_revision": 0}
    assert client.put("/api/documents/note.md", json={**payload, "title": ""}).status_code == 422
    assert client.put("/api/documents/note.json", json=payload).status_code == 404
    registry = api.knowledge.DOCS_DIR / "metadata.json"
    registry.write_text("private broken registry")
    for response in (client.get("/api/documents"), client.put("/api/documents/note.md", json=payload)):
        assert response.status_code == 503
        assert "private broken" not in response.text
    registry.unlink()
    def fail(*args):
        raise OSError("private disk failure")
    monkeypatch.setattr(api.documents.os, "replace", fail)
    response = client.put("/api/documents/note.md", json=payload)
    assert response.status_code == 503
    assert "private disk" not in response.text
    assert not registry.exists()


@pytest.mark.parametrize("method,path,payload", [
    ("GET", "/api/knowledge", None),
    ("POST", "/api/knowledge", {"title": "Test", "kind": "technique", "engine_version": "5.5", "content": "note"}),
    ("PUT", "/api/documents/note.md", {"title": "Title", "engine_version": "5.5", "expected_revision": 0}),
    ("POST", "/api/chat", {"question": "nanite"}),
])
def test_storage_lock_timeout_returns_redacted_service_error(client, monkeypatch, method, path, payload):
    monkeypatch.setattr(api.knowledge, "LOCK_TIMEOUT", 0.05)
    monkeypatch.setattr(api, "create_llm", lambda *args: FakeLLM([Route(worker="technique", reason="검색")]))
    with api.knowledge.locked():
        response = client.request(method, path, headers={"X-LLM-API-Key": "provider-secret"}, json=payload)
    assert response.status_code == 503
    assert ".documents.lock" not in response.text
    assert "provider-secret" not in response.text
    assert str(api.knowledge.DOCS_DIR.parent) not in response.text


def test_corpus_change_rejects_stale_answer_and_retry_uses_latest_sources(client, workspace, monkeypatch):
    from agents import docs
    from tests.test_retrieval import LocalEmbeddings, add
    monkeypatch.setattr(docs, "_embeddings", LocalEmbeddings)
    record = add("nanite")
    def routes():
        yield Route(worker="technique", reason="검색")
        api.knowledge.update(record["id"], api.knowledge.KnowledgeUpdate(title="nanite", kind="technique",
                             engine_version="5.5", content="LATEST_SOURCE_BODY", expected_revision=1))
        yield Route(worker="answer", reason="완료")
    first_model = FakeLLM(routes(), "STALE_RESPONSE_BODY [1]")
    retry_model = FakeLLM([Route(worker="technique", reason="검색"), Route(worker="answer", reason="완료")])
    models = iter([first_model, retry_model])
    monkeypatch.setattr(api, "create_llm", lambda *args: next(models))
    payload = {"question": "nanite", "engine_version": "5.5"}
    headers = {"X-LLM-API-Key": "provider-secret"}
    failed = client.post("/api/chat", json=payload, headers=headers)
    assert failed.status_code == 409
    assert "지식이 변경" in failed.json()["detail"]
    assert "STALE_RESPONSE_BODY" not in failed.text
    assert "provider-secret" not in failed.text
    assert first_model.writer_calls == 0
    assert not list((workspace / "traces").glob("*.jsonl"))
    retried = client.post("/api/chat", json=payload, headers=headers)
    assert retried.status_code == 200
    assert retried.json()["status"] == "done"
    assert "LATEST_SOURCE_BODY" in retried.json()["sources"][0]["excerpt"]
    trace = "".join(path.read_text() for path in (workspace / "traces").glob("*.jsonl"))
    assert "LATEST_SOURCE_BODY" not in trace
    assert "STALE_RESPONSE_BODY" not in trace
    assert "provider-secret" not in trace


@pytest.mark.parametrize("text,status", [
    ("```cpp\nValues[0] = 1;\n```\n확인된 설명 [1]", "done"),
    ("`Values[1]`", "invalid_citations"),
])
def test_code_examples_preserve_answer_and_do_not_supply_citations(client, monkeypatch, text, status):
    monkeypatch.setattr(api, "create_llm", lambda *args: FakeLLM([
        Route(worker="technique", reason="검색"), Route(worker="answer", reason="완료")], text))
    monkeypatch.setattr(api, "build_graph", lambda model: build_graph(model, lambda *args, **kwargs: [EVIDENCE]))
    response = client.post("/api/chat", json={"question": "배열 사용법"}, headers={"X-LLM-API-Key": "provider-secret"})
    assert response.status_code == 200
    assert response.json()["status"] == status
    if status == "done":
        assert response.json()["answer"] == text
        assert [source["number"] for source in response.json()["sources"]] == [1]
    else:
        assert response.json()["answer"] != text
        assert response.json()["sources"] == []
