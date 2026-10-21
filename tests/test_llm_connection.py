import json

import httpx
import pytest
from fastapi.testclient import TestClient

import api
from agents import docs, llm


@pytest.fixture
def client(workspace, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("INTERNAL_API_TOKEN", raising=False)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://provider.example/v1")
    monkeypatch.setenv("OPENAI_MODEL", "server-model")
    monkeypatch.setattr(api.trace, "TRACE_DIR", workspace / "traces")
    return TestClient(api.app)


@pytest.fixture
def provider(monkeypatch):
    state = {"requests": [], "status": 200, "arguments": json.dumps({"worker": "answer", "query": "", "reason": "PRIVATE_PROVIDER_REPLY"}),
             "tool": "Route", "plain_text": False, "connection_error": False}

    def respond(request):
        body = json.loads(request.content)
        state["requests"].append({"headers": dict(request.headers), "url": str(request.url), "body": body})
        if state["connection_error"]:
            raise httpx.ConnectError("PRIVATE_NETWORK_DETAILS", request=request)
        if state["status"] != 200:
            return httpx.Response(state["status"], json={"error": {"message": "PRIVATE_PROVIDER_ERROR", "type": "test_error"}})
        message = {"role": "assistant", "content": "PRIVATE_PROVIDER_REPLY" if state["plain_text"] else None}
        if not state["plain_text"]:
            message["tool_calls"] = [{"id": "test-call", "type": "function",
                                      "function": {"name": state["tool"], "arguments": state["arguments"]}}]
        return httpx.Response(200, json={"id": "test-response", "object": "chat.completion", "created": 0,
                                        "model": body["model"], "choices": [{"index": 0, "finish_reason": "stop" if state["plain_text"] else "tool_calls",
                                        "message": message}]})

    # SDK·structured output 파서는 실제로 실행하고 네트워크만 대체한다.
    factory = llm.ChatOpenAI
    with httpx.Client(transport=httpx.MockTransport(respond)) as transport:
        monkeypatch.setattr(llm, "ChatOpenAI", lambda **options: factory(**options, http_client=transport))
        yield state


def test_check_uses_real_sdk_with_route_schema_and_isolated_keys(client, provider, workspace, monkeypatch):
    monkeypatch.setenv("INTERNAL_API_TOKEN", "internal-secret")
    def forbidden(*args, **kwargs):
        pytest.fail("연결 확인이 RAG·지식 파일·트레이스에 접근했습니다.")
    monkeypatch.setattr(api, "build_graph", forbidden)
    monkeypatch.setattr(docs, "_embeddings", forbidden)
    monkeypatch.setattr(api.knowledge, "list_entries", forbidden)
    monkeypatch.setattr(api.documents, "list_documents", forbidden)
    monkeypatch.setattr(api.trace, "log", forbidden)
    for key in ("first-provider-key", "second-provider-key"):
        response = client.post("/api/llm/check", headers={"Authorization": "Bearer internal-secret", "X-LLM-API-Key": key},
                               json={"model": "  selected-model  "})
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
        assert "PRIVATE_PROVIDER_REPLY" not in response.text
        assert key not in response.text
    assert len(provider["requests"]) == 2
    for request, key in zip(provider["requests"], ("first-provider-key", "second-provider-key")):
        assert request["headers"]["authorization"] == f"Bearer {key}"
        assert "internal-secret" not in str(request)
        assert "x-llm-api-key" not in request["headers"]
        assert request["url"] == "https://provider.example/v1/chat/completions"
        assert request["body"]["model"] == "selected-model"
        assert request["body"]["tool_choice"] == {"type": "function", "function": {"name": "Route"}}
        assert request["body"]["tools"][0]["function"]["parameters"]["properties"]["worker"]["enum"] == ["technique", "troubleshooting", "answer"]
    assert provider["requests"][0]["body"]["messages"] == provider["requests"][1]["body"]["messages"]
    assert list(workspace.rglob("*")) == []


def test_server_key_and_model_defaults_and_request_key_override(client, provider, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "server-key")
    assert client.post("/api/llm/check", json={}).status_code == 200
    first = provider["requests"][0]
    assert first["headers"]["authorization"] == "Bearer server-key"
    assert first["body"]["model"] == "server-model"
    assert client.post("/api/llm/check", headers={"X-LLM-API-Key": "request-key"}, json={"model": None}).status_code == 200
    assert provider["requests"][1]["headers"]["authorization"] == "Bearer request-key"


@pytest.mark.parametrize("settings", [
    {"plain_text": True}, {"tool": "UnknownTool"}, {"arguments": "not json"},
    {"arguments": json.dumps({"worker": "unsupported", "reason": "PRIVATE_PROVIDER_REPLY"})},
    {"arguments": json.dumps({"worker": "answer"})},
])
def test_invalid_tool_responses_return_redacted_format_errors(client, provider, settings):
    provider.update(settings)
    response = client.post("/api/llm/check", headers={"X-LLM-API-Key": "provider-secret"}, json={})
    assert response.status_code == 502
    assert "tool calling" in response.json()["detail"]
    assert "PRIVATE_PROVIDER_REPLY" not in response.text
    assert "provider-secret" not in response.text


@pytest.mark.parametrize("status,expected", [(400, 502), (401, 401), (429, 429), (500, 502)])
def test_provider_http_errors_are_actionable_and_redacted(client, provider, status, expected):
    provider["status"] = status
    response = client.post("/api/llm/check", headers={"X-LLM-API-Key": "provider-secret"}, json={})
    assert response.status_code == expected
    assert isinstance(response.json()["detail"], str)
    assert "PRIVATE_PROVIDER_ERROR" not in response.text
    assert "provider-secret" not in response.text


def test_provider_connection_failure_is_redacted(client, provider):
    provider["connection_error"] = True
    response = client.post("/api/llm/check", headers={"X-LLM-API-Key": "provider-secret"}, json={})
    assert response.status_code == 502
    assert "PRIVATE_NETWORK_DETAILS" not in response.text


@pytest.mark.parametrize("body", [{"model": " "}, {"model": "m" * 101}, {"api_key": "body-secret"},
                                 {"question": "private-question"}, {"base_url": "https://untrusted.example"}])
def test_invalid_check_payload_does_not_reach_provider(client, provider, body):
    response = client.post("/api/llm/check", headers={"X-LLM-API-Key": "provider-secret"}, json=body)
    assert response.status_code == 422
    assert "body-secret" not in response.text
    assert "private-question" not in response.text
    assert provider["requests"] == []


@pytest.mark.parametrize("key", [None, " ", "k" * 4097])
def test_missing_or_invalid_key_does_not_reach_provider(client, provider, key):
    headers = {"X-LLM-API-Key": key} if key is not None else {}
    assert client.post("/api/llm/check", headers=headers, json={}).status_code == 400
    assert provider["requests"] == []


def test_unexpected_failure_is_redacted(client, monkeypatch):
    def fail(*args):
        raise RuntimeError("PRIVATE_SDK_DETAILS")
    monkeypatch.setattr(api, "create_llm", fail)
    response = client.post("/api/llm/check", headers={"X-LLM-API-Key": "provider-secret"}, json={})
    assert response.status_code == 503
    assert "PRIVATE_SDK_DETAILS" not in response.text
