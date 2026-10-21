import json

import httpx
import pytest

from agents import docs, web
from agents.graph import build_graph
from agents.state import Route
from tests.test_graph import EVIDENCE, FakeLLM


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setenv('TAVILY_API_KEY', 'server-tavily-secret')
    state = {'calls': [], 'status': 200, 'payload': {'results': [{
        'title': 'Epic reference', 'url': 'https://dev.epicgames.com/documentation/guide',
        'content': 'Public Unreal reference', 'score': 0.9, 'published_date': '2026-01-01',
    }]}}

    def handle(request):
        state['calls'].append(request)
        if state.get('exception'):
            raise state['exception']
        if 'raw' in state:
            return httpx.Response(state['status'], content=state['raw'])
        return httpx.Response(state['status'], json=state['payload'])

    monkeypatch.setattr(web, '_client', lambda: httpx.Client(transport=httpx.MockTransport(handle)))
    return state


def test_tavily_request_and_public_evidence_conversion(provider):
    found = web.search('Nanite hotfix', engine_version='5.4.2')
    request = provider['calls'][0]
    assert str(request.url) == web.SEARCH_URL
    assert request.headers['authorization'] == 'Bearer server-tavily-secret'
    assert json.loads(request.content) == {
        'query': 'Unreal Engine 5.4.2 Nanite hotfix', 'topic': 'general', 'search_depth': 'basic',
        'max_results': 5, 'include_answer': False, 'include_raw_content': False, 'include_images': False,
    }
    assert len(found) == 1
    assert found[0]['source_type'] == found[0]['kind'] == 'web'
    assert found[0]['engine_version'] == '미지정'
    assert found[0]['url'].startswith('https://dev.epicgames.com/')
    assert found[0]['retrieved_at'].endswith('+00:00')
    assert found[0]['published_date'] == '2026-01-01'
    assert 'server-tavily-secret' not in str(found)


def test_missing_key_and_empty_or_oversized_queries_never_send_http(provider, monkeypatch):
    for query in ('', '  ', 'x' * 1001, None):
        with pytest.raises(web.WebSearchError, match='공개 웹 검색어'):
            web.search(query)
    monkeypatch.delenv('TAVILY_API_KEY')
    assert not web.available()
    with pytest.raises(web.WebSearchError, match='설정되지 않아'):
        web.search('Nanite')
    assert provider['calls'] == []


@pytest.mark.parametrize('url', ['javascript:alert(1)', 'file:///etc/passwd', '/relative',
    'https://name:password@example.com/x', 'https://example.com:bad/x', 'https://example.com/\npath',
    'https://[broken/', 'https://example.com/' + 'x' * 4096])
def test_invalid_source_urls_are_discarded(provider, url):
    provider['payload']['results'][0]['url'] = url
    assert web.search('Nanite') == []


def test_result_limits_deduplication_and_content_bound(provider):
    first = provider['payload']['results'][0]
    provider['payload']['results'] = [None, {'url': 'https://example.com/empty', 'content': ''},
        {**first, 'content': 'a' * (web.MAX_CONTENT + 10)}, {**first, 'url': first['url'] + '#section'},
        *[{**first, 'url': f'https://example.com/{index}'} for index in range(10)]]
    found = web.search('Nanite')
    assert len(found) == web.MAX_RESULTS
    assert len({item['doc_id'] for item in found}) == web.MAX_RESULTS
    assert len(found[0]['content']) == web.MAX_CONTENT


@pytest.mark.parametrize('status,code', [(401, 'unauthorized'), (403, 'unauthorized'),
    (429, 'rate_limited'), (432, 'rate_limited'), (433, 'rate_limited'), (500, 'unavailable'), (302, 'unavailable')])
def test_http_errors_are_redacted_and_not_retried(provider, status, code):
    provider.update(status=status, payload={'detail': 'server-tavily-secret PRIVATE_PROVIDER_BODY'})
    with pytest.raises(web.WebSearchError) as error:
        web.search('PRIVATE_QUERY')
    assert error.value.code == code
    assert 'secret' not in str(error.value)
    assert 'PRIVATE' not in str(error.value)
    assert len(provider['calls']) == 1


@pytest.mark.parametrize('error', [httpx.ConnectError('server-tavily-secret'), httpx.ReadTimeout('PRIVATE_QUERY')])
def test_transport_failures_are_redacted(provider, error):
    provider['exception'] = error
    with pytest.raises(web.WebSearchError) as failure:
        web.search('Nanite')
    assert failure.value.code == 'unavailable'
    assert 'secret' not in str(failure.value)
    assert 'PRIVATE' not in str(failure.value)


@pytest.mark.parametrize('raw', [b'not-json server-tavily-secret', b'[]', b'{}', b'{"results":{}}'])
def test_malformed_responses_are_redacted(provider, raw):
    provider['raw'] = raw
    with pytest.raises(web.WebSearchError) as error:
        web.search('Nanite')
    assert error.value.code == 'invalid_response'
    assert 'secret' not in str(error.value)


def test_web_only_graph_does_not_load_or_fingerprint_local_corpus(provider, monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail('Web-only search must not access local models, index or corpus')
    monkeypatch.setattr(docs, '_fingerprint', fail)
    monkeypatch.setattr(docs, '_embeddings', fail)
    monkeypatch.setattr(docs, 'retrieve', fail)
    model = FakeLLM([Route(worker='web_search', query='Nanite release notes', reason='공개 최신 자료'),
                     Route(worker='answer', reason='완료')])
    result = build_graph(model).invoke({'question': '공식 문서를 웹에서 찾아줘', 'engine_version': '5.4'})
    assert result['status'] == 'done'
    assert result['rounds'] == 1
    assert result['sources'][0]['source_type'] == 'web'
    assert result['sources'][0]['engine_version'] == '미지정'
    assert 'corpus_sha256' not in result
    assert 'server-tavily-secret' not in str(model.router_messages + model.writer_messages + [result])
    assert json.loads(model.router_messages[0][-1].content)['web_search_available'] is True


@pytest.mark.parametrize('web_first', [False, True])
def test_web_and_local_sources_survive_version_filter_and_share_citation_numbers(provider, web_first):
    local = Route(worker='technique', query='local search', reason='팀 기록')
    external = Route(worker='web_search', query='Nanite hotfix', reason='공개 문서')
    model = FakeLLM([external, local] if web_first else [local, external], '근거를 확인하세요. [1][2]')
    seen = []

    def retrieve(query, **kwargs):
        seen.append(kwargs)
        return [EVIDENCE, {**EVIDENCE, 'doc_id': 'other', 'engine_version': '5.6', 'content': 'WRONG_MINOR'}]

    result = build_graph(model, retrieve).invoke({'question': 'PRIVATE_QUESTION', 'engine_version': '5.5',
        'history': [{'role': 'user', 'content': 'PRIVATE_HISTORY'}]})
    assert result['status'] == 'done'
    assert result['rounds'] == 2
    assert result['decisions'][-1]['worker'] == 'answer'
    assert {source['engine_version'] for source in result['sources']} == {'5.5', '미지정'}
    assert [source['number'] for source in result['sources']] == [1, 2]
    assert seen == [{'k': 5, 'kind': 'technique', 'engine_version': '5.5'}]
    assert 'WRONG_MINOR' not in str(model.writer_messages)
    request = json.loads(provider['calls'][0].content)
    assert request['query'] == 'Unreal Engine 5.5 Nanite hotfix'
    assert 'PRIVATE_' not in str(request)
    assert EVIDENCE['content'] not in str(request)


def test_unconfigured_web_route_falls_back_to_local_without_http(monkeypatch):
    def fail():
        pytest.fail('No HTTP client may be created without a key')
    monkeypatch.setattr(web, '_client', fail)
    model = FakeLLM([Route(worker='web_search', query='Nanite', reason='검색'), Route(worker='answer', reason='완료')])
    result = build_graph(model, lambda *args, **kwargs: [EVIDENCE]).invoke({'question': 'Nanite'})
    assert result['status'] == 'done'
    assert result['decisions'][0]['worker'] == 'technique'
    assert len(result['warnings']) == 1


def test_web_failure_preserves_local_answer_and_exposes_safe_warning(provider):
    provider['status'] = 429
    model = FakeLLM([Route(worker='technique', reason='검색'), Route(worker='web_search', query='Nanite', reason='보완')])
    result = build_graph(model, lambda *args, **kwargs: [EVIDENCE]).invoke({'question': 'Nanite'})
    assert result['status'] == 'done'
    assert len(result['sources']) == 1
    assert '사용 한도' in result['warnings'][0]
    assert len(provider['calls']) == 1
    assert result['rounds'] == 2


@pytest.mark.parametrize('failure', [False, True])
def test_empty_or_failed_web_search_does_not_invent_evidence(provider, failure):
    provider['payload'] = {'results': []}
    if failure:
        provider['status'] = 500
    model = FakeLLM([Route(worker='web_search', query='Nanite', reason='검색'), Route(worker='answer', reason='완료')])
    result = build_graph(model).invoke({'question': 'Nanite'})
    assert result['status'] == 'insufficient'
    assert result['sources'] == []
    assert model.writer_calls == 0
    assert bool(result['warnings']) is failure


def test_two_web_searches_are_bounded_and_deduplicate_urls(provider):
    model = FakeLLM([Route(worker='web_search', query='Nanite', reason='검색')] * 2)
    result = build_graph(model).invoke({'question': 'Nanite'})
    assert result['rounds'] == 2
    assert len(provider['calls']) == 2
    assert len(result['sources']) == 1
    assert result['decisions'][-1]['worker'] == 'answer'


def test_web_sources_still_require_valid_citations(provider):
    model = FakeLLM([Route(worker='web_search', query='Nanite', reason='검색'), Route(worker='answer', reason='완료')],
                    '근거를 확인했습니다. [99]')
    result = build_graph(model).invoke({'question': 'Nanite'})
    assert result['status'] == 'invalid_citations'
    assert result['sources'] == []


def test_api_returns_web_sources_and_safe_warnings_without_leaking_keys(provider, workspace, monkeypatch):
    import api
    from fastapi.testclient import TestClient

    monkeypatch.delenv('INTERNAL_API_TOKEN', raising=False)
    monkeypatch.setattr(api.trace, 'TRACE_DIR', workspace / 'traces')
    seen = []

    def create(key, model):
        seen.append(key)
        return FakeLLM([Route(worker='web_search', query='Nanite docs', reason='웹 검색'),
                        Route(worker='answer', reason='완료')])

    monkeypatch.setattr(api, 'create_llm', create)
    client = TestClient(api.app)
    response = client.post('/api/chat', headers={'X-LLM-API-Key': 'provider-llm-secret'},
                           json={'question': 'PRIVATE_QUESTION', 'engine_version': '5.4'})
    assert response.status_code == 200
    assert response.json()['sources'][0]['source_type'] == 'web'
    assert response.json()['warnings'] == []
    provider['status'] = 401
    denied = client.post('/api/chat', headers={'X-LLM-API-Key': 'provider-llm-secret'}, json={'question': 'PRIVATE_QUESTION'})
    assert denied.status_code == 200
    assert denied.json()['status'] == 'insufficient'
    assert 'Tavily API 키가 거부' in denied.json()['warnings'][0]
    assert seen == ['provider-llm-secret'] * 2
    traces = ''.join(path.read_text() for path in (workspace / 'traces').glob('*.jsonl'))
    for secret in ('server-tavily-secret', 'provider-llm-secret', 'PRIVATE_QUESTION'):
        assert secret not in response.text + denied.text + traces
    assert 'Public Unreal reference' not in traces
    assert 'server-tavily-secret' not in client.get('/api/config').text


@pytest.mark.parametrize('status', [200, 429])
def test_cli_displays_web_url_or_search_failure_without_secrets(provider, monkeypatch, capsys, status):
    import app
    import sys

    provider['status'] = status
    monkeypatch.setenv('OPENAI_API_KEY', 'provider-llm-secret')
    monkeypatch.setattr(sys, 'argv', ['app.py', '웹 검색'])
    monkeypatch.setattr(app, 'create_llm', lambda *args: FakeLLM([
        Route(worker='web_search', query='Nanite docs', reason='웹 검색'), Route(worker='answer', reason='완료')]))
    assert app.main() == 0
    output = capsys.readouterr().out
    if status == 200:
        assert '웹 검색 · 버전 미확인' in output
        assert 'https://dev.epicgames.com/documentation/guide' in output
    else:
        assert '안내: Tavily 웹 검색 사용 한도' in output
    assert 'server-tavily-secret' not in output
    assert 'provider-llm-secret' not in output
