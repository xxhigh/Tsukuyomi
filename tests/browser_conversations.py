"""Optional browser regression for local conversation history. All API calls are mocked.

Requires Playwright/Chromium. Set BROWSER_BASE_URL to the frontend server URL.
No real keys, documents or model calls are used. Screenshots go to the temp folder.
"""

import json
import os
import tempfile
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError, expect, sync_playwright

STORAGE_KEY = 'tsukuyomi.conversations.v1'


def run():
    output = Path(tempfile.gettempdir()) / 'tsukuyomi-conversation-check'
    output.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1280, 'height': 900})
        errors, chats, pending = [], [], []
        access = {'required': False}
        page.on('pageerror', lambda error: errors.append(str(error)))

        def respond(route, data, status=200):
            route.fulfill(status=status, content_type='application/json', body=json.dumps(data))

        def chat_response(route, question):
            respond(route, {'answer': f'{question}의 답변입니다. [1]', 'status': 'done',
                'sources': [{'number': 1, 'title': '팀의 패치 기록', 'engine_version': '5.4.2',
                    'excerpt': f'{question} 당시의 검색 원문', 'excerpt_truncated': False}],
                'answer_blocks': [{'type': 'code', 'language': 'cpp', 'content': 'Values[0] = 1;'}]})

        def api(route):
            path = route.request.url.split('/api/', 1)[1]
            if access['required'] and route.request.headers.get('authorization') != 'Bearer internal-secret':
                respond(route, {'detail': {'code': 'internal_access_required', 'message': '접근 토큰이 필요합니다.'}}, 401)
            elif path == 'config':
                respond(route, {'model': 'test-model', 'has_server_key': False})
            elif path in ('knowledge', 'documents'):
                respond(route, {'entries': []})
            elif path == 'chat':
                payload = route.request.post_data_json
                chats.append(payload)
                assert route.request.headers.get('x-llm-api-key') == 'provider-secret'
                if payload['question'] in ('기다리는 질문', '중단될 질문'):
                    pending.append((route, payload['question']))
                elif payload['question'] == '실패할 질문':
                    respond(route, {'status': 'invalid_citations', 'answer': '출처 확인 실패', 'sources': []})
                else:
                    chat_response(route, payload['question'])
            else:
                raise AssertionError(f'Unexpected API request: {path}')

        page.route('**/api/**', api)
        page.goto(os.getenv('BROWSER_BASE_URL', 'http://127.0.0.1:8000'), wait_until='networkidle')
        nav = page.get_by_role('navigation', name='작업 선택')

        def configure_key():
            nav.get_by_role('button', name='설정').click()
            page.get_by_label('API 키', exact=True).fill('provider-secret')
            nav.get_by_role('button', name='대화').click()

        def choose(version):
            page.locator('.version-trigger').click()
            page.get_by_role('menuitemradio', name=f'UE {version}', exact=True).click()

        def send(question):
            page.get_by_label('언리얼 기술 질문').fill(question)
            page.get_by_role('button', name='질문 보내기').click()

        def finished():
            expect(page.locator('.pending')).to_have_count(0)

        def saved():
            return page.evaluate('(key) => JSON.parse(localStorage.getItem(key))', STORAGE_KEY)

        configure_key()
        choose('5.4')
        send('첫 번째 셰이더 질문')
        expect(page.locator('.message.assistant')).to_have_count(1)
        assert chats[-1]['history'] == []
        page.get_by_label('언리얼 기술 질문').fill('첫 대화의 작성 중 초안')
        page.get_by_role('button', name='새 대화 시작', exact=True).click()
        expect(page.locator('.start-wordmark')).to_be_visible()
        expect(page.get_by_label('언리얼 기술 질문')).to_have_value('')
        choose('5.6')
        send('두 번째 독립 질문')
        expect(page.locator('.message.assistant')).to_have_count(1)
        assert chats[-1]['history'] == []
        assert len(saved()['conversations']) == 2

        first = page.get_by_role('button', name='대화 열기: 첫 번째 셰이더 질문', exact=True)
        first.click()
        expect(page.get_by_label('언리얼 기술 질문')).to_have_value('첫 대화의 작성 중 초안')
        expect(page.locator('.version-trigger')).to_contain_text('5.4')
        expect(page.locator('.message.user')).to_contain_text('첫 번째 셰이더 질문')
        page.locator('.source-excerpt summary').click()
        expect(page.locator('.source-excerpt blockquote')).to_have_text('첫 번째 셰이더 질문 당시의 검색 원문')
        expect(page.locator('.answer-code code')).to_have_text('Values[0] = 1;')
        send('첫 대화의 후속 질문')
        expect(page.locator('.message.assistant')).to_have_count(2)
        assert [item['role'] for item in chats[-1]['history']] == ['user', 'assistant']
        assert chats[-1]['history'][0]['content'] == '첫 번째 셰이더 질문'

        send('기다리는 질문')
        expect(page.locator('.pending')).to_be_visible()
        expect(first).to_be_disabled()
        expect(page.get_by_role('button', name='새 대화 시작', exact=True)).to_be_disabled()
        route, question = pending.pop()
        chat_response(route, question)
        expect(page.locator('.message.assistant')).to_have_count(3)

        send('실패할 질문')
        expect(page.get_by_role('alert')).to_contain_text('출처 확인 실패')
        expect(page.get_by_label('언리얼 기술 질문')).to_have_value('실패할 질문')
        assert next(item for item in saved()['conversations'] if item['id'] == saved()['activeId'])['messages'][-1]['failed']
        raw = page.evaluate('(key) => localStorage.getItem(key)', STORAGE_KEY)
        assert 'provider-secret' not in raw and 'internal-secret' not in raw

        # Access denial hides cached records without overwriting them.
        access['required'] = True
        page.reload(wait_until='networkidle')
        expect(page.get_by_role('heading', name='연결 설정', exact=True)).to_be_visible()
        expect(page.locator('.conversation-history')).to_have_count(0)
        assert len(saved()['conversations']) == 2
        page.get_by_label('접근 토큰', exact=True).fill('internal-secret')
        page.get_by_role('button', name='연결', exact=True).click()
        expect(page.locator('.conversation-link')).to_have_count(2)
        page.get_by_label('API 키', exact=True).fill('provider-secret')
        first.click()
        expect(page.get_by_label('언리얼 기술 질문')).to_have_value('실패할 질문')
        send('실패 후 재개 질문')
        expect(page.locator('.message.assistant')).to_have_count(4)
        assert all(item['content'] != '실패할 질문' for item in chats[-1]['history'])
        assert all('두 번째 독립 질문' not in item['content'] for item in chats[-1]['history'])

        # Keys are not restored; conversation messages and drafts are.
        access['required'] = False
        page.get_by_label('언리얼 기술 질문').fill('새로고침 뒤에도 남을 초안')
        page.reload(wait_until='networkidle')
        expect(page.get_by_label('언리얼 기술 질문')).to_have_value('새로고침 뒤에도 남을 초안')
        expect(page.locator('.message.assistant')).to_have_count(4)
        configure_key()
        for number in range(5):
            count = page.locator('.message.assistant').count()
            send(f'추가 후속 질문 {number}')
            expect(page.locator('.message.assistant')).to_have_count(count + 1)
        assert len(chats[-1]['history']) == 10
        assert all(set(item) == {'role', 'content'} for item in chats[-1]['history'])
        choose('5.6')
        send('첫 대화에서 다른 버전 질문')
        expect(page.locator('.message.assistant')).to_have_count(10)
        assert chats[-1]['history'] == []

        # Reloading during a request restores a retryable question, not a valid history turn.
        page.get_by_role('button', name='새 대화 시작', exact=True).click()
        send('중단될 질문')
        expect(page.locator('.pending')).to_be_visible()
        page.reload(wait_until='networkidle')
        expect(page.get_by_label('언리얼 기술 질문')).to_have_value('중단될 질문')
        expect(page.get_by_role('alert')).to_be_visible()
        assert next(item for item in saved()['conversations'] if item['id'] == saved()['activeId'])['messages'][-1]['failed']
        for interrupted_route, _ in pending:
            try:
                interrupted_route.abort()
            except PlaywrightError:
                pass
        pending.clear()
        configure_key()
        send('중단 후 재시도')
        expect(page.locator('.message.assistant')).to_have_count(1)
        assert chats[-1]['history'] == []

        # Explicit deletion is durable and does not remove the active unrelated thread.
        page.get_by_role('button', name='대화 삭제: 첫 번째 셰이더 질문', exact=True).click()
        page.get_by_role('button', name='삭제 확인', exact=True).click()
        expect(first).to_have_count(0)
        page.reload(wait_until='networkidle')
        expect(first).to_have_count(0)
        expect(page.locator('.conversation-link')).to_have_count(2)
        expect(page.locator('.message.assistant')).to_have_count(1)
        for width in (320, 375, 414, 768, 1280):
            page.set_viewport_size({'width': width, 'height': 900})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(output / f'history-{width}.png'), full_page=True)
        assert not errors, errors
        print(f'PASS: new/resume/reload, drafts/versions/sources, isolated and bounded history, failed/interrupted turns, access, deletion, 5 widths; {output}')
        browser.close()


if __name__ == '__main__':
    run()
