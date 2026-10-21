"""Optional Tavily source display regression; all API responses are mocked.

Requires Playwright/Chromium and BROWSER_BASE_URL for a nondefault frontend server.
No Tavily or LLM requests are sent.
"""

import json
import os
import tempfile
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def run():
    output = Path(tempfile.gettempdir()) / 'tsukuyomi-web-search-check'
    output.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': 1280, 'height': 900})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))

        def respond(route, result):
            route.fulfill(status=200, content_type='application/json', body=json.dumps(result))

        def api(route):
            path = route.request.url.split('/api/', 1)[1]
            if path == 'config':
                respond(route, {'model': 'test-model', 'has_server_key': True})
            elif path in ('documents', 'knowledge'):
                respond(route, {'entries': []})
            elif path == 'chat':
                assert route.request.post_data_json['engine_version'] == '5.4'
                respond(route, {'status': 'done', 'answer': '공개 문서의 적용 버전을 확인하세요. [1]',
                    'warnings': ['Tavily 웹 검색 사용 한도 안내'], 'sources': [{
                        'number': 1, 'title': 'Unreal Engine public reference',
                        'source_type': 'web', 'engine_version': '미지정',
                        'url': 'https://dev.epicgames.com/documentation/en-us/unreal-engine',
                        'retrieved_at': '2026-10-21T00:00:00+00:00', 'published_date': '2026-01-01',
                        'excerpt': '<script>window.webExecuted = true</script>\nPublic web excerpt',
                        'excerpt_truncated': False,
                    }]})
            else:
                raise AssertionError(path)

        page.route('**/api/**', api)
        page.goto(os.getenv('BROWSER_BASE_URL', 'http://127.0.0.1:8000'), wait_until='networkidle')
        page.locator('.version-trigger').click()
        page.get_by_role('menuitemradio', name='UE 5.4', exact=True).click()
        page.get_by_label('언리얼 기술 질문').fill('공식 문서를 웹에서 찾아줘')
        page.get_by_role('button', name='질문 보내기').click()
        expect(page.locator('.message.assistant')).to_be_visible()
        expect(page.locator('.sources small')).to_have_text('웹 · 버전 미확인')
        expect(page.get_by_role('status')).to_contain_text('Tavily 웹 검색 사용 한도 안내')
        expect(page.locator('.sources a')).to_have_attribute('href', 'https://dev.epicgames.com/documentation/en-us/unreal-engine')
        page.locator('.source-excerpt summary').click()
        expect(page.get_by_text('Tavily가 반환한 웹 문서 발췌입니다.', exact=True)).to_be_visible()
        expect(page.locator('.source-excerpt blockquote')).to_contain_text('<script>')
        assert page.evaluate('window.webExecuted') is None
        page.reload(wait_until='networkidle')
        expect(page.locator('.sources small')).to_have_text('웹 · 버전 미확인')
        expect(page.get_by_role('status')).to_contain_text('Tavily 웹 검색 사용 한도 안내')
        data = page.evaluate('JSON.parse(localStorage.getItem("tsukuyomi.conversations.v1"))')
        source = data['conversations'][0]['messages'][1]['sources'][0]
        assert source['source_type'] == 'web'
        assert source['retrieved_at'] == '2026-10-21T00:00:00+00:00'
        assert source['published_date'] == '2026-01-01'
        for width in (320, 375, 414, 768, 1280):
            page.set_viewport_size({'width': width, 'height': 900})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(output / f'web-source-{width}.png'), full_page=True)
        assert not errors, errors
        print(f'PASS: web source label/link/excerpt/warning, safe text, reload persistence, 5 widths; {output}')
        browser.close()


if __name__ == '__main__':
    run()
