"""Optional: run a built local server, then python tests/browser_smoke.py.

LLM/knowledge HTTP responses are mocked; no keys or synthetic records are persisted.
Requires Playwright with Chromium installed. Screenshots go to the system temp folder.
"""

import json
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from playwright.sync_api import expect, sync_playwright


def run():
    base_url = os.getenv("BROWSER_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    screenshots = Path(tempfile.gettempdir()) / "tsukuyomi-browser-check"
    screenshots.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 800})

        def expect_idle_chat():
            workspace = page.locator('#workspace')
            expect(page.locator('.chat-workspace')).to_have_class(re.compile(r'\bis-empty\b'))
            expect(workspace.locator('h1:visible, h2:visible, .workspace-header, .setup-notice, .footnote, .suggestions')).to_have_count(0)
            assert workspace.inner_text().replace('언리얼 기술 질문', '').strip() == '↑'
            area = workspace.bounding_box()
            composer = page.locator('.composer').bounding_box()
            assert abs(composer['x'] + composer['width'] / 2 - area['x'] - area['width'] / 2) < 2
            assert abs(composer['y'] + composer['height'] / 2 - area['y'] - area['height'] / 2) < 2
            expect(page.get_by_label('언리얼 기술 질문')).to_be_visible()

        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        api_requests = []
        page.on("request", lambda request: api_requests.append(request.url) if "/api/" in request.url else None)
        records, chats, edits, deletions = [], [], [], []
        reference = {"filename": "GAS 팀 참고 + source.md", "doc_id": "GAS 팀 참고 + source", "title": "GAS reference",
                     "engine_version": "미지정", "source_url": None, "revision": 0}
        metadata_edits = []
        checks, pending_checks = [], []
        connection_config = {"model": "test-model", "has_server_key": False}
        source_excerpt = ('BROWSER_SOURCE_MARKER\n<script>window.sourceExecuted = true</script>\n'
                          '<img src="x" onerror="window.sourceExecuted = true">\n' + 'LongUnbrokenSourceToken' * 50)[:1000]
        answer_code = ('    const auto Value = Values[0]; // BROWSER_CODE_MARKER\n'
                       '<script>window.answerExecuted = true</script>\n'
                       '<img src="x" onerror="window.answerExecuted = true">\n' + 'LongCodeIdentifier' * 80 + '\n')
        code_intro = '팀 기록의 해결 절차입니다. [1]\n\n'
        raw_code_answer = code_intro + '```cpp\n' + answer_code + '```'

        def respond(route, data, status=200):
            route.fulfill(status=status, content_type="application/json", body=json.dumps(data))

        def knowledge(route):
            if route.request.method == "GET":
                respond(route, {"entries": records, "count": len(records)})
            elif route.request.method == "POST":
                record = {**route.request.post_data_json, "id": "a" * 32, "created_at": "2026-10-20", "revision": 1}
                records.append(record)
                respond(route, {"entry": record, "message": "저장했습니다. 다음 질문에서 검색 인덱스에 반영합니다."}, 201)
            elif route.request.method == "PUT":
                payload = route.request.post_data_json
                edits.append(payload)
                if len(edits) == 1:
                    records[0].update(title="Edited elsewhere", revision=2)
                    respond(route, {"detail": "다른 작업에서 지식이 변경되었습니다. 목록을 새로고침한 뒤 다시 편집해 주세요."}, 409)
                else:
                    assert payload.pop("expected_revision") == 2
                    records[0].update(**payload, revision=3)
                    respond(route, {"entry": records[0], "message": "수정했습니다. 다음 질문에서 검색 인덱스에 반영합니다."})
            elif route.request.method == "DELETE":
                deletions.append(route.request.url)
                assert parse_qs(urlsplit(route.request.url).query)["expected_revision"] == ["3"]
                if len(deletions) == 1:
                    respond(route, {"detail": "테스트 삭제 실패. 다시 요청해 주세요."}, 503)
                else:
                    records.clear()
                    route.fulfill(status=204)

        def chat(route):
            chats.append(route.request.post_data_json)
            assert route.request.headers["x-llm-api-key"] == "test-key"
            assert "BROWSER_SOURCE_MARKER" not in route.request.post_data
            assert all(set(message) == {'role', 'content'} for message in chats[-1]['history'])
            if len(chats) == 2:
                assert chats[-1]['history'][-1]['content'] == raw_code_answer
                respond(route, {"detail": "테스트 연결 실패. 다시 요청해 주세요."}, 502)
            elif len(chats) == 5:
                respond(route, {"detail": "답변을 만들던 중 지식이 변경되었습니다. 같은 질문을 다시 보내 주세요."}, 409)
            elif len(chats) == 7:
                respond(route, {"answer": "출처 검증에 실패했습니다. 질문을 구체화해 다시 요청해 주세요.",
                                "status": "invalid_citations", "sources": []})
            elif len(chats) == 9:
                respond(route, {"answer": "BROWSER_INSUFFICIENT_MARKER: 등록된 근거가 부족합니다.",
                                "status": "insufficient", "sources": []})
            else:
                source = {"number": 1, "title": "Shader test note", "engine_version": chats[-1]["engine_version"] or "5.5",
                          "url": "javascript:alert(1)", "page": 3}
                if len(chats) != 3:  # 원문이 없는 이전 형식의 응답도 표시한다.
                    source.update(excerpt=source_excerpt if len(chats) == 1 else "짧은 검색 원문",
                                  excerpt_truncated=len(chats) == 1)
                if len(chats) == 6:
                    source["excerpt"] = "UPDATED_BROWSER_SOURCE_MARKER"
                result = {"answer": "팀 기록의 해결 절차입니다. [1]", "status": "done", "sources": [source]}
                if len(chats) == 1:
                    result.update(answer=raw_code_answer, answer_blocks=[
                        {"type": "text", "content": code_intro},
                        {"type": "code", "content": answer_code, "language": "cpp"},
                    ])
                respond(route, result)

        def documents(route):
            if route.request.method == "GET":
                respond(route, {"entries": [reference], "count": 1})
            else:
                payload = route.request.post_data_json
                metadata_edits.append(payload)
                if len(metadata_edits) == 1:
                    reference.update(title="Changed reference", revision=1)
                    respond(route, {"detail": "다른 작업에서 문서 정보가 변경되었습니다. 목록을 새로고침한 뒤 다시 편집해 주세요."}, 409)
                elif len(metadata_edits) == 2:
                    respond(route, {"detail": "문서 정보 저장 실패. 다시 요청해 주세요."}, 503)
                else:
                    assert payload.pop("expected_revision") == 1
                    reference.update(**payload, revision=2)
                    respond(route, {"entry": reference, "message": "문서 정보를 저장했습니다. 다음 질문에서 검색에 반영합니다."})

        def llm_check(route):
            checks.append({"body": route.request.post_data_json, "headers": route.request.headers})
            assert route.request.post_data_json == {"model": "test-model"}
            if not connection_config["has_server_key"]:
                assert route.request.headers["x-llm-api-key"] == "test-key"
            pending_checks.append(route)

        page.route("**/api/config", lambda route: respond(route, connection_config))
        page.route("**/api/llm/check", llm_check)
        page.route("**/api/knowledge**", knowledge)
        page.route("**/api/chat", chat)
        page.route("**/api/documents**", documents)
        page.goto(base_url + "/", wait_until="networkidle")
        expect_idle_chat()
        expect(page.get_by_role("button", name="질문 보내기")).to_be_disabled()
        expect(page.locator('.sidebar input')).to_have_count(1)
        expect(page.locator('.sidebar .connection, .sidebar .access-settings')).to_have_count(0)
        page.get_by_role('navigation').get_by_role('button', name='설정', exact=False).click()
        expect(page.get_by_role('heading', name='연결 설정', exact=True)).to_be_visible()
        expect(page.get_by_role("button", name="LLM 연결 확인", exact=True)).to_be_disabled()
        assert checks == []
        page.get_by_label("API 키", exact=True).fill("test-key")
        checker = page.locator('.llm-check')
        checker.locator('button').click()
        expect(checker.locator('button')).to_contain_text('확인 중')
        expect(checker.locator('button')).to_be_disabled()
        expect(page.get_by_label('API 키', exact=True)).to_be_disabled()
        expect(page.get_by_label('모델', exact=True)).to_be_disabled()
        expect(page.get_by_role('navigation').get_by_role('button', name='대화', exact=False)).to_be_disabled()
        expect(page.get_by_role('button', name='연결', exact=True)).to_be_disabled()
        respond(pending_checks.pop(), {"status": "ok", "message": "LLM 키·모델과 tool calling 응답을 확인했습니다."})
        expect(checker.get_by_role('status')).to_contain_text('확인했습니다')
        assert not chats and not records
        page.get_by_label('모델', exact=True).fill('changed-model')
        expect(checker.get_by_role('status')).to_have_count(0)
        page.get_by_label('모델', exact=True).fill('test-model')
        checker.locator('button').click()
        expect(checker.locator('button')).to_contain_text('확인 중')
        respond(pending_checks.pop(), {"detail": "모델의 tool calling 응답 형식을 확인해 주세요."}, 502)
        expect(checker.get_by_role('alert')).to_contain_text('tool calling')
        page.get_by_label('API 키', exact=True).fill('changed-key')
        expect(checker.get_by_role('alert')).to_have_count(0)
        page.get_by_label('API 키', exact=True).fill('test-key')
        checker.locator('button').click()
        expect(checker.locator('button')).to_contain_text('확인 중')
        respond(pending_checks.pop(), {"status": "ok", "message": "LLM 연결을 확인했습니다."})
        expect(checker.get_by_role('status')).to_contain_text('확인했습니다')
        page.get_by_role('navigation').get_by_role('button', name='대화', exact=False).click()
        page.get_by_label('언리얼 기술 질문').fill('메뉴 이동 후에도 유지할 질문')
        page.get_by_role('navigation').get_by_role('button', name='설정', exact=False).click()
        expect(page.get_by_label('API 키', exact=True)).to_have_value('test-key')
        expect(page.get_by_label('모델', exact=True)).to_have_value('test-model')
        expect(checker.get_by_role('status')).to_contain_text('확인했습니다')
        for width in (320, 375, 414, 768, 1280):
            page.set_viewport_size({'width': width, 'height': 900})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), f'settings overflow at {width}'
            expect(page.get_by_label('API 키', exact=True)).to_be_visible()
        page.screenshot(path=str(screenshots / 'settings-desktop.png'), full_page=True)
        page.set_viewport_size({'width': 375, 'height': 900})
        page.screenshot(path=str(screenshots / 'settings-mobile.png'), full_page=True)
        page.set_viewport_size({'width': 1280, 'height': 800})
        page.get_by_role('navigation').get_by_role('button', name='대화', exact=False).click()
        expect(page.get_by_label('언리얼 기술 질문')).to_have_value('메뉴 이동 후에도 유지할 질문')
        page.reload(wait_until='networkidle')
        page.get_by_role('navigation').get_by_role('button', name='설정', exact=False).click()
        expect(checker.get_by_role('status')).to_have_count(0)
        expect(page.get_by_label('API 키', exact=True)).to_have_value('')
        assert len(checks) == 3
        page.get_by_label('API 키', exact=True).fill('test-key')
        page.get_by_role('navigation').get_by_role('button', name='대화', exact=False).click()
        page.get_by_label("검색 엔진 버전", exact=True).fill("5.5")
        for width in (320, 375, 414, 768, 1280):
            page.set_viewport_size({"width": width, "height": 900})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"chat overflow at {width}"
            expect_idle_chat()
            composer = page.locator('.composer').bounding_box()
            assert composer['y'] + composer['height'] <= 900
        page.set_viewport_size({"width": 1280, "height": 800})
        assert page.get_by_role("button", name="질문 보내기").bounding_box()["y"] < 760
        page.screenshot(path=str(screenshots / "chat-desktop.png"), full_page=True)
        page.set_viewport_size({'width': 375, 'height': 900})
        page.screenshot(path=str(screenshots / 'chat-idle-mobile.png'), full_page=True)
        page.set_viewport_size({'width': 1280, 'height': 800})
        page.get_by_role("button", name=re.compile("지식 저장소")).click()
        page.get_by_label("기록 종류").select_option("error")
        page.get_by_label("제목", exact=True).fill("Shader test note")
        page.get_by_label("엔진 버전", exact=True).fill("5.5")
        page.get_by_label("오류 메시지 / 증상").fill("Test failure")
        page.get_by_label("재현 방법", exact=True).fill("Test reproduction")
        page.get_by_label("해결 방법 / 확인 결과").fill("Test resolution")
        page.get_by_role("button", name="지식 저장", exact=True).click()
        expect(page.get_by_role("status")).to_contain_text("저장했습니다")
        expect(page.locator(".entry summary")).to_contain_text("Shader test note")
        assert records[0]["solution"] == "Test resolution"
        records.append({"id": "c" * 32, "title": "Nanite pilot", "kind": "technique", "engine_version": "5.6",
                        "context": "전용 검증 환경", "content": "한글 본문 검색", "reproduction": "", "solution": "",
                        "revision": 1, "created_at": "2026-10-20"})
        page.get_by_role("button", name="새로고침", exact=True).click()
        expect(page.locator('.entry')).to_have_count(2)
        search = page.get_by_label('등록된 기록 검색', exact=True)
        listing = page.locator('.entry-list')
        expect(search).to_be_enabled()
        requests_before_filtering = len(api_requests)
        for query, title in [('SHADER', 'Shader test note'), ('5.6', 'Nanite pilot'),
                             ('검증 환경', 'Nanite pilot'), ('한글 본문', 'Nanite pilot'),
                             ('reproduction', 'Shader test note'), ('  tEsT rEsOlUtIoN ', 'Shader test note')]:
            search.fill(query)
            expect(page.locator('.entry')).to_have_count(1)
            expect(page.locator('.entry summary')).to_contain_text(title)
            expect(listing.get_by_role('status')).to_have_text('검색 결과 1개 / 전체 2개')
        search.fill('.*')
        expect(page.locator('.entry')).to_have_count(0)
        expect(listing).to_contain_text('검색어와 일치하는 기록이 없습니다.')
        expect(listing.get_by_role('status')).to_have_text('검색 결과 0개 / 전체 2개')
        search.fill('   ')
        expect(page.locator('.entry')).to_have_count(2)
        assert len(api_requests) == requests_before_filtering
        search.fill('Nanite')
        page.get_by_role('button', name='새로고침', exact=True).click()
        expect(search).to_be_enabled()
        expect(search).to_have_value('Nanite')
        expect(page.locator('.entry')).to_have_count(1)
        expect(page.locator('.entry summary')).to_contain_text('Nanite pilot')
        for width in (320, 375, 414, 768, 1280):
            page.set_viewport_size({'width': width, 'height': 900})
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth'), f'entry search overflow at {width}'
        listing.screenshot(path=str(screenshots / 'entry-search-desktop.png'))
        page.set_viewport_size({'width': 375, 'height': 900})
        listing.screenshot(path=str(screenshots / 'entry-search-mobile.png'))
        page.get_by_role('button', name='검색 초기화', exact=True).click()
        expect(search).to_have_value('')
        expect(page.locator('.entry')).to_have_count(2)
        records.pop()
        page.get_by_role('button', name='새로고침', exact=True).click()
        expect(page.locator('.entry')).to_have_count(1)
        for width in (320, 375, 414, 768):
            page.set_viewport_size({"width": width, "height": 900})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"form overflow at {width}"
        page.set_viewport_size({"width": 375, "height": 900})
        page.evaluate("scrollTo(0, 0)")
        page.screenshot(path=str(screenshots / "knowledge-mobile.png"), full_page=True)
        page.set_viewport_size({"width": 1280, "height": 800})
        page.locator(".entry summary").click()
        page.get_by_role("button", name="수정", exact=True).click()
        page.get_by_label("제목", exact=True).fill("Unsent draft")
        page.get_by_role("button", name="편집 취소").click()
        assert not edits
        expect(page.get_by_role("heading", name="지식 등록", exact=True)).to_be_visible()
        page.get_by_role("button", name="수정", exact=True).click()
        page.get_by_label("제목", exact=True).fill("Shader updated note")
        page.get_by_role("button", name="수정 저장").click()
        expect(page.get_by_role("alert")).to_contain_text("다른 작업")
        expect(page.get_by_label("제목", exact=True)).to_have_value("Shader updated note")
        page.get_by_role("button", name="새로고침", exact=True).click()
        expect(page.locator(".entry summary")).to_contain_text("Edited elsewhere")
        page.get_by_role("button", name="편집 취소").click()
        page.get_by_role("button", name="수정", exact=True).click()
        page.get_by_label("제목", exact=True).fill("Shader updated note")
        page.get_by_role("button", name="수정 저장").click()
        expect(page.get_by_role("status")).to_contain_text("수정했습니다")
        expect(page.locator(".entry summary")).to_contain_text("Shader updated note")
        assert records[0]["revision"] == 3
        page.get_by_role("button", name="삭제", exact=True).click()
        page.get_by_role("button", name="삭제 취소").click()
        assert not deletions
        page.get_by_role("button", name="수정", exact=True).click()
        page.get_by_role("button", name="삭제", exact=True).click()
        page.get_by_role("button", name="삭제 확인", exact=True).click()
        expect(page.get_by_role("alert")).to_contain_text("테스트 삭제 실패")
        assert len(records) == 1
        page.get_by_role("button", name="삭제 확인", exact=True).click()
        expect(page.get_by_role("status")).to_contain_text("삭제했습니다")
        expect(page.get_by_role("heading", name="지식 등록", exact=True)).to_be_visible()
        assert not records
        library = page.locator('.document-library')
        expect(library).to_contain_text("GAS reference")
        library.get_by_role("button", name=re.compile("정보 편집")).click()
        page.get_by_label("문서 제목", exact=True).fill("Unsent metadata")
        page.get_by_role("button", name="문서 편집 취소", exact=True).click()
        assert not metadata_edits
        library.get_by_role("button", name=re.compile("정보 편집")).click()
        page.get_by_label("문서 제목", exact=True).fill("GAS team reference")
        page.get_by_label("문서 엔진 버전", exact=True).fill("5.7")
        page.get_by_role("button", name="문서 정보 저장", exact=True).click()
        expect(library.get_by_role("alert")).to_contain_text("다른 작업")
        expect(page.get_by_label("문서 제목", exact=True)).to_have_value("GAS team reference")
        page.get_by_role("button", name="새로고침", exact=True).click()
        expect(library.locator('.document-entry')).to_contain_text("Changed reference")
        library.get_by_role("button", name=re.compile("정보 편집")).click()
        page.get_by_label("문서 제목", exact=True).fill("GAS team reference")
        page.get_by_label("문서 엔진 버전", exact=True).fill("5.7")
        page.get_by_label("문서 출처 URL", exact=False).fill("https://example.com/gas")
        page.get_by_role("button", name="문서 정보 저장", exact=True).click()
        expect(library.get_by_role("alert")).to_contain_text("저장 실패")
        expect(page.get_by_label("문서 엔진 버전", exact=True)).to_have_value("5.7")
        for width in (320, 375, 414, 768, 1280):
            page.set_viewport_size({"width": width, "height": 900})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"document form overflow at {width}"
        page.screenshot(path=str(screenshots / "documents-desktop.png"), full_page=True)
        page.get_by_role("button", name="문서 정보 저장", exact=True).click()
        expect(library.get_by_role("status")).to_contain_text("저장했습니다")
        expect(library.locator('.document-entry')).to_contain_text("GAS team reference")
        expect(page.locator('#engine-versions option[value="5.7"]')).to_have_count(1)
        assert reference["revision"] == 2
        page.get_by_role("button", name="대화", exact=False).first.click()
        page.get_by_label("언리얼 기술 질문").fill("UE 5.5 shader 오류 해결법")
        page.get_by_role("button", name="질문 보내기").click()
        expect(page.locator(".assistant .message-content")).to_contain_text("해결 절차")
        expect(page.locator('.chat-workspace')).not_to_have_class(re.compile(r'\bis-empty\b'))
        expect(page.get_by_role('heading', name='기록에서 답을 찾으세요.')).to_be_visible()
        code_block = page.locator('.answer-code').first
        expect(code_block.locator('.code-language')).to_have_text('cpp')
        expect(code_block.locator('code')).to_have_text(answer_code)
        assert code_block.locator('script, img').count() == 0
        assert page.evaluate('window.answerExecuted') is None
        code_block.locator('pre').focus()
        assert code_block.locator('pre').evaluate('(element) => element === document.activeElement')
        assert chats[0]["engine_version"] == "5.5"
        expect(page.locator(".assistant .message-role")).to_contain_text("UE 5.5")
        assert page.locator('a[href^="javascript:"]').count() == 0
        excerpt = page.locator('.source-excerpt').first
        expect(excerpt.locator('blockquote')).to_be_hidden()
        excerpt.locator('summary').focus()
        excerpt.locator('summary').press('Enter')
        expect(excerpt.locator('blockquote')).to_have_text(source_excerpt)
        expect(excerpt.locator('blockquote')).to_be_visible()
        expect(excerpt).to_contain_text('앞 1,000자')
        expect(page.locator('.sources small').first).to_contain_text('UE 5.5 · p.3')
        assert excerpt.locator('script, img').count() == 0
        assert page.evaluate('window.sourceExecuted') is None
        assert len(chats) == 1
        for width in (320, 375, 414, 768, 1280):
            page.set_viewport_size({"width": width, "height": 900})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"source excerpt overflow at {width}"
            assert code_block.locator('pre').evaluate('(element) => element.scrollWidth > element.clientWidth')
        code_block.screenshot(path=str(screenshots / 'answer-code-desktop.png'))
        page.locator('.assistant').first.screenshot(path=str(screenshots / "source-desktop.png"))
        page.set_viewport_size({"width": 375, "height": 900})
        code_block.screenshot(path=str(screenshots / 'answer-code-mobile.png'))
        page.locator('.assistant').first.screenshot(path=str(screenshots / "source-mobile.png"))
        page.set_viewport_size({"width": 1280, "height": 800})
        excerpt.locator('summary').focus()
        excerpt.locator('summary').press('Enter')
        expect(excerpt.locator('blockquote')).to_be_hidden()
        page.get_by_label("언리얼 기술 질문").fill("후속 질문")
        page.get_by_role("button", name="질문 보내기").click()
        expect(page.get_by_role("alert")).to_contain_text("테스트 연결 실패")
        expect(page.get_by_label("언리얼 기술 질문")).to_have_value("후속 질문")
        assert len(chats[1]["history"]) == 2
        page.get_by_label("검색 엔진 버전", exact=True).fill("5.6")
        page.get_by_label("언리얼 기술 질문").fill("다른 버전의 오류")
        page.get_by_role("button", name="질문 보내기").click()
        expect(page.locator(".assistant .message-role").last).to_contain_text("UE 5.6")
        assert chats[2]["engine_version"] == "5.6"
        assert chats[2]["history"] == []
        expect(page.locator('.assistant').last.locator('.source-excerpt')).to_have_count(0)
        expect(page.locator('.assistant').last.locator('.answer-code')).to_have_count(0)
        page.get_by_label("검색 엔진 버전", exact=True).fill("")
        page.get_by_label("언리얼 기술 질문").fill("전체 버전의 기록")
        page.get_by_role("button", name="질문 보내기").click()
        expect(page.locator(".assistant .message-role").last).to_contain_text("전체 버전")
        assert chats[3]["engine_version"] is None
        assert chats[3]["history"] == []
        short_excerpt = page.locator('.assistant').last.locator('.source-excerpt')
        short_excerpt.locator('summary').click()
        expect(short_excerpt.locator('blockquote')).to_have_text('짧은 검색 원문')
        assert '앞 1,000자' not in short_excerpt.inner_text()
        page.get_by_label("언리얼 기술 질문").fill("자료 변경 후 같은 질문")
        page.get_by_role("button", name="질문 보내기").click()
        expect(page.get_by_role("alert")).to_contain_text("지식이 변경")
        expect(page.get_by_label("언리얼 기술 질문")).to_have_value("자료 변경 후 같은 질문")
        page.get_by_role('navigation').get_by_role('button', name='설정', exact=False).click()
        expect(page.get_by_label("API 키", exact=True)).to_have_value("test-key")
        page.get_by_role('navigation').get_by_role('button', name='대화', exact=False).click()
        expect(page.get_by_label('언리얼 기술 질문')).to_have_value('자료 변경 후 같은 질문')
        expect(page.locator('.assistant')).to_have_count(3)
        expect(page.get_by_label("검색 엔진 버전", exact=True)).to_have_value("")
        expect(page.locator('.assistant')).to_have_count(3)
        assert len(chats) == 5  # 변경 감지 뒤 자동으로 LLM 요청을 재시도하지 않는다.
        page.get_by_role("button", name="질문 보내기").click()
        expect(page.locator('.assistant')).to_have_count(4)
        assert chats[5]["history"] == chats[4]["history"]
        page.locator('.assistant').last.locator('.source-excerpt summary').click()
        expect(page.locator('.assistant').last.locator('blockquote')).to_have_text('UPDATED_BROWSER_SOURCE_MARKER')
        page.get_by_label("언리얼 기술 질문").fill("인용 검증 실패 질문")
        page.get_by_role("button", name="질문 보내기").click()
        expect(page.get_by_role("alert")).to_contain_text("출처 검증에 실패")
        expect(page.get_by_label("언리얼 기술 질문")).to_have_value("인용 검증 실패 질문")
        page.get_by_role('navigation').get_by_role('button', name='설정', exact=False).click()
        expect(page.get_by_label("API 키", exact=True)).to_have_value("test-key")
        expect(page.get_by_label("모델", exact=True)).to_have_value("test-model")
        page.get_by_role('navigation').get_by_role('button', name='대화', exact=False).click()
        expect(page.get_by_label("검색 엔진 버전", exact=True)).to_have_value("")
        expect(page.locator('.assistant')).to_have_count(4)
        expect(page.locator('.user.failed').last).to_contain_text("처리 미완료")
        assert len(chats) == 7  # 출처 검증 실패도 자동으로 재시도하지 않는다.
        assert len(chats[6]["history"]) == 4
        for width in (320, 375, 414, 768, 1280):
            page.set_viewport_size({"width": width, "height": 900})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"citation failure overflow at {width}"
        page.screenshot(path=str(screenshots / "citation-failure-desktop.png"), full_page=True)
        page.get_by_role("button", name="질문 보내기").click()
        expect(page.locator('.assistant')).to_have_count(5)
        expect(page.get_by_role("alert")).to_have_count(0)
        expect(page.get_by_label("언리얼 기술 질문")).to_have_value("")
        assert chats[7]["question"] == chats[6]["question"]
        assert chats[7]["history"] == chats[6]["history"]
        page.get_by_label("언리얼 기술 질문").fill("자료 없는 질문")
        page.get_by_role("button", name="질문 보내기").click()
        expect(page.locator('.assistant')).to_have_count(6)
        expect(page.locator('.assistant').last).to_contain_text('BROWSER_INSUFFICIENT_MARKER')
        expect(page.locator('.assistant').last.locator('.sources')).to_have_count(0)
        expect(page.get_by_role("alert")).to_have_count(0)
        expect(page.get_by_label("언리얼 기술 질문")).to_have_value("")
        page.get_by_label("언리얼 기술 질문").fill("근거 부족 답변 뒤 후속 질문")
        page.get_by_role("button", name="질문 보내기").click()
        expect(page.locator('.assistant')).to_have_count(7)
        assert chats[9]["history"][-2:] == [
            {"role": "user", "content": "자료 없는 질문"},
            {"role": "assistant", "content": "BROWSER_INSUFFICIENT_MARKER: 등록된 근거가 부족합니다."},
        ]
        assert sum(item["content"] == "인용 검증 실패 질문" for item in chats[9]["history"]) == 1
        page.get_by_role('button', name='새 대화', exact=True).click()
        expect_idle_chat()
        expect(page.locator('.message')).to_have_count(0)
        question_input = page.get_by_label('언리얼 기술 질문')
        question_input.fill('첫 번째 줄')
        question_input.press('End')
        question_input.press('Shift+Enter')
        question_input.type('두 번째 줄')
        expect(question_input).to_have_value('첫 번째 줄\n두 번째 줄')
        assert len(chats) == 10
        question_input.press('Enter')
        expect(page.locator('.assistant')).to_have_count(1)
        assert chats[10]['question'] == '첫 번째 줄\n두 번째 줄'
        assert chats[10]['history'] == []
        page.get_by_role('button', name='새 대화', exact=True).click()
        expect_idle_chat()
        assert page.evaluate("localStorage.length + sessionStorage.length") == 0
        page.reload(wait_until="networkidle")
        page.get_by_role('navigation').get_by_role('button', name='설정', exact=False).click()
        expect(page.get_by_label("API 키", exact=True)).to_have_value("")
        expect(page.get_by_label("검색 엔진 버전", exact=True)).to_have_value("")
        expect(page.locator('.source-excerpt')).to_have_count(0)
        connection_config['has_server_key'] = True
        page.reload(wait_until='networkidle')
        page.get_by_role('navigation').get_by_role('button', name='설정', exact=False).click()
        expect(page.get_by_label('API 키', exact=True)).to_have_value('')
        expect(checker.locator('button')).to_be_enabled()
        checker.locator('button').click()
        expect(checker.locator('button')).to_contain_text('확인 중')
        respond(pending_checks.pop(), {"status": "ok", "message": "서버 키로 LLM 연결을 확인했습니다."})
        expect(checker.get_by_role('status')).to_contain_text('서버 키')
        assert 'x-llm-api-key' not in checks[-1]['headers']
        assert not errors, errors

        secure = browser.new_page(viewport={"width": 1280, "height": 800})
        secure.on("pageerror", lambda error: errors.append(str(error)))
        access = {"revoked": False, "chats": 0, "checks": 0}
        pending_knowledge = []
        private_note = {"id": "b" * 32, "title": "Protected note", "kind": "technique",
                        "engine_version": "5.5", "content": "Private knowledge", "revision": 1}
        def protected_api(route):
            if access["revoked"] and urlsplit(route.request.url).path == "/api/knowledge":
                # 병렬 목록 요청 중 접근 거부 뒤 도착한 과거 응답도 화면에 복원하면 안 된다.
                pending_knowledge.append(route)
            elif access["revoked"] or route.request.headers.get("authorization") != "Bearer browser-internal-token":
                respond(route, {"detail": {"code": "internal_access_required", "message": "내부 접근 토큰을 확인한 뒤 다시 연결해 주세요."}}, 401)
            elif urlsplit(route.request.url).path == "/api/config":
                respond(route, {"model": "test-model", "has_server_key": False})
            elif urlsplit(route.request.url).path == "/api/knowledge":
                respond(route, {"entries": [private_note], "count": 1})
            elif urlsplit(route.request.url).path == "/api/documents":
                respond(route, {"entries": [reference], "count": 1})
            elif urlsplit(route.request.url).path == "/api/llm/check":
                assert route.request.headers['x-llm-api-key'] == 'browser-llm-key'
                assert 'browser-internal-token' not in route.request.post_data
                access['checks'] += 1
                if access['checks'] == 1:
                    respond(route, {"detail": "LLM API 키가 거부되었습니다."}, 401)
                else:
                    respond(route, {"status": "ok", "message": "LLM 연결을 확인했습니다."})
            else:
                assert route.request.headers["x-llm-api-key"] == "browser-llm-key"
                assert "browser-internal-token" not in route.request.post_data
                access["chats"] += 1
                if access["chats"] == 1:
                    respond(route, {"detail": "LLM API 키가 거부되었습니다."}, 401)
                else:
                    respond(route, {"answer": "Private answer [1]", "status": "done", "sources": [
                        {"number": 1, "title": "Protected source", "excerpt": "PRIVATE_SOURCE_MARKER", "excerpt_truncated": False}]})
        secure.route("**/api/**", protected_api)
        secure.goto(base_url + "/", wait_until="networkidle")
        expect(secure.get_by_role("alert")).to_contain_text("내부 접근 토큰")
        for width in (320, 375, 414, 768):
            secure.set_viewport_size({"width": width, "height": 900})
            assert secure.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"access form overflow at {width}"
        secure.set_viewport_size({"width": 375, "height": 900})
        secure.screenshot(path=str(screenshots / "access-mobile.png"), full_page=True)
        secure.set_viewport_size({"width": 1280, "height": 800})
        expect(secure.get_by_label("API 키", exact=True)).to_be_disabled()
        secure.get_by_label("내부 접근 토큰", exact=True).fill("wrong-internal-token")
        secure.get_by_role("button", name="연결", exact=True).click()
        expect(secure.get_by_role("alert")).to_contain_text("내부 접근 토큰")
        secure.get_by_label("내부 접근 토큰", exact=True).fill("browser-internal-token")
        secure.get_by_role("button", name="연결", exact=True).click()
        expect(secure.get_by_label("API 키", exact=True)).to_be_enabled()
        secure.get_by_label("API 키", exact=True).fill("browser-llm-key")
        secure.get_by_role('button', name='LLM 연결 확인', exact=True).click()
        expect(secure.locator('.llm-check').get_by_role('alert')).to_contain_text('LLM API 키')
        expect(secure.get_by_label('API 키', exact=True)).to_be_enabled()
        secure.get_by_role('button', name='LLM 연결 확인', exact=True).click()
        expect(secure.locator('.llm-check').get_by_role('status')).to_contain_text('확인했습니다')
        secure.get_by_label('모델', exact=True).fill('test-model-edited')
        expect(secure.locator('.llm-check').get_by_role('status')).to_have_count(0)
        secure.get_by_label('모델', exact=True).fill('test-model')
        secure.get_by_role('navigation').get_by_role('button', name='대화', exact=False).click()
        secure.get_by_label("언리얼 기술 질문").fill("Private question")
        secure.get_by_role("button", name="질문 보내기").click()
        expect(secure.get_by_role("alert")).to_contain_text("LLM API 키")
        secure.get_by_role('navigation').get_by_role('button', name='설정', exact=False).click()
        expect(secure.get_by_label("API 키", exact=True)).to_be_enabled()
        secure.get_by_role('navigation').get_by_role('button', name='대화', exact=False).click()
        secure.get_by_role("button", name="질문 보내기").click()
        expect(secure.locator(".assistant .message-content")).to_contain_text("Private answer")
        secure.locator('.source-excerpt summary').click()
        expect(secure.locator('.source-excerpt blockquote')).to_have_text('PRIVATE_SOURCE_MARKER')
        secure.get_by_role("button", name=re.compile("지식 저장소")).click()
        secure.locator("summary").filter(has_text="Protected note").click()
        secure.get_by_role("button", name="수정", exact=True).click()
        secure.get_by_label("제목", exact=True).fill("Unsaved private title")
        secure.locator('.document-library').get_by_role("button", name=re.compile("정보 편집")).click()
        secure.get_by_label("문서 제목", exact=True).fill("Unsaved private metadata")
        secure.get_by_label('등록된 기록 검색', exact=True).fill('Protected')
        secure.get_by_role('navigation').get_by_role('button', name='설정', exact=False).click()
        expect(secure.get_by_label('내부 접근 토큰', exact=True)).to_have_value('browser-internal-token')
        secure.get_by_role('navigation').get_by_role('button', name=re.compile('지식 저장소')).click()
        expect(secure.get_by_label('제목', exact=True)).to_have_value('Unsaved private title')
        expect(secure.get_by_label('문서 제목', exact=True)).to_have_value('Unsaved private metadata')
        expect(secure.get_by_label('등록된 기록 검색', exact=True)).to_have_value('Protected')
        secure.get_by_role('navigation').get_by_role('button', name='설정', exact=False).click()
        secure.get_by_role("button", name="연결 해제", exact=True).click()
        expect(secure.get_by_label("내부 접근 토큰", exact=True)).to_have_value("")
        expect(secure.get_by_label("API 키", exact=True)).to_have_value("")
        secure.get_by_role('navigation').get_by_role('button', name=re.compile('지식 저장소')).click()
        expect(secure.get_by_label("제목", exact=True)).to_have_value("")
        expect(secure.get_by_label('등록된 기록 검색', exact=True)).to_have_value('')
        assert secure.locator(".entry").count() == 0
        assert secure.locator(".document-entry").count() == 0
        expect(secure.get_by_role("form", name="문서 정보 편집", exact=True)).to_have_count(0)
        secure.get_by_role('navigation').get_by_role('button', name='설정', exact=False).click()
        secure.get_by_label("내부 접근 토큰", exact=True).fill("browser-internal-token")
        secure.get_by_role("button", name="연결", exact=True).click()
        expect(secure.get_by_label('API 키', exact=True)).to_be_enabled()
        secure.get_by_role('navigation').get_by_role('button', name=re.compile('지식 저장소')).click()
        expect(secure.locator(".entry")).to_contain_text("Protected note")
        secure.get_by_label('등록된 기록 검색', exact=True).fill('Protected')
        access["revoked"] = True
        secure.get_by_role("button", name="새로고침", exact=True).click()
        expect(secure.get_by_role("alert")).to_contain_text("내부 접근 토큰")
        expect(secure.get_by_role('heading', name='연결 설정', exact=True)).to_be_visible()
        assert len(pending_knowledge) == 1
        respond(pending_knowledge.pop(), {"entries": [private_note], "count": 1})
        expect(secure.get_by_role("button", name="연결", exact=True)).to_be_enabled()
        secure.get_by_role('navigation').get_by_role('button', name=re.compile('지식 저장소')).click()
        assert secure.locator(".entry").count() == 0
        assert secure.locator(".document-entry").count() == 0
        expect(secure.get_by_label('등록된 기록 검색', exact=True)).to_have_value('')
        secure.get_by_role("button", name="대화", exact=False).first.click()
        assert secure.locator(".message").count() == 0
        assert secure.locator(".source-excerpt").count() == 0
        assert secure.evaluate("localStorage.length + sessionStorage.length") == 0
        access['revoked'] = False
        secure.get_by_role('navigation').get_by_role('button', name='설정', exact=False).click()
        secure.get_by_label('내부 접근 토큰', exact=True).fill('browser-internal-token')
        secure.get_by_role('button', name='연결', exact=True).click()
        expect(secure.get_by_label('API 키', exact=True)).to_be_enabled()
        secure.get_by_label('API 키', exact=True).fill('browser-llm-key')
        access['revoked'] = True
        secure.get_by_role('button', name='LLM 연결 확인', exact=True).click()
        expect(secure.locator('.access-settings').get_by_role('alert')).to_contain_text('내부 접근 토큰')
        expect(secure.locator('.llm-check').get_by_role('status')).to_have_count(0)
        expect(secure.locator('.llm-check').get_by_role('alert')).to_have_count(0)
        expect(secure.get_by_label('API 키', exact=True)).to_have_value('')
        expect(secure.get_by_role('button', name='LLM 연결 확인', exact=True)).to_be_disabled()
        secure.reload(wait_until="networkidle")
        expect(secure.get_by_label("내부 접근 토큰", exact=True)).to_have_value("")
        expect(secure.get_by_role("alert")).to_contain_text("내부 접근 토큰")
        assert not errors, errors
        print("Browser checks passed: centered composer-only idle/new chat, Enter/Shift+Enter, settings navigation/state preservation/auth redirect, answer code/plain fallback/HTML-as-text/raw history, local knowledge filtering/reset/refresh/auth lifecycle, citation-failure retry/history, insufficient followup, explicit LLM check/progress/errors/key-model invalidation/server key/auth denial, corpus-change retry, sources, document metadata, knowledge CRUD, 5 viewport widths.")
        print(f"Screenshots: {screenshots}")
        browser.close()


if __name__ == "__main__":
    run()
