"""Optional: serve the frontend, then python3.11 tests/browser_versions.py.

Uses mock API responses only. Requires Playwright and Chromium.
Set BROWSER_BASE_URL for a nondefault server; screenshots stay in the temp folder.
"""

import json
import os
import tempfile
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


def run():
    base_url = os.getenv("BROWSER_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    output = Path(tempfile.gettempdir()) / "tsukuyomi-version-check"
    output.mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})
        errors, edits, chats = [], [], []
        page.on("pageerror", lambda error: errors.append(str(error)))
        reference = {"filename": "GAS.md", "title": "GAS patch reference", "engine_version": "5.4.2", "revision": 0}

        def respond(route, data):
            route.fulfill(status=200, content_type="application/json", body=json.dumps(data))

        def api(route):
            path = route.request.url.split("/api/", 1)[1]
            if path == "config":
                respond(route, {"model": "test-model", "has_server_key": True})
            elif path == "knowledge":
                respond(route, {"entries": []})
            elif path == "documents":
                respond(route, {"entries": [reference]})
            elif path.startswith("documents/"):
                payload = route.request.post_data_json
                edits.append(payload)
                reference.update({key: value for key, value in payload.items() if key != "expected_revision"})
                reference["revision"] += 1
                respond(route, {"entry": reference, "message": "문서 정보 저장 완료"})
            elif path == "chat":
                chats.append(route.request.post_data_json)
                respond(route, {"status": "done", "answer": "패치별 적용 조건을 확인하세요. [1]", "sources": [{
                    "number": 1, "title": "패치 수정 기록", "engine_version": "5.4.2", "excerpt": "5.4.2에서 확인한 기록",
                }]})
            else:
                raise AssertionError(f"Unexpected API request: {path}")

        page.route("**/api/**", api)
        page.goto(base_url, wait_until="networkidle")
        page.get_by_role("navigation").get_by_role("button", name="지식 저장소").click()
        page.get_by_role("button", name="GAS.md 정보 편집").click()
        patch = page.get_by_role("checkbox", name="UE 5.4.2", exact=True)
        expect(patch).to_be_checked()
        patch.uncheck()
        expect(patch).to_be_visible()
        patch.check()
        patch_input = page.get_by_label("확인한 패치 버전 추가", exact=False)
        patch_input.fill("5.4")
        page.get_by_role("button", name="버전 추가", exact=True).click()
        expect(page.get_by_role("alert")).to_contain_text("5.4.2 형식")
        patch_input.fill("5.4.3")
        page.get_by_role("button", name="버전 추가", exact=True).click()
        expect(page.get_by_role("checkbox", name="UE 5.4.3", exact=True)).to_be_checked()
        patch_input.fill("5.4.3")
        page.get_by_role("button", name="버전 추가", exact=True).click()
        expect(page.get_by_role("checkbox", name="UE 5.4.3", exact=True)).to_have_count(1)
        patch_input.fill("5.4.4")
        page.get_by_role("button", name="문서 정보 저장", exact=True).click()
        expect(page.get_by_text("문서 정보 저장 완료", exact=True)).to_be_visible()
        assert edits == [{"title": reference["title"], "engine_version": "5.4.2, 5.4.3, 5.4.4",
                          "source_url": None, "expected_revision": 0}]

        page.get_by_role("button", name="GAS.md 정보 편집").click()
        for width in (320, 375, 414, 768, 1280):
            page.set_viewport_size({"width": width, "height": 900})
            assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            patch_row = page.locator(".patch-version-row")
            box = patch_row.bounding_box()
            assert box["x"] >= 0 and box["x"] + box["width"] <= width
            page.screenshot(path=str(output / f"patch-editor-{width}.png"), full_page=True)
        page.get_by_role("button", name="문서 편집 취소", exact=True).click()
        page.get_by_role("navigation").get_by_role("button", name="대화").click()
        page.locator(".version-trigger").click()
        expect(page.get_by_role("menuitemradio", name="UE 5.4", exact=True)).to_be_visible()
        expect(page.get_by_role("menuitemradio", name="UE 5.4.2", exact=True)).to_have_count(0)
        page.get_by_role("menuitemradio", name="UE 5.4", exact=True).focus()
        page.keyboard.press("Enter")
        page.get_by_label("언리얼 기술 질문").fill("5.4.1에서 발생한 오류의 수정 패치를 찾아줘")
        page.get_by_role("button", name="질문 보내기").click()
        expect(page.locator(".message.assistant")).to_be_visible()
        assert chats[0]["engine_version"] == "5.4"
        assert "5.4.1" in chats[0]["question"]
        expect(page.locator(".sources small")).to_have_text("UE 5.4.2")
        assert not errors, errors
        print(f"PASS: patch add/validation/deduplication/preservation/save, minor picker, raw source version, 5 widths; {output}")
        browser.close()


if __name__ == "__main__":
    run()
