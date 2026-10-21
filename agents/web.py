"""Tavily 검색 결과를 답변 근거로 변환한다. 키·HTTP 오류 원문은 반환하지 않는다."""

import hashlib
import math
import os
import re
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

import httpx

SEARCH_URL = "https://api.tavily.com/search"
MAX_RESULTS = 5
MAX_CONTENT = 4000
TIMEOUT = 20


class WebSearchError(RuntimeError):
    MESSAGES = {
        "unconfigured": "Tavily API 키가 설정되지 않아 웹 검색을 사용할 수 없습니다.",
        "invalid_query": "공개 웹 검색어를 만들지 못했습니다. 공개 기술명이나 오류 코드로 다시 질문해 주세요.",
        "unauthorized": "Tavily API 키가 거부되었습니다. 서버의 웹 검색 설정을 확인해 주세요.",
        "rate_limited": "Tavily 웹 검색 사용 한도에 도달했습니다. 잠시 후 다시 시도하거나 서버 설정을 확인해 주세요.",
        "unavailable": "Tavily 웹 검색에 연결하지 못했습니다. 확보한 자료만 사용합니다.",
        "invalid_response": "Tavily 웹 검색 응답을 읽지 못했습니다. 확보한 자료만 사용합니다.",
    }

    def __init__(self, code: str):
        self.code = code
        super().__init__(self.MESSAGES[code])


def available() -> bool:
    return bool(os.getenv("TAVILY_API_KEY", "").strip())


def _client():
    return httpx.Client(timeout=TIMEOUT, follow_redirects=False)


def _url(value) -> str | None:
    if not isinstance(value, str) or len(value) > 4096:
        return None
    value = value.strip()
    if any(character.isspace() or ord(character) < 32 for character in value):
        return None
    try:
        parts = urlsplit(value)
        if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
            return None
        parts.port  # Reject malformed ports before rendering a link.
        return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, parts.fragment))
    except ValueError:
        return None


def search(query: str, *, engine_version: str | None = None) -> list[dict]:
    """검색어만 전송한다. 질문 원문·대화·로컬 근거를 이 함수에서 추가하지 않는다."""
    key = os.getenv("TAVILY_API_KEY", "").strip()
    if not key:
        raise WebSearchError("unconfigured")
    if not isinstance(query, str) or not query.strip() or len(query) > 1000:
        raise WebSearchError("invalid_query")
    version = (engine_version or "").strip()
    version = version if re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", version) else ""
    search_query = " ".join(part for part in ("Unreal Engine", version, query.strip()) if part)
    try:
        with _client() as client:
            response = client.post(SEARCH_URL, headers={"Authorization": f"Bearer {key}"}, json={
                "query": search_query, "topic": "general", "search_depth": "basic", "max_results": MAX_RESULTS,
                "include_answer": False, "include_raw_content": False, "include_images": False,
            })
        if response.status_code in {401, 403}:
            raise WebSearchError("unauthorized")
        if response.status_code in {429, 432, 433}:
            raise WebSearchError("rate_limited")
        if not response.is_success:
            raise WebSearchError("unavailable")
        payload = response.json()
    except httpx.HTTPError:
        raise WebSearchError("unavailable") from None
    except (ValueError, UnicodeError):
        raise WebSearchError("invalid_response") from None
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise WebSearchError("invalid_response")
    results, seen = [], set()
    retrieved_at = datetime.now(timezone.utc).isoformat()
    for item in payload["results"]:
        if not isinstance(item, dict):
            continue
        url = _url(item.get("url"))
        content = item.get("content")
        if not url or not isinstance(content, str) or not content.strip():
            continue
        identity = urlunsplit(urlsplit(url)._replace(fragment=""))
        if identity in seen:
            continue
        seen.add(identity)
        title = item.get("title")
        score = item.get("score")
        score = float(score) if isinstance(score, (int, float)) and math.isfinite(score) else 0.0
        results.append({
            "doc_id": "web:" + hashlib.sha256(identity.encode()).hexdigest(), "chunk_id": 0,
            "title": title.strip()[:300] if isinstance(title, str) and title.strip() else urlsplit(url).hostname,
            "kind": "web", "source_type": "web", "engine_version": "미지정", "url": url,
            "page": 0, "content": content.strip()[:MAX_CONTENT], "score": score, "retrieved_at": retrieved_at,
            "published_date": item["published_date"][:100] if isinstance(item.get("published_date"), str) else None,
        })
        if len(results) >= MAX_RESULTS:
            break
    return results
