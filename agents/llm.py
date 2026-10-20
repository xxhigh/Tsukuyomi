"""요청별 외부 LLM 클라이언트. API 키를 전역 상태에 넣지 않는다."""

import os

from langchain_openai import ChatOpenAI

from .state import Route


def create_llm(api_key: str, model: str | None = None):
    return ChatOpenAI(api_key=api_key, model=model or os.getenv("OPENAI_MODEL", "gpt-5-mini"),
                      base_url=os.getenv("OPENAI_BASE_URL") or None,
                      timeout=60, max_retries=1)


def check_connection(llm) -> None:
    """팀 자료 없이 Supervisor와 같은 tool calling 응답 형식을 검사한다."""
    result = llm.with_structured_output(Route, method="function_calling").invoke(
        "Connection check only. Call Route with worker='answer', query='', reason='OK'.")
    Route.model_validate(result)
