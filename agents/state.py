"""요청별 Supervisor 상태. 키와 worker 호출 내역은 저장하지 않는다."""

from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class Route(BaseModel):
    worker: Literal["technique", "troubleshooting", "answer"]
    query: str = Field(default="", max_length=1000)
    reason: str = Field(max_length=500)


class ChatState(TypedDict, total=False):
    question: str
    history: list[dict]
    engine_version: str | None
    rounds: int
    route: dict
    decisions: list[dict]
    evidence: list[dict]
    corpus_sha256: str | None
    answer: str
    sources: list[dict]
    status: Literal["done", "insufficient", "invalid_citations"]
