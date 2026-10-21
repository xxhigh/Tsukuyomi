"""요청별 Supervisor 상태. 키와 worker 호출 내역은 저장하지 않는다."""

from typing import Annotated, Literal, TypedDict

from pydantic import BaseModel, Field, StringConstraints

SearchText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]


class Route(BaseModel):
    worker: Literal["technique", "troubleshooting", "web_search", "answer", "direct_answer", "orchestrator"]
    query: str = Field(default="", max_length=1000)
    reason: str = Field(max_length=500)


class ResearchTask(BaseModel):
    id: int = Field(ge=1, le=6)
    question: SearchText
    query: SearchText
    worker: Literal["technique", "troubleshooting", "web_search"]
    depends_on: list[int] = Field(default_factory=list, max_length=6)


class ResearchPlan(BaseModel):
    # 의존 작업은 선행 결과를 확인한 다음 계획 호출에서 생성한다.
    tasks: list[ResearchTask] = Field(default_factory=list, max_length=3)
    reason: str = Field(max_length=500)


class TaskAssessment(BaseModel):
    task_id: int = Field(ge=1, le=6)
    relevant_sources: list[int] = Field(default_factory=list, max_length=30)
    sufficient: bool
    finding: str = Field(max_length=1500)
    missing: str = Field(max_length=500)


class EvidenceAssessment(BaseModel):
    tasks: list[TaskAssessment] = Field(default_factory=list, max_length=3)


class AnswerAssessment(BaseModel):
    supported: bool
    addresses_question: bool
    feedback: str = Field(max_length=1500)


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
    warnings: list[str]
    complex_mode: bool
    research_tasks: list[dict]
    active_tasks: list[dict]
    accepted_sources: list[int]
    planning_rounds: int
    research_started_at: float
    next_stage: str
    research_gaps: list[str]
    draft_answer: str
    draft_sources: list[dict]
    revision_count: int
    revision_feedback: str
