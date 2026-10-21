"""Supervisor → 기술/오류/웹 검색 → 근거 기반 답변."""

import json
import re
from contextlib import nullcontext
from uuid import UUID

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, StateGraph
from markdown_it import MarkdownIt

from prompts import SUPERVISOR_PROMPT, WRITER_PROMPT

from . import docs, documents, knowledge, web
from .state import ChatState, Route

MAX_RESEARCH_ROUNDS = 2
MAX_SOURCE_EXCERPT = 1000
NO_EVIDENCE = "관련된 언리얼 지식이 충분하지 않습니다. 엔진 버전과 오류 로그 또는 관련 기술·해결 사례를 등록해 주세요."


def _prompt(system: str, name: str):
    # 시스템 지시는 메시지 객체로 넣어 중괄호를 템플릿 변수로 해석하지 않는다.
    return ChatPromptTemplate.from_messages([SystemMessage(content=system), MessagesPlaceholder("history"),
                                             ("human", "{input}")]).with_config(run_name=name)


SUPERVISOR_TEMPLATE = _prompt(SUPERVISOR_PROMPT, "supervisor_prompt")
WRITER_TEMPLATE = _prompt(WRITER_PROMPT, "writer_prompt")


def run_config(source: str, *, run_id: UUID | None = None, engine_version: str | None = None) -> dict:
    """그래프 실행 설정. LANGSMITH_TRACING=true이면 이 이름·태그·metadata로 LangSmith에 기록된다."""
    config = {"recursion_limit": 12, "run_name": "tsukuyomi_chat", "tags": [source],
              "metadata": {"source": source, "engine_version": engine_version or "전체"}}
    if run_id:
        config["run_id"] = run_id
    return config


def _history(state):
    return [(HumanMessage if item["role"] == "user" else AIMessage)(content=item["content"])
            for item in state.get("history", [])[-10:]]


def _check_corpus(expected: str | None) -> str:
    with knowledge.locked():
        current = docs._fingerprint()
        if expected is not None and expected != current:
            raise docs.CorpusChangedError()
        return current


def _citation_numbers(text: str) -> set[str]:
    """설명 텍스트만 검사해 코드의 배열 인덱스와 링크를 인용으로 오인하지 않는다."""
    citations = set()
    for block in MarkdownIt("commonmark", {"html": False}).parse(text):
        in_link = False
        for token in block.children or []:
            if token.type in {"link_open", "link_close"}:
                in_link = token.type == "link_open"
            elif token.type == "text" and not in_link:
                citations.update(re.findall(r"\[(\d+)\]", token.content))
    return citations


def _matches_version(item, version):
    # 웹 문서의 버전은 검색어로 검증할 수 없다. 외부 참고 자료로 구분해 Writer가 조건을 확인한다.
    return item.get("source_type") == "web" or not version or documents.version_matches(item.get("engine_version"), version)


def build_graph(llm, retrieve=None, web_search=None):
    """기본 로컬 검색은 자료 변경을 검사한다. 주입한 검색 함수의 일관성은 호출자가 관리한다."""
    local_retrieval = retrieve is None
    if local_retrieval:
        retrieve = docs.retrieve
    web_available = web_search is not None or web.available()
    web_search = web_search or web.search
    router = llm.with_structured_output(Route, method="function_calling")

    def supervisor(state: ChatState):
        rounds = state.get("rounds", 0)
        warnings = state.get("warnings", [])
        if rounds >= MAX_RESEARCH_ROUNDS:
            route = Route(worker="answer", reason="검색 상한에 도달했습니다.")
        else:
            # 자료 본문은 제어 노드에 전달하지 않는다.
            summary = [{"title": item["title"], "engine_version": item.get("engine_version"),
                        "kind": item.get("kind")} for item in state.get("evidence", [])]
            messages = SUPERVISOR_TEMPLATE.invoke({"history": _history(state), "input": json.dumps(
                {"question": state["question"], "engine_version": state.get("engine_version"),
                 "search_version_family": documents.version_family(state.get("engine_version")) or None,
                 "web_search_available": web_available, "warnings": warnings,
                 "rounds": rounds, "sources": summary}, ensure_ascii=False)}).to_messages()
            route = router.invoke(messages)
            if route.worker == "web_search" and not web_available:
                warnings = list(dict.fromkeys([*warnings, str(web.WebSearchError("unconfigured"))]))
                route = Route(worker="technique", reason="웹 검색 키가 없어 로컬 자료를 검색합니다.")
            if rounds == 0 and route.worker == "answer":
                # 빈 검색어는 research에서 원문을 사용하므로 Route의 검색어 길이 제한을 피한다.
                route = Route(worker="technique", reason="답변 전 검색이 필요합니다.")
        decision = route.model_dump()
        return {"route": decision, "decisions": [*state.get("decisions", []), decision], "warnings": warnings}

    def research(state: ChatState):
        route = state["route"]
        kind = "error" if route["worker"] == "troubleshooting" else "technique"
        version = state.get("engine_version")
        options = {"engine_version": version} if version else {}
        # LLM 호출 사이에 자료가 바뀌어도 이전/현재 원문을 섞지 않는다.
        with knowledge.locked() if local_retrieval else nullcontext():
            fingerprint = _check_corpus(state.get("corpus_sha256")) if local_retrieval else None
            found = retrieve(route["query"] or state["question"], k=5, kind=kind, **options)
            if local_retrieval:
                _check_corpus(fingerprint)
        evidence = {(item["doc_id"], item["chunk_id"]): item for item in state.get("evidence", [])
                    if _matches_version(item, version)}
        for item in found:
            if not _matches_version(item, version):
                continue
            evidence[(item["doc_id"], item["chunk_id"])] = item
        return {"evidence": list(evidence.values())[:10], "rounds": state.get("rounds", 0) + 1,
                "corpus_sha256": fingerprint}

    def research_web(state: ChatState):
        warnings = state.get("warnings", [])
        try:
            # 빈 웹 검색어를 사용자 원문으로 대체하지 않는다.
            found = web_search(state["route"]["query"], engine_version=state.get("engine_version"))
        except web.WebSearchError as error:
            found = []
            warnings = list(dict.fromkeys([*warnings, str(error)]))
        evidence = {(item["doc_id"], item["chunk_id"]): item for item in state.get("evidence", [])}
        for item in found:
            evidence[(item["doc_id"], item["chunk_id"])] = item
        return {"evidence": list(evidence.values())[:10], "rounds": state.get("rounds", 0) + 1,
                "warnings": warnings}

    def answer(state: ChatState):
        if local_retrieval and state.get("corpus_sha256") is not None:
            _check_corpus(state.get("corpus_sha256"))
        evidence = state.get("evidence", [])
        numbered = [{"number": index, **item} for index, item in enumerate(evidence, 1)]
        if not numbered:
            prefix = (f"선택한 엔진 버전 계열({documents.version_family(state['engine_version'])})에서 "
                      if state.get("engine_version") else "")
            return {"answer": prefix + NO_EVIDENCE, "sources": [], "status": "insufficient"}
        messages = WRITER_TEMPLATE.invoke({"history": _history(state), "input": json.dumps(
            {"question": state["question"], "engine_version": state.get("engine_version"),
             "search_version_family": documents.version_family(state.get("engine_version")) or None,
             "warnings": state.get("warnings", []),
             "evidence": numbered}, ensure_ascii=False)}).to_messages()
        response = llm.invoke(messages)
        if local_retrieval and state.get("corpus_sha256") is not None:
            _check_corpus(state.get("corpus_sha256"))
        text = response.content if isinstance(response.content, str) else ""
        citations = _citation_numbers(text)
        if not citations or not citations.issubset({str(item["number"]) for item in numbered}):
            return {"answer": "검색 자료의 출처를 확인할 수 있는 답변을 생성하지 못했습니다. 질문을 구체화해 다시 요청해 주세요.",
                    "sources": [], "status": "invalid_citations"}
        sources = [{**{key: value for key, value in item.items() if key != "content"},
                    "excerpt": item["content"][:MAX_SOURCE_EXCERPT],
                    "excerpt_truncated": len(item["content"]) > MAX_SOURCE_EXCERPT}
                   for item in numbered if str(item["number"]) in citations]
        return {"answer": text, "sources": sources, "status": "done"}

    graph = StateGraph(ChatState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("technique", research)
    graph.add_node("troubleshooting", research)
    graph.add_node("web_search", research_web)
    graph.add_node("answer", answer)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", lambda state: state["route"]["worker"],
                                {"technique": "technique", "troubleshooting": "troubleshooting",
                                 "web_search": "web_search", "answer": "answer"})
    graph.add_edge("technique", "supervisor")
    graph.add_edge("troubleshooting", "supervisor")
    graph.add_edge("web_search", "supervisor")
    graph.add_edge("answer", END)
    return graph.compile()
