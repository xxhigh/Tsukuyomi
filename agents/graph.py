"""Supervisor → 기술/오류 검색 → Supervisor → 근거 기반 답변."""

import json
import re
from contextlib import nullcontext

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from markdown_it import MarkdownIt

from prompts import SUPERVISOR_PROMPT, WRITER_PROMPT

from . import docs, documents, knowledge
from .state import ChatState, Route

MAX_RESEARCH_ROUNDS = 2
MAX_SOURCE_EXCERPT = 1000
NO_EVIDENCE = "관련된 언리얼 지식이 충분하지 않습니다. 엔진 버전과 오류 로그 또는 관련 기술·해결 사례를 등록해 주세요."


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


def build_graph(llm, retrieve=None):
    """기본 로컬 검색은 자료 변경을 검사한다. 주입한 검색 함수의 일관성은 호출자가 관리한다."""
    local_retrieval = retrieve is None
    if local_retrieval:
        retrieve = docs.retrieve
    router = llm.with_structured_output(Route, method="function_calling")

    def supervisor(state: ChatState):
        rounds = state.get("rounds", 0)
        if rounds >= MAX_RESEARCH_ROUNDS:
            route = Route(worker="answer", reason="검색 상한에 도달했습니다.")
        else:
            # 자료 본문은 제어 노드에 전달하지 않는다.
            summary = [{"title": item["title"], "engine_version": item.get("engine_version"),
                        "kind": item.get("kind")} for item in state.get("evidence", [])]
            route = router.invoke([SystemMessage(content=SUPERVISOR_PROMPT), *_history(state),
                                   HumanMessage(content=json.dumps({"question": state["question"],
                                                "engine_version": state.get("engine_version"),
                                                "rounds": rounds, "sources": summary}, ensure_ascii=False))])
            if rounds == 0 and route.worker == "answer":
                # 빈 검색어는 research에서 원문을 사용하므로 Route의 검색어 길이 제한을 피한다.
                route = Route(worker="technique", reason="답변 전 검색이 필요합니다.")
        decision = route.model_dump()
        return {"route": decision, "decisions": [*state.get("decisions", []), decision]}

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
                    if not version or documents.version_matches(item.get("engine_version"), version)}
        for item in found:
            if version and not documents.version_matches(item.get("engine_version"), version):
                continue
            evidence[(item["doc_id"], item["chunk_id"])] = item
        return {"evidence": list(evidence.values())[:10], "rounds": state.get("rounds", 0) + 1,
                "corpus_sha256": fingerprint}

    def answer(state: ChatState):
        if local_retrieval:
            _check_corpus(state.get("corpus_sha256"))
        evidence = state.get("evidence", [])
        if not evidence:
            prefix = f"선택한 엔진 버전({state['engine_version']})에서 " if state.get("engine_version") else ""
            return {"answer": prefix + NO_EVIDENCE, "sources": [], "status": "insufficient"}
        numbered = [{"number": index, **item} for index, item in enumerate(evidence, 1)]
        response = llm.invoke([SystemMessage(content=WRITER_PROMPT), *_history(state),
                               HumanMessage(content=json.dumps({"question": state["question"],
                                            "engine_version": state.get("engine_version"),
                                            "evidence": numbered}, ensure_ascii=False))])
        if local_retrieval:
            _check_corpus(state.get("corpus_sha256"))
        text = response.content if isinstance(response.content, str) else ""
        citations = _citation_numbers(text)
        if not citations or not citations.issubset({str(number) for number in range(1, len(evidence) + 1)}):
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
    graph.add_node("answer", answer)
    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", lambda state: state["route"]["worker"],
                                {"technique": "technique", "troubleshooting": "troubleshooting", "answer": "answer"})
    graph.add_edge("technique", "supervisor")
    graph.add_edge("troubleshooting", "supervisor")
    graph.add_edge("answer", END)
    return graph.compile()
