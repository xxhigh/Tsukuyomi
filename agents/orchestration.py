"""복합 질문의 제한된 계획·병렬 검색·근거 평가·답변 수정 노드."""

import json
from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from time import monotonic

from prompts import ANSWER_ASSESSOR_PROMPT, EVIDENCE_ASSESSOR_PROMPT, ORCHESTRATOR_PROMPT

from .state import AnswerAssessment, EvidenceAssessment, ResearchPlan

MAX_COMPLEX_SEARCHES = 6
MAX_PARALLEL_TASKS = 3
MAX_PLANNING_ROUNDS = 4
MAX_RESEARCH_SECONDS = 120
MAX_ANSWER_REVISIONS = 1


def _query_key(worker, query):
    return worker, " ".join(query.casefold().split())


def _numbered(state):
    return [{"number": number, **item} for number, item in enumerate(state.get("evidence", []), 1)]


def _expired(state):
    return monotonic() - state["research_started_at"] >= MAX_RESEARCH_SECONDS


def build_orchestration(llm, *, search, snapshot, make_prompt, history, answer, web_available):
    """검색·자료 일관성·기존 Writer는 주 그래프의 구현을 재사용한다."""
    planner = llm.with_structured_output(ResearchPlan, method="function_calling")
    assessor = llm.with_structured_output(EvidenceAssessment, method="function_calling")
    reviewer = llm.with_structured_output(AnswerAssessment, method="function_calling")
    plan_prompt = make_prompt(ORCHESTRATOR_PROMPT, "orchestrator_prompt")
    evidence_prompt = make_prompt(EVIDENCE_ASSESSOR_PROMPT, "evidence_assessor_prompt")
    review_prompt = make_prompt(ANSWER_ASSESSOR_PROMPT, "answer_assessor_prompt")

    def messages(template, state, payload):
        return template.invoke({"history": history(state), "input": json.dumps(payload, ensure_ascii=False)}).to_messages()

    def decision(state, worker):
        return [*state.get("decisions", []), {"worker": worker, "query": "", "reason": "복합 질문 처리"}]

    def finish(state, warning=None):
        warnings = list(state.get("warnings", []))
        if warning:
            warnings.append(warning)
        gaps = [item["missing"] or item["question"] for item in state.get("research_tasks", [])
                if not item["sufficient"]]
        if gaps:
            warnings.append("일부 하위 질문의 근거가 부족합니다. 답변의 적용 조건과 미확인 내용을 확인하세요.")
        return {"complex_mode": True, "active_tasks": [], "next_stage": "compose",
                "research_gaps": gaps, "warnings": list(dict.fromkeys(warnings))}

    def orchestrator(state):
        started = state.get("research_started_at", monotonic())
        state = {**state, "research_started_at": started}
        snapshot(state)
        if _expired(state):
            return {**finish(state, "복합 조사 시간 한도에 도달해 추가 검색을 중단했습니다."),
                    "research_started_at": started}
        rounds = state.get("rounds", 0)
        planning_rounds = state.get("planning_rounds", 0)
        if rounds >= MAX_COMPLEX_SEARCHES or planning_rounds >= MAX_PLANNING_ROUNDS:
            return finish(state, "복합 조사 횟수 한도에 도달해 확보한 근거로 답변합니다.")
        tasks = state.get("research_tasks", [])
        plan = ResearchPlan.model_validate(planner.invoke(messages(plan_prompt, state, {
            "question": state["question"], "engine_version": state.get("engine_version"),
            "web_search_available": web_available, "warnings": state.get("warnings", []),
            "remaining_searches": MAX_COMPLEX_SEARCHES - rounds,
            "tasks": tasks, "evidence": _numbered(state),
        })))
        snapshot(state)
        updates = {"complex_mode": True, "research_started_at": started,
                   "planning_rounds": planning_rounds + 1, "decisions": decision(state, "orchestrator")}
        if not plan.tasks:
            warning = "조사 계획을 만들지 못했습니다. 질문을 구체화해 주세요." if not tasks else None
            return {**updates, **finish(state, warning)}
        if _expired(state):
            return {**updates, **finish(state, "복합 조사 시간 한도에 도달해 추가 검색을 중단했습니다.")}
        existing_ids = {item["id"] for item in tasks}
        used_queries = {_query_key(item["worker"], item["query"]) for item in tasks}
        active = []
        rejected = False
        warnings = list(state.get("warnings", []))
        for task in plan.tasks[:MAX_COMPLEX_SEARCHES - rounds]:
            item = task.model_dump()
            # 같은 배치/미래 작업의 결과는 아직 없으므로 의존을 허용하지 않는다.
            if item["id"] in existing_ids or any(dep not in existing_ids for dep in item["depends_on"]):
                rejected = True
                continue
            if item["worker"] == "web_search" and not web_available:
                item["worker"] = "technique"
                warnings.append("Tavily 키가 없어 계획된 웹 검색을 로컬 검색으로 대체했습니다.")
            key = _query_key(item["worker"], item["query"])
            if key in used_queries or any(other["id"] == item["id"] for other in active):
                rejected = True
                continue
            used_queries.add(key)
            active.append(item)
        if rejected:
            warnings.append("중복되거나 의존 관계가 잘못된 조사 작업은 실행하지 않았습니다.")
        updates["warnings"] = list(dict.fromkeys(warnings))
        if not active:
            return {**updates, **finish({**state, **updates}, "실행 가능한 추가 조사 작업이 없습니다.")}
        return {**updates, "active_tasks": active, "next_stage": "workers"}

    def workers(state):
        if _expired(state):
            return finish(state, "복합 조사 시간 한도에 도달해 추가 검색을 중단했습니다.")
        active = state["active_tasks"]
        fingerprint = snapshot(state, local=any(item["worker"] != "web_search" for item in active))
        if _expired(state):
            return {**finish(state, "복합 조사 시간 한도에 도달해 추가 검색을 중단했습니다."),
                    "corpus_sha256": fingerprint}
        search_state = {**state, "corpus_sha256": fingerprint}
        # 각 스레드에 tracing context를 복사하고 결과 병합은 계획 순서로 수행한다.
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_TASKS) as executor:
            futures = [executor.submit(copy_context().run, search, search_state, item) for item in active]
            results = [future.result() for future in futures]
        snapshot(search_state)
        evidence = list(state.get("evidence", []))
        numbers = {(item["doc_id"], item["chunk_id"]): i for i, item in enumerate(evidence, 1)}
        warnings = list(state.get("warnings", []))
        candidates = []
        previous = {item["id"]: item for item in state.get("research_tasks", [])}
        for task, result in zip(active, results):
            sources = []
            for item in result.get("evidence", [])[:5]:
                key = item["doc_id"], item["chunk_id"]
                if key not in numbers:
                    evidence.append(item)
                    numbers[key] = len(evidence)
                sources.append(numbers[key])
            dependencies = sorted({number for dep in task["depends_on"]
                                   for number in previous[dep]["relevant_sources"]})
            candidates.append({**task, "source_numbers": sorted(set(sources)),
                               "dependency_sources": dependencies,
                               "search_failed": bool(result.get("warnings") and not sources)})
            warnings.extend(result.get("warnings", []))
        merged = {**search_state, "evidence": evidence}
        assessments = []
        # 빈 검색이나 시간 초과는 평가 모델을 호출하지 않고 미해결로 남긴다.
        if any(item["source_numbers"] or item["dependency_sources"] for item in candidates) and not _expired(state):
            assessments = EvidenceAssessment.model_validate(assessor.invoke(messages(evidence_prompt, state, {
                "question": state["question"], "engine_version": state.get("engine_version"),
                "tasks": candidates, "evidence": _numbered(merged),
                "warnings": list(dict.fromkeys(warnings)),
            }))).tasks
            snapshot(search_state)
        tasks = list(state.get("research_tasks", []))
        accepted = set(state.get("accepted_sources", []))
        for task in candidates:
            matches = [item for item in assessments if item.task_id == task["id"]]
            review = matches[0] if len(matches) == 1 else None
            allowed = set(task["source_numbers"] + task["dependency_sources"])
            valid = review is not None and set(review.relevant_sources).issubset(allowed)
            relevant = sorted(set(review.relevant_sources)) if valid else []
            accepted.update(relevant)
            sufficient = bool(valid and review.sufficient and relevant)
            tasks.append({**task, "relevant_sources": relevant, "sufficient": sufficient,
                          "status": "failed" if task["search_failed"] else "done" if sufficient else "insufficient",
                          "finding": review.finding if valid and relevant else "",
                          "missing": review.missing if valid else "검색 자료의 근거를 확인하지 못했습니다."})
        return {"evidence": evidence, "corpus_sha256": fingerprint, "research_tasks": tasks,
                "accepted_sources": sorted(accepted), "active_tasks": [],
                "rounds": state.get("rounds", 0) + len(active),
                "warnings": list(dict.fromkeys(warnings)), "next_stage": "orchestrator",
                "decisions": [*state.get("decisions", []),
                              *[{"worker": item["worker"], "query": "", "reason": "복합 질문 검색"} for item in active]]}

    def compose(state):
        snapshot(state)
        if _expired(state):
            return {"answer": "조사 시간 한도에 도달해 답변 검증을 완료하지 못했습니다. 질문 범위를 좁혀 다시 요청해 주세요.",
                    "sources": [], "status": "insufficient", "next_stage": "end"}
        result = answer(state)
        return {"draft_answer": result["answer"], "draft_sources": result["sources"],
                "next_stage": "verify" if result["status"] == "done" else "end",
                **(result if result["status"] != "done" else {}), "decisions": decision(state, "answer")}

    def verify(state):
        snapshot(state)
        if not _expired(state):
            allowed = set(state.get("accepted_sources", []))
            review = AnswerAssessment.model_validate(reviewer.invoke(messages(review_prompt, state, {
                "question": state["question"], "engine_version": state.get("engine_version"),
                "answer": state["draft_answer"], "evidence": [item for item in _numbered(state) if item["number"] in allowed],
                "research_gaps": state.get("research_gaps", []),
            })))
            snapshot(state)
            if review.supported and review.addresses_question:
                return {"answer": state["draft_answer"], "sources": state["draft_sources"], "status": "done",
                        "next_stage": "end", "decisions": decision(state, "answer_review")}
            if state.get("revision_count", 0) < MAX_ANSWER_REVISIONS and not _expired(state):
                return {"revision_feedback": review.feedback, "revision_count": state.get("revision_count", 0) + 1,
                        "next_stage": "compose", "decisions": decision(state, "answer_review")}
        return {"answer": "확보한 자료로 답변의 근거와 질문 해결 여부를 확인하지 못했습니다. 엔진 버전·오류 로그·질문 범위를 구체화해 다시 요청해 주세요.",
                "sources": [], "status": "insufficient", "next_stage": "end",
                "decisions": decision(state, "answer_review")}

    return {"orchestrator": orchestrator, "research_workers": workers,
            "compose": compose, "verify": verify}
