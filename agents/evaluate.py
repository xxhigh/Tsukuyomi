"""정답 문서 ID로 검색 품질을 평가한다. LLM 호출이나 답변 채점은 하지 않는다."""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from . import docs, documents

DocumentId = Annotated[str, Field(min_length=1, max_length=200)]


class EvaluationError(ValueError):
    """질문 본문을 포함하지 않는 평가 자료·실행 조건 오류."""


class EvaluationCase(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=8000)
    expected_doc_ids: list[DocumentId] = Field(max_length=100)
    engine_version: str | None = Field(default=None, min_length=1, max_length=50)
    kind: Literal["technique", "error"] | None = None


class EvaluationSet(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cases: list[EvaluationCase] = Field(min_length=1, max_length=10000)

    @model_validator(mode="after")
    def unique_case_ids(self):
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("평가 질문 ID는 중복될 수 없습니다.")
        return self


def validate_corpus(dataset: EvaluationSet, chunks) -> None:
    if not chunks:
        raise EvaluationError("평가할 언리얼 지식이 없습니다. 먼저 지식을 등록하세요.")
    versions = {}
    for chunk in chunks:
        versions.setdefault(chunk.metadata["doc_id"], set()).add(chunk.metadata.get("engine_version"))
    for case in dataset.cases:
        for doc_id in set(case.expected_doc_ids):
            if doc_id not in versions:
                raise EvaluationError(f"{case.id}: 정답 문서 ID가 현재 자료에 없습니다: {doc_id}")
            if case.engine_version and not any(documents.version_matches(stored, case.engine_version)
                                               for stored in versions[doc_id]):
                raise EvaluationError(f"{case.id}: 정답 문서가 선택한 엔진 버전과 일치하지 않습니다: {doc_id}")


def evaluate(dataset: EvaluationSet, k: int = 5, retrieve=None) -> dict:
    if isinstance(k, bool) or not isinstance(k, int) or not 1 <= k <= 10:
        raise ValueError("k는 1~10의 정수여야 합니다.")
    retrieve = retrieve or docs.retrieve
    results = []
    for case in dataset.cases:
        found = retrieve(case.question, k=k, kind=case.kind, engine_version=case.engine_version)[:k]
        ranked = [item["doc_id"] for item in found]
        expected = set(case.expected_doc_ids)
        first_rank = next((rank for rank, doc_id in enumerate(ranked, 1) if doc_id in expected), None)
        results.append({"id": case.id, "engine_version": case.engine_version,
                        "expected_doc_ids": sorted(expected),
                        "ranked_doc_ids": ranked, "first_relevant_rank": first_rank,
                        "hit": first_rank is not None if expected else None,
                        "recall": len(expected.intersection(ranked)) / len(expected) if expected else None,
                        "reciprocal_rank": (1 / first_rank if first_rank else 0.0) if expected else None,
                        "correct_rejection": not ranked if not expected else None,
                        "missing_doc_ids": sorted(expected.difference(ranked))})
    positive = [item for item in results if item["hit"] is not None]
    negative = [item for item in results if item["correct_rejection"] is not None]

    def average(items, key):
        return sum(item[key] for item in items) / len(items) if items else None

    return {"k": k, "case_count": len(results), "answerable_count": len(positive),
            "unanswerable_count": len(negative),
            "metrics": {"hit_rate_at_k": average(positive, "hit"),
                        "recall_at_k": average(positive, "recall"),
                        "mrr_at_k": average(positive, "reciprocal_rank"),
                        "correct_rejection_rate": average(negative, "correct_rejection")},
            "cases": results}


def _write_report(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description="언리얼 RAG 검색 품질 평가 (외부 LLM 미사용)")
    parser.add_argument("cases", type=Path, help="cases 목록이 담긴 JSON 파일")
    parser.add_argument("--k", type=int, choices=range(1, 11), default=5, help="검색 청크 수 (기본 5)")
    parser.add_argument("--output", type=Path, help="결과 JSON 저장 경로 (기존 보고서 덮어쓰기)")
    args = parser.parse_args()
    if args.output:
        output = args.output.resolve()
        if (output == args.cases.resolve() or output.is_relative_to(docs.DOCS_DIR.resolve())
                or output.is_relative_to(docs.STORE_DIR.resolve())):
            parser.error("평가 자료·원본 문서·인덱스 경로에 보고서를 저장할 수 없습니다.")
    load_dotenv()
    try:
        raw = args.cases.read_bytes()
        dataset = EvaluationSet.model_validate_json(raw)
        threshold = float(os.getenv("RAG_MIN_SIMILARITY", "0.45"))
        if not math.isfinite(threshold):
            raise EvaluationError("RAG_MIN_SIMILARITY는 유한한 숫자여야 합니다.")
        fingerprint = docs._fingerprint()
        validate_corpus(dataset, docs._chunks())
        report = evaluate(dataset, k=args.k)
        if docs._fingerprint() != fingerprint:
            raise EvaluationError("평가 도중 자료가 변경되었습니다. 자료 변경을 마친 뒤 다시 실행하세요.")
        report.update({"created_at": datetime.now(timezone.utc).isoformat(),
                       "dataset_sha256": hashlib.sha256(raw).hexdigest(), "corpus_sha256": fingerprint,
                       "retrieval_code_sha256": hashlib.sha256(Path(docs.__file__).read_bytes()).hexdigest(),
                       "settings": {"embedding_model": docs.EMBED_MODEL, "min_similarity": threshold,
                                    "keyword_weight": docs.KEYWORD_WEIGHT, "min_keyword_score": docs.MIN_KEYWORD_SCORE,
                                    "candidates": docs.CANDIDATES, "keyword_candidates": docs.CANDIDATES,
                                    "chunk_size": docs.CHUNK_SIZE,
                                    "chunk_overlap": docs.CHUNK_OVERLAP}})
        content = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            _write_report(args.output, content)
        print(content, end="")
        return 0
    except ValidationError as error:
        fields = [".".join(map(str, item["loc"])) or "루트(JSON 형식/질문 ID 중복)" for item in error.errors()]
        print(f"평가 JSON 필드를 확인하세요: {', '.join(fields)}", file=sys.stderr)
        return 1
    except (EvaluationError, docs.DocumentIdConflictError) as error:
        print(f"평가 실패: {error}", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError, ImportError):
        # 예외 원문에는 질문·문서 내용이 포함될 수 있으므로 출력하지 않는다.
        print("평가에 실패했습니다. JSON 파일·검색 모델·인덱스·저장 권한을 확인하세요.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
