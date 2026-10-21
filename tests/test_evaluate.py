import hashlib
import json
import sys

import pytest
from langchain_core.documents import Document
from pydantic import ValidationError

from agents import docs, evaluate
from tests.test_retrieval import LocalEmbeddings, add


def dataset(cases):
    return evaluate.EvaluationSet.model_validate({"cases": cases})


def case(case_id="test", expected=None, **values):
    return {"id": case_id, "question": case_id,
            "expected_doc_ids": ["note"] if expected is None else expected, **values}


def test_metrics_separate_no_answer_cases_and_count_real_chunk_rank():
    data = dataset([case("partial", ["B", "B", "C"]), case("miss", ["D"]),
                    case("rejected", []), case("false-positive", [])])
    results = {"partial": ["A", "A", "B", "C"], "miss": [], "rejected": [], "false-positive": ["A"]}
    report = evaluate.evaluate(data, k=3, retrieve=lambda query, **kwargs: [{"doc_id": doc} for doc in results[query]])
    assert report["metrics"] == pytest.approx({"hit_rate_at_k": 0.5, "recall_at_k": 0.25,
                                             "mrr_at_k": 1 / 6, "correct_rejection_rate": 0.5})
    assert report["answerable_count"] == report["unanswerable_count"] == 2
    assert report["cases"][0]["ranked_doc_ids"] == ["A", "A", "B"]
    assert report["cases"][0]["first_relevant_rank"] == 3
    assert report["cases"][0]["expected_doc_ids"] == ["B", "C"]
    assert report["cases"][0]["missing_doc_ids"] == ["C"]


def test_absent_case_group_is_reported_as_null():
    no_results = lambda *args, **kwargs: []
    positive = evaluate.evaluate(dataset([case()]), retrieve=no_results)
    assert positive["metrics"]["correct_rejection_rate"] is None
    negative = evaluate.evaluate(dataset([case(expected=[])]), retrieve=no_results)
    assert negative["metrics"]["hit_rate_at_k"] is None
    assert negative["metrics"]["recall_at_k"] is None
    assert negative["metrics"]["mrr_at_k"] is None
    assert negative["cases"][0]["reciprocal_rank"] is None


def test_version_and_search_kind_reach_existing_retriever():
    calls = []
    def retrieve(query, **options):
        calls.append((query, options))
        return []
    evaluate.evaluate(dataset([case(engine_version="5.5", kind="error")]), k=4, retrieve=retrieve)
    assert calls == [("test", {"k": 4, "kind": "error", "engine_version": "5.5"})]


@pytest.mark.parametrize("k", [0, 11, True, 1.5])
def test_invalid_k_never_runs_retrieval(k):
    def fail(*args, **kwargs):
        pytest.fail("잘못된 k로 검색하면 안 됩니다")
    with pytest.raises(ValueError):
        evaluate.evaluate(dataset([case()]), k=k, retrieve=fail)


@pytest.mark.parametrize("cases", [[], [case(), case()], [case(question=" ")],
    [case(expected=[" "])], [{"id": "missing-gold", "question": "question"}],
    [case(engine_version=" ")]])
def test_invalid_evaluation_data_is_rejected(cases):
    with pytest.raises(ValidationError):
        dataset(cases)


@pytest.mark.parametrize("doc_id,version", [("missing", "5.5"), ("note", "5.6")])
def test_missing_or_wrong_version_gold_is_rejected(doc_id, version):
    data = dataset([case(expected=[doc_id], engine_version=version)])
    with pytest.raises(evaluate.EvaluationError):
        evaluate.validate_corpus(data, [Document(page_content="text", metadata={"doc_id": "note", "engine_version": "5.5"})])


def prepare_cli(workspace, monkeypatch):
    record = add("nanite")
    path = workspace / "cases.json"
    path.write_text(json.dumps({"cases": [case("positive", [record["id"]], question="private-eval-question nanite", engine_version="5.5"),
                                         case("negative", [], question="nanite", engine_version="5.6")]}), encoding="utf-8")
    output = workspace / "report.json"
    monkeypatch.setattr(sys, "argv", ["agents.evaluate", str(path), "--output", str(output)])
    return path, output, record


def test_cli_evaluates_real_faiss_and_writes_reproducible_report(workspace, monkeypatch, capsys):
    path, output, record = prepare_cli(workspace, monkeypatch)
    original = path.read_bytes()
    monkeypatch.setattr(docs, "_embeddings", LocalEmbeddings)
    assert evaluate.main() == 0
    report = json.loads(output.read_text())
    assert json.loads(capsys.readouterr().out) == report
    assert report["metrics"] == {"hit_rate_at_k": 1.0, "recall_at_k": 1.0,
                                  "mrr_at_k": 1.0, "correct_rejection_rate": 1.0}
    assert report["cases"][0]["ranked_doc_ids"] == [record["id"]]
    assert report["dataset_sha256"] == hashlib.sha256(original).hexdigest()
    assert report["corpus_sha256"] == docs._fingerprint()
    assert report["settings"]["embedding_model"] == "BAAI/bge-m3"
    assert report["settings"]["keyword_candidates"] == docs.CANDIDATES
    assert "private-eval-question" not in output.read_text()
    assert path.read_bytes() == original


@pytest.mark.parametrize("failure", ["corpus_changed", "retrieval_failed", "write_failed"])
def test_failed_evaluation_preserves_existing_report(workspace, monkeypatch, capsys, failure):
    path, output, record = prepare_cli(workspace, monkeypatch)
    output.write_text("previous report", encoding="utf-8")
    def retrieve(*args, **kwargs):
        if failure == "corpus_changed":
            add("new source")
        elif failure == "retrieval_failed":
            raise RuntimeError("private error content")
        return [{"doc_id": record["id"]}]
    monkeypatch.setattr(docs, "retrieve", retrieve)
    if failure == "write_failed":
        def fail(*args):
            raise OSError("private error content")
        monkeypatch.setattr(evaluate.os, "replace", fail)
    assert evaluate.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "private error content" not in captured.err
    assert "private-eval-question" not in captured.err
    assert output.read_text() == "previous report"
    assert not list(workspace.glob("*.tmp"))


def test_empty_corpus_stops_before_model_loading(workspace, monkeypatch, capsys):
    path = workspace / "cases.json"
    path.write_text(json.dumps({"cases": [case(expected=[])]}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["agents.evaluate", str(path)])
    def fail():
        pytest.fail("빈 자료로 모델을 로딩하면 안 됩니다")
    monkeypatch.setattr(docs, "_embeddings", fail)
    assert evaluate.main() == 1
    assert "지식이 없습니다" in capsys.readouterr().err


def test_duplicate_document_ids_stop_evaluation_and_preserve_report(workspace, monkeypatch, capsys):
    path, output, record = prepare_cli(workspace, monkeypatch)
    original_cases = path.read_bytes()
    output.write_text("previous report", encoding="utf-8")
    (docs.DOCS_DIR / f"{record['id']}.md").write_text("private-source-body", encoding="utf-8")

    def unexpected():
        pytest.fail("문서 ID가 중복되면 평가 모델을 로드하면 안 됩니다")

    monkeypatch.setattr(docs, "_embeddings", unexpected)
    assert evaluate.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "문서 ID가 중복" in captured.err
    assert "private" not in captured.err
    assert output.read_text() == "previous report"
    assert path.read_bytes() == original_cases


@pytest.mark.parametrize("destination", ["dataset", "source", "index"])
def test_output_cannot_overwrite_evaluation_data_or_retrieval_files(workspace, monkeypatch, destination):
    path = workspace / "cases.json"
    path.write_text("keep evaluation labels", encoding="utf-8")
    output = {"dataset": path, "source": docs.DOCS_DIR / "report.json", "index": docs.STORE_DIR / "report.json"}[destination]
    monkeypatch.setattr(sys, "argv", ["agents.evaluate", str(path), "--output", str(output)])
    with pytest.raises(SystemExit) as error:
        evaluate.main()
    assert error.value.code == 2
    assert path.read_text() == "keep evaluation labels"
