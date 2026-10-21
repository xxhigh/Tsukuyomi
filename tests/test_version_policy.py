import json

import pytest
from langchain_core.documents import Document

from agents import docs, documents, evaluate, knowledge
from agents.graph import build_graph
from agents.state import Route
from tests.test_graph import EVIDENCE, FakeLLM
from tests.test_retrieval import LocalEmbeddings


@pytest.mark.parametrize("stored,wanted,expected", [
    ("5.4", "5.4", True), ("5.4.2", "5.4", True),
    ("5.4", "5.4.2", True), ("5.4.1", "5.4.2", True),
    (" 5.4.12 ", " 5.4 ", True), ("5.3, 5.4.2", "5.4", True),
    ("4.27.2", "4.27", True), ("5.4.20", "5.4.2", True),
    ("5.40.2", "5.4", False), ("5.3.2", "5.4", False),
    ("6.4.2", "5.4", False), ("5.4.2", "5.5", False),
    (None, "5.4", False), ("미지정", "5.4", False), ("", "5.4", False),
    ("UE 5.4", "5.4", False), ("5.4-preview", "5.4", False),
    ("5.4.2.1", "5.4", False), ("UE 5.4", "UE 5.4", True),
    ("5.4-preview", "5.4-preview", True), ("5.4", "", False),
])
def test_version_filter_groups_only_numeric_minor_families(stored, wanted, expected):
    assert documents.version_matches(stored, wanted) is expected


@pytest.mark.parametrize("requested", ["5.4", "5.4.1", "5.4.2"])
def test_patch_metadata_survives_storage_index_reload_and_minor_search(workspace, monkeypatch, requested):
    monkeypatch.setattr(docs, "_embeddings", LocalEmbeddings)
    records = [knowledge.save(knowledge.KnowledgeInput(
        title="nanite reference", kind="technique", engine_version=version,
        content="nanite shader verification")) for version in ("5.4", "5.4.1", "5.4.2", "5.40.2", "5.5.2")]
    path = docs.DOCS_DIR / "patch-note.md"
    path.write_text("nanite shader patch notes", encoding="utf-8")
    documents.update(path.name, documents.DocumentUpdate(
        title="Patch reference", engine_version="5.3, 5.4.2", expected_revision=0))
    original = {file.name: file.read_bytes() for file in docs.DOCS_DIR.iterdir()}

    for _ in range(2):
        found = docs.retrieve("nanite shader", k=10, engine_version=requested)
        assert {item["doc_id"] for item in found} == {item["id"] for item in records[:3]} | {"patch-note"}
        assert {item["engine_version"] for item in found} == {"5.4", "5.4.1", "5.4.2", "5.3, 5.4.2"}
        monkeypatch.setattr(docs, "_store", None)
    assert {file.name: file.read_bytes() for file in docs.DOCS_DIR.iterdir()} == original


@pytest.mark.parametrize("requested", ["5.4", "5.4.1"])
def test_graph_passes_patch_question_and_exact_evidence_versions_to_writer(requested):
    question = "UE 5.4.1의 오류가 어느 패치에서 수정되었나요?"
    evidence = [
        {**EVIDENCE, "doc_id": "affected", "engine_version": "5.4.1", "content": "5.4.1에서 증상을 재현했다."},
        {**EVIDENCE, "doc_id": "fixed", "engine_version": "5.4.2", "content": "핫픽스 기록: 해당 오류는 5.4.2에서 수정되었다."},
        {**EVIDENCE, "doc_id": "other", "engine_version": "5.5.2", "content": "OTHER_MINOR_EXCLUDED"},
    ]
    model = FakeLLM([Route(worker="troubleshooting", query=question, reason="패치 기록 검색"),
                     Route(worker="answer", reason="근거 확보")],
                    "5.4.1 재현 기록과 5.4.2 수정 기록을 확인하세요. [1][2]")
    result = build_graph(model, lambda *args, **kwargs: evidence).invoke(
        {"question": question, "engine_version": requested})

    router = json.loads(model.router_messages[0][-1].content)
    writer = json.loads(model.writer_messages[0][-1].content)
    assert router["search_version_family"] == writer["search_version_family"] == "5.4"
    assert writer["engine_version"] == requested
    assert writer["question"] == question
    assert [item["engine_version"] for item in writer["evidence"]] == ["5.4.1", "5.4.2"]
    assert [item["content"] for item in writer["evidence"]] == [item["content"] for item in evidence[:2]]
    assert [source["engine_version"] for source in result["sources"]] == ["5.4.1", "5.4.2"]
    assert result["status"] == "done"
    assert result["rounds"] == 1


@pytest.mark.parametrize("wanted,stored", [("5.4", "5.4.2"), ("5.4.2", "5.4"), ("5.4.1", "5.3, 5.4.2")])
def test_evaluation_gold_validation_uses_the_same_minor_policy(wanted, stored):
    dataset = evaluate.EvaluationSet.model_validate({"cases": [{
        "id": "patch", "question": "shader", "expected_doc_ids": ["patch-note"], "engine_version": wanted,
    }]})
    evaluate.validate_corpus(dataset, [Document(page_content="record", metadata={
        "doc_id": "patch-note", "engine_version": stored,
    })])
