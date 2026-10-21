import json
import math

import pytest
from langchain_core.embeddings import Embeddings

from agents import docs, documents, knowledge


class LocalEmbeddings(Embeddings):
    """FAISS와 저장/갱신을 실제 실행하되 모델 다운로드만 대체한다."""
    def embed_query(self, text):
        vector = [float(term in text.lower()) for term in ("nanite", "shader", "lumen")]
        if not any(vector):
            vector = [0.0, 0.0, 1.0]
        length = math.sqrt(sum(value * value for value in vector))
        return [value / length for value in vector]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


def add(title):
    return knowledge.save(knowledge.KnowledgeInput(title=title, kind="technique", engine_version="5.5",
                                                  content=f"{title} 팀 기술 기록"))


def test_empty_library_does_not_load_embedding_model(workspace, monkeypatch):
    def fail():
        pytest.fail("빈 저장소에서 모델을 불러오면 안 됩니다")
    monkeypatch.setattr(docs, "_embeddings", fail)
    assert docs.retrieve("Nanite") == []


def test_fingerprint_separates_filenames_and_contents_to_invalidate_stale_index(workspace, monkeypatch):
    monkeypatch.setattr(docs, "_embeddings", LocalEmbeddings)
    docs.DOCS_DIR.mkdir()
    first = docs.DOCS_DIR / "a.md"
    first.write_bytes(b"b.mdc")
    fingerprint = docs._fingerprint()
    assert docs.retrieve("lumen")[0]["doc_id"] == "a"
    first.unlink()
    # 파일명+내용을 구분 없이 연결하면 두 저장소 모두 b'a.mdb.mdc'가 된다.
    (docs.DOCS_DIR / "a.mdb.md").write_bytes(b"c")
    assert docs._fingerprint() != fingerprint
    assert docs.retrieve("lumen")[0]["doc_id"] == "a.mdb"


def test_faiss_round_trip_new_document_reindex_and_irrelevance(workspace, monkeypatch):
    monkeypatch.setattr(docs, "_embeddings", LocalEmbeddings)
    original = add("Nanite")
    found = docs.retrieve("Nanite")
    assert found[0]["doc_id"] == original["id"]
    pointer = json.loads((docs.STORE_DIR / "current.json").read_text())
    assert "data/vectorstores" not in str(docs.STORE_DIR)
    assert docs.retrieve("shader") == []
    monkeypatch.setattr(docs, "_store", None)
    assert docs.retrieve("Nanite")[0]["title"] == "Nanite"
    added = add("shader")
    assert docs.retrieve("shader")[0]["doc_id"] == added["id"]
    assert json.loads((docs.STORE_DIR / "current.json").read_text())["generation"] != pointer["generation"]


def test_failed_rebuild_preserves_published_index(workspace, monkeypatch):
    monkeypatch.setattr(docs, "_embeddings", LocalEmbeddings)
    add("Nanite")
    docs.build()
    pointer = (docs.STORE_DIR / "current.json").read_text()
    add("shader")
    def fail():
        raise RuntimeError("model unavailable")
    monkeypatch.setattr(docs, "_embeddings", fail)
    with pytest.raises(RuntimeError):
        docs.retrieve("shader")
    assert (docs.STORE_DIR / "current.json").read_text() == pointer


def test_markdown_and_pdf_ingestion_include_source_page(workspace):
    import pymupdf
    docs.DOCS_DIR.mkdir()
    (docs.DOCS_DIR / "Rendering.md").write_text("# Rendering\nNanite team notes", encoding="utf-8")
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((50, 50), "Shader team reference")
        pdf.save(docs.DOCS_DIR / "Shader.pdf")
    chunks = docs._chunks()
    assert {chunk.metadata["title"] for chunk in chunks} == {"Rendering", "Shader"}
    assert next(chunk for chunk in chunks if chunk.metadata["title"] == "Shader").metadata["page"] == 1


def test_updated_and_deleted_knowledge_never_reuses_stale_index(workspace, monkeypatch):
    monkeypatch.setattr(docs, "_embeddings", LocalEmbeddings)
    record = add("Nanite")
    assert docs.retrieve("Nanite")[0]["doc_id"] == record["id"]
    knowledge.update(record["id"], knowledge.KnowledgeUpdate(title="shader", kind="technique",
                     engine_version="5.6", content="shader 새 팀 기록", expected_revision=1))
    assert docs.retrieve("Nanite") == []
    assert docs.retrieve("shader")[0]["engine_version"] == "5.6"
    other = add("lumen")
    knowledge.delete(record["id"], 2)
    assert docs.retrieve("shader") == []
    assert docs.retrieve("lumen")[0]["doc_id"] == other["id"]
    knowledge.delete(other["id"], 1)
    assert docs.retrieve("lumen") == []


def test_version_filter_finds_matching_document_beyond_global_candidate_limit(workspace, monkeypatch):
    monkeypatch.setattr(docs, "_embeddings", LocalEmbeddings)
    for index in range(docs.CANDIDATES + 1):
        knowledge.save(knowledge.KnowledgeInput(title=f"Nanite old {index}", kind="technique",
                       engine_version="5.4", content="nanite"))
    selected = knowledge.save(knowledge.KnowledgeInput(title="Nanite 5.5", kind="technique",
                              engine_version="5.5", content="nanite shader"))
    assert selected["id"] not in {item["doc_id"] for item in docs.retrieve("nanite")}
    found = docs.retrieve("nanite", engine_version="5.5")
    assert [item["doc_id"] for item in found] == [selected["id"]]
    assert [item["doc_id"] for item in docs.retrieve("nanite", engine_version="5.5.4")] == [selected["id"]]
    assert docs.retrieve("nanite", engine_version="5.6") == []


def test_versioned_search_excludes_unknown_version_references(workspace, monkeypatch):
    monkeypatch.setattr(docs, "_embeddings", LocalEmbeddings)
    record = add("nanite")
    (docs.DOCS_DIR / "note.md").write_text("nanite reference without engine version", encoding="utf-8")
    assert len(docs.retrieve("nanite")) == 2
    assert [item["doc_id"] for item in docs.retrieve("nanite", engine_version="5.5")] == [record["id"]]


@pytest.mark.parametrize("suffix", [".md", ".pdf"])
def test_reference_metadata_changes_reindex_with_stable_identity(workspace, monkeypatch, suffix):
    import pymupdf
    monkeypatch.setattr(docs, "_embeddings", LocalEmbeddings)
    docs.DOCS_DIR.mkdir()
    path = docs.DOCS_DIR / f"Rendering{suffix}"
    if suffix == ".md":
        path.write_text("Nanite team reference", encoding="utf-8")
    else:
        with pymupdf.open() as pdf:
            pdf.new_page().insert_text((50, 50), "Nanite team reference")
            pdf.set_metadata({"title": "Original PDF title"})
            pdf.save(path)
    original = path.read_bytes()
    assert docs.retrieve("nanite")[0]["title"] == ("Rendering" if suffix == ".md" else "Original PDF title")
    assert docs.retrieve("nanite", engine_version="5.5") == []
    documents.update(path.name, documents.DocumentUpdate(title="팀 렌더링 자료", engine_version="5.5",
                     source_url="https://example.com/reference", expected_revision=0))
    found = docs.retrieve("nanite", engine_version="5.5")[0]
    assert found["title"] == "팀 렌더링 자료"
    assert found["doc_id"] == "Rendering"
    assert found["url"] == "https://example.com/reference"
    assert found["page"] == (0 if suffix == ".md" else 1)
    documents.update(path.name, documents.DocumentUpdate(title="팀 렌더링 자료", engine_version="5.6", expected_revision=1))
    assert docs.retrieve("nanite", engine_version="5.5") == []
    assert docs.retrieve("nanite", engine_version="5.6")[0]["url"] == ""
    assert path.read_bytes() == original
