import pytest

from agents import docs
from tests.test_retrieval import LocalEmbeddings, add


def references(directory):
    import pymupdf

    directory.mkdir(exist_ok=True)
    (directory / "Guide.md").write_text("Nanite markdown reference", encoding="utf-8")
    with pymupdf.open() as pdf:
        pdf.new_page().insert_text((50, 50), "Nanite PDF reference")
        pdf.save(directory / "Guide.pdf")


def test_different_sources_with_same_document_id_are_rejected(workspace):
    references(docs.DOCS_DIR)

    with pytest.raises(ValueError, match="문서 ID가 중복"):
        docs._chunks()


@pytest.mark.parametrize("suffix", [".md", ".PDF"])
def test_reference_cannot_share_structured_knowledge_id(workspace, monkeypatch, suffix):
    record = add("Nanite")
    path = docs.DOCS_DIR / f"{record['id']}{suffix}"
    path.write_text("private reference body", encoding="utf-8")
    originals = {path.name: path.read_bytes() for path in docs.DOCS_DIR.iterdir()}

    def unexpected():
        pytest.fail("ID 중복 확인 전에 임베딩 모델을 불러오면 안 됩니다")

    monkeypatch.setattr(docs, "_embeddings", unexpected)
    with pytest.raises(docs.DocumentIdConflictError):
        docs.build()

    assert not (docs.STORE_DIR / "current.json").exists()
    assert {path.name: path.read_bytes() for path in docs.DOCS_DIR.iterdir()} == originals


@pytest.mark.parametrize("cached", [True, False])
def test_conflict_preserves_existing_index_and_removal_restores_search(workspace, monkeypatch, cached):
    docs.DOCS_DIR.mkdir()
    (docs.DOCS_DIR / "Guide.md").write_text("Nanite team reference", encoding="utf-8")
    monkeypatch.setattr(docs, "_embeddings", LocalEmbeddings)
    assert docs.retrieve("Nanite")[0]["doc_id"] == "Guide"
    pointer = docs.STORE_DIR / "current.json"
    published = pointer.read_bytes()
    if not cached:
        monkeypatch.setattr(docs, "_store", None)
    (docs.DOCS_DIR / "Guide.pdf").write_bytes(b"must reject before parsing this PDF")

    def unexpected(*args, **kwargs):
        pytest.fail("ID 중복 확인 전에 기존 캐시를 읽거나 새 인덱스를 만들면 안 됩니다")

    with monkeypatch.context() as guard:
        from langchain_community.vectorstores import FAISS
        guard.setattr(docs, "_build", unexpected)
        guard.setattr(docs, "_embeddings", unexpected)
        guard.setattr(FAISS, "load_local", unexpected)
        with pytest.raises(docs.DocumentIdConflictError):
            docs.retrieve("Nanite")

    assert pointer.read_bytes() == published
    (docs.DOCS_DIR / "Guide.pdf").unlink()
    assert docs.retrieve("Nanite")[0]["doc_id"] == "Guide"
    assert pointer.read_bytes() == published


def test_metadata_registry_ignored_files_and_symlinks_do_not_create_conflicts(workspace):
    docs.DOCS_DIR.mkdir()
    reference = docs.DOCS_DIR / "metadata.md"
    reference.write_text("Nanite " * 200, encoding="utf-8")
    registry = docs.DOCS_DIR / "metadata.json"
    registry.write_text("{}", encoding="utf-8")
    (docs.DOCS_DIR / "metadata.txt").write_text("ignored", encoding="utf-8")
    (docs.DOCS_DIR / "metadata.pdf").symlink_to(reference)
    chunks = docs._chunks()

    assert len(chunks) > 1
    assert {chunk.metadata["doc_id"] for chunk in chunks} == {"metadata"}
    assert len({chunk.metadata["chunk_id"] for chunk in chunks}) == len(chunks)
    original_fingerprint = docs._fingerprint()
    registry.write_text('{"metadata.md": {"title": "New title", "engine_version": "5.5", "revision": 1}}',
                        encoding="utf-8")
    assert docs._fingerprint() != original_fingerprint
    assert {chunk.metadata["title"] for chunk in docs._chunks()} == {"New title"}
