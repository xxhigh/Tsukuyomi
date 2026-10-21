import pytest
from pydantic import ValidationError

from agents import docs, documents, knowledge


def source(name="Nanite 팀 노트.md"):
    knowledge.DOCS_DIR.mkdir(exist_ok=True)
    path = knowledge.DOCS_DIR / name
    path.write_text("Nanite team reference", encoding="utf-8")
    return path


def edit():
    return documents.DocumentUpdate(title="Nanite 참고 자료", engine_version="5.5",
                                    source_url="https://example.com/nanite", expected_revision=0)


def test_listing_and_metadata_updates_preserve_original_and_other_records(workspace):
    assert documents.list_documents() == []
    path = source()
    other = source("Other.PDF")
    (knowledge.DOCS_DIR / "ignored.txt").write_text("skip")
    original = path.read_bytes()
    initial = next(item for item in documents.list_documents() if item["filename"] == path.name)
    assert initial == {"filename": path.name, "doc_id": path.stem, "title": path.stem,
                       "engine_version": "미지정", "source_url": None, "revision": 0}
    updated = documents.update(path.name, edit())
    assert updated["revision"] == 1
    assert updated["doc_id"] == initial["doc_id"]
    documents.update(other.name, edit())
    second = documents.update(path.name, documents.DocumentUpdate(title="제목 수정", engine_version="미지정",
                             source_url=None, expected_revision=1))
    assert second["revision"] == 2
    assert len(documents.read_metadata(knowledge.DOCS_DIR)) == 2
    assert path.read_bytes() == original
    assert not list(knowledge.DOCS_DIR.glob("*.tmp"))
    assert knowledge.list_entries() == []


def test_conflicting_or_failed_save_preserves_previous_registry(workspace, monkeypatch):
    path = source()
    documents.update(path.name, edit())
    registry = knowledge.DOCS_DIR / "metadata.json"
    previous = registry.read_bytes()
    with pytest.raises(knowledge.RevisionConflict):
        documents.update(path.name, edit())
    def fail(*args):
        raise OSError("private failure details")
    monkeypatch.setattr(documents.os, "replace", fail)
    with pytest.raises(OSError):
        documents.update(path.name, documents.DocumentUpdate(title="수정", engine_version="5.6", expected_revision=1))
    assert registry.read_bytes() == previous
    assert not list(knowledge.DOCS_DIR.glob("*.tmp"))


@pytest.mark.parametrize("name", ["../outside.md", "..\\outside.pdf", "/tmp/outside.md", "note.json", "absent.md", "bad\x00.md"])
def test_rejects_missing_or_non_document_paths(workspace, name):
    source()
    with pytest.raises(FileNotFoundError):
        documents.update(name, edit())
    assert not (knowledge.DOCS_DIR / "metadata.json").exists()


@pytest.mark.parametrize("values", [{"title": "  "}, {"engine_version": ""}, {"source_url": "javascript:alert(1)"},
                                   {"expected_revision": -1}, {"expected_revision": True}, {"content": "unexpected"}])
def test_rejects_invalid_metadata(values):
    with pytest.raises(ValidationError):
        documents.DocumentUpdate.model_validate({"title": "Title", "engine_version": "5.5", "expected_revision": 0, **values})


@pytest.mark.parametrize("data", ["not json", "[]", '{"../outside.md": {}}', '{"Nanite 팀 노트.md": {"title": "incomplete"}}'])
def test_malformed_registry_is_not_silently_replaced(workspace, data):
    path = source()
    registry = knowledge.DOCS_DIR / "metadata.json"
    registry.write_text(data)
    with pytest.raises(ValueError):
        documents.list_documents()
    with pytest.raises(ValueError):
        documents.update(path.name, edit())
    with pytest.raises(ValueError):
        docs._chunks()
    assert registry.read_text() == data


def test_symlinked_source_and_registry_are_not_followed(workspace):
    path = source()
    linked = knowledge.DOCS_DIR / "linked.md"
    linked.symlink_to(path)
    assert len(documents.list_documents()) == 1
    assert len(docs._chunks()) == 1
    with pytest.raises(FileNotFoundError):
        documents.update(linked.name, edit())
    registry_target = workspace / "private.json"
    registry_target.write_text("{}")
    (knowledge.DOCS_DIR / "metadata.json").symlink_to(registry_target)
    with pytest.raises(ValueError):
        documents.update(path.name, edit())
    assert registry_target.read_text() == "{}"
