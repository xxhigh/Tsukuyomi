import json

import pytest
from pydantic import ValidationError

from agents import knowledge


def test_knowledge_persists_metadata_and_error_solution(workspace):
    entry = knowledge.KnowledgeInput(title="셰이더 컴파일 오류", kind="error", engine_version="5.5",
                                     context="Windows", content="컴파일 실패", reproduction="프로젝트 빌드",
                                     solution="팀에서 확인한 해결 절차", source_url="https://example.com/issue")
    record = knowledge.save(entry)
    assert knowledge.list_entries() == [record]
    path = knowledge.DOCS_DIR / f"{record['id']}.json"
    assert json.loads(path.read_text())["solution"] == entry.solution
    assert len(record["id"]) == 32
    assert not list(knowledge.DOCS_DIR.glob("*.tmp"))


@pytest.mark.parametrize("values", [
    {"kind": "error"}, {"title": "  "}, {"engine_version": "  "},
    {"source_url": "javascript:alert(1)"}, {"kind": "investment"},
])
def test_rejects_incomplete_or_invalid_knowledge(values):
    with pytest.raises(ValidationError):
        knowledge.KnowledgeInput.model_validate({"title": "Nanite", "kind": "technique",
                                                "engine_version": "5.5", "content": "설명", **values})


def test_update_preserves_identity_and_rejects_stale_writes(workspace):
    original = knowledge.save(knowledge.KnowledgeInput(title="초기 기록", kind="technique",
                                                        engine_version="5.5", content="기존 내용"))
    edit = knowledge.KnowledgeUpdate(title="수정 기록", kind="technique", engine_version="5.6",
                                     content="새로운 내용", expected_revision=1)
    updated = knowledge.update(original["id"], edit)
    assert updated["id"] == original["id"]
    assert updated["created_at"] == original["created_at"]
    assert updated["revision"] == 2
    assert updated["content"] == "새로운 내용"
    with pytest.raises(knowledge.RevisionConflict):
        knowledge.update(original["id"], edit)
    with pytest.raises(knowledge.RevisionConflict):
        knowledge.delete(original["id"], 1)
    assert knowledge.list_entries() == [updated]
    knowledge.delete(original["id"], 2)
    assert knowledge.list_entries() == []


def test_existing_record_without_revision_remains_editable(workspace):
    original = knowledge.save(knowledge.KnowledgeInput(title="이전 형식", kind="technique",
                                                        engine_version="5.5", content="기술 내용"))
    path = knowledge.DOCS_DIR / f"{original['id']}.json"
    original.pop("revision")
    original.pop("updated_at")
    path.write_text(json.dumps(original), encoding="utf-8")
    assert knowledge.list_entries()[0]["revision"] == 1
    knowledge.update(original["id"], knowledge.KnowledgeUpdate(title="수정", kind="technique",
                     engine_version="5.5", content="변경 내용", expected_revision=1))
    assert knowledge.list_entries()[0]["revision"] == 2


@pytest.mark.parametrize("entry_id", ["../metadata", "../outside", "metadata", "not-a-uuid"])
def test_delete_rejects_non_record_paths(workspace, entry_id):
    knowledge.DOCS_DIR.mkdir()
    metadata = knowledge.DOCS_DIR / "metadata.json"
    metadata.write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        knowledge.delete(entry_id, 1)
    assert metadata.read_text() == "{}"


def test_failed_atomic_update_leaves_previous_record(workspace, monkeypatch):
    original = knowledge.save(knowledge.KnowledgeInput(title="기존 기록", kind="technique",
                                                        engine_version="5.5", content="보존할 내용"))
    def fail(*args):
        raise OSError("write failure")
    monkeypatch.setattr(knowledge.os, "replace", fail)
    with pytest.raises(OSError):
        knowledge.update(original["id"], knowledge.KnowledgeUpdate(title="수정", kind="technique",
                         engine_version="5.5", content="변경 내용", expected_revision=1))
    assert knowledge.list_entries() == [original]
    assert not list(knowledge.DOCS_DIR.glob("*.tmp"))
