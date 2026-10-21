import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from filelock import FileLock

from agents import docs, documents, knowledge
from tests.test_retrieval import LocalEmbeddings, add


WORKER = """
import json
import os
import sys
import time
from pathlib import Path
from agents import docs, documents, knowledge

root, operation, target, title, gate, timeout = sys.argv[1:]
root = Path(root)
knowledge.DOCS_DIR = docs.DOCS_DIR = root / "docs"
docs.STORE_DIR = root / "indexes"
knowledge.LOCK_TIMEOUT = float(timeout)
if gate:
    (root / f"ready-{os.getpid()}").touch()
    deadline = time.monotonic() + 15
    while not Path(gate).exists():
        if time.monotonic() > deadline:
            raise RuntimeError("start gate timeout")
        time.sleep(0.01)
try:
    if operation == "list":
        knowledge.list_entries()
    elif operation == "save":
        knowledge.save(knowledge.KnowledgeInput(title=title, kind="technique", engine_version="5.6", content=title))
    elif operation == "update":
        knowledge.update(target, knowledge.KnowledgeUpdate(title=title, kind="technique", engine_version="5.6",
                         content=title, expected_revision=1))
    elif operation == "delete":
        knowledge.delete(target, 1)
    elif operation == "documents":
        documents.list_documents()
    elif operation == "metadata":
        documents.update(target, documents.DocumentUpdate(title=title, engine_version="5.6", expected_revision=0))
    elif operation in {"retrieve", "build", "cleanup"}:
        def no_model():
            raise RuntimeError("model reached before storage lock")
        docs._embeddings = no_model
        if operation == "retrieve":
            docs.retrieve("nanite")
        elif operation == "build":
            docs.build()
        else:
            docs.cleanup()
    else:
        raise RuntimeError("unknown operation")
except knowledge.RevisionConflict:
    print("conflict")
except FileNotFoundError:
    print("missing")
except TimeoutError:
    print("blocked")
else:
    print("ok")
"""


@pytest.fixture
def worker(workspace):
    processes = []

    def start(operation, target="", title="shader", gate=None, timeout=5):
        process = subprocess.Popen([sys.executable, "-c", WORKER, str(workspace), operation, target, title,
                                    str(gate) if gate else "", str(timeout)],
                                   cwd=Path(__file__).resolve().parent.parent,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(process)
        return process

    yield start
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


def finish(process):
    output, errors = process.communicate(timeout=20)
    assert process.returncode == 0, errors
    return output.strip()


def release_gate(workspace, gate, count):
    deadline = time.monotonic() + 10
    while len(list(workspace.glob("ready-*"))) < count:
        assert time.monotonic() < deadline, "작업 프로세스가 준비되지 않았습니다."
        time.sleep(0.01)
    gate.touch()


@pytest.mark.parametrize("operation", ["list", "save", "update", "delete", "documents", "metadata",
                                       "retrieve", "build", "cleanup"])
def test_all_store_operations_share_the_process_lock(workspace, worker, operation):
    original = add("nanite")
    path = knowledge.DOCS_DIR / "reference.md"
    path.write_text("Original source", encoding="utf-8")
    before = docs._fingerprint()
    target = path.name if operation == "metadata" else original["id"]
    with knowledge.locked():
        assert finish(worker(operation, target, timeout=0.1)) == "blocked"
    assert docs._fingerprint() == before
    assert knowledge.list_entries() == [original]
    assert not docs.STORE_DIR.exists()


@pytest.mark.parametrize("operation", ["update", "metadata"])
def test_competing_process_updates_allow_only_one_revision(workspace, worker, operation):
    if operation == "update":
        target = add("nanite")["id"]
    else:
        knowledge.DOCS_DIR.mkdir()
        target = "reference.md"
        (knowledge.DOCS_DIR / target).write_text("Original source", encoding="utf-8")
    gate = workspace / "go"
    first = worker(operation, target, "first edit", gate)
    second = worker(operation, target, "second edit", gate)
    release_gate(workspace, gate, 2)
    assert sorted([finish(first), finish(second)]) == ["conflict", "ok"]
    current = knowledge.list_entries()[0] if operation == "update" else documents.list_documents()[0]
    assert current["revision"] == (2 if operation == "update" else 1)
    assert current["title"] in {"first edit", "second edit"}


def test_update_and_delete_of_the_same_revision_cannot_both_succeed(workspace, worker):
    original = add("nanite")
    gate = workspace / "go"
    editing = worker("update", original["id"], gate=gate)
    deleting = worker("delete", original["id"], gate=gate)
    release_gate(workspace, gate, 2)
    # 삭제가 먼저면 수정할 문서가 없고, 수정이 먼저면 삭제는 revision 충돌이다.
    update_status = finish(editing)
    delete_status = finish(deleting)
    if delete_status == "ok":
        assert update_status == "missing"
        assert knowledge.list_entries() == []
    else:
        assert delete_status == "conflict"
        assert update_status == "ok"
        assert knowledge.list_entries()[0]["revision"] == 2


def test_parallel_metadata_updates_preserve_both_files(workspace, worker):
    knowledge.DOCS_DIR.mkdir()
    for filename in ("one.md", "two.md"):
        (knowledge.DOCS_DIR / filename).write_text("Original reference", encoding="utf-8")
    gate = workspace / "go"
    first = worker("metadata", "one.md", "First source", gate)
    second = worker("metadata", "two.md", "Second source", gate)
    release_gate(workspace, gate, 2)
    assert [finish(first), finish(second)] == ["ok", "ok"]
    metadata = documents.read_metadata(knowledge.DOCS_DIR)
    assert metadata["one.md"]["title"] == "First source"
    assert metadata["two.md"]["title"] == "Second source"
    assert {item["revision"] for item in metadata.values()} == {1}
    assert all(path.read_text() == "Original reference" for path in knowledge.DOCS_DIR.glob("*.md"))


def test_nested_lock_releases_after_exception(workspace, worker):
    with pytest.raises(RuntimeError):
        with knowledge.locked(), knowledge.locked():
            raise RuntimeError("operation failed")
    assert finish(worker("list", timeout=0.1)) == "ok"


def test_thread_timeout_and_file_timeout_leave_lock_usable(workspace, monkeypatch):
    monkeypatch.setattr(knowledge, "LOCK_TIMEOUT", 0.05)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with knowledge.locked():
            with pytest.raises(TimeoutError):
                executor.submit(knowledge.list_entries).result(timeout=2)
        with FileLock(workspace / ".documents.lock"):
            with pytest.raises(TimeoutError):
                knowledge.list_entries()
        assert executor.submit(knowledge.list_entries).result(timeout=2) == []


def test_index_build_blocks_edits_and_later_search_reloads_external_changes(workspace, worker, monkeypatch):
    original = add("nanite")
    def embeddings():
        assert finish(worker("update", original["id"], timeout=0.1)) == "blocked"
        return LocalEmbeddings()
    monkeypatch.setattr(docs, "_embeddings", embeddings)
    docs.build()
    previous = json.loads((docs.STORE_DIR / "current.json").read_text())["generation"]
    assert finish(worker("update", original["id"])) == "ok"
    monkeypatch.setattr(docs, "_embeddings", LocalEmbeddings)
    found = docs.retrieve("shader", engine_version="5.6")
    assert found[0]["doc_id"] == original["id"]
    assert found[0]["title"] == "shader"
    assert docs.retrieve("nanite", engine_version="5.5") == []
    assert json.loads((docs.STORE_DIR / "current.json").read_text())["generation"] != previous
