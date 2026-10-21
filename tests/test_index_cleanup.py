import json
import os
import subprocess
import sys
from pathlib import Path
from uuid import uuid4

import pytest
from filelock import FileLock

from agents import docs, knowledge
from tests.test_retrieval import LocalEmbeddings, add


def generation(timestamp=1):
    path = docs.STORE_DIR / f"index-{uuid4().hex}"
    path.mkdir(parents=True)
    for name in docs.INDEX_FILES:
        (path / name).write_bytes(name.encode())
    os.utime(path, ns=(timestamp, timestamp))
    return path


def publish(path):
    (docs.STORE_DIR / "current.json").write_text(json.dumps({
        "generation": path.name, "fingerprint": "test", "files": docs._artifact_hashes(path),
    }), encoding="utf-8")


def test_cleanup_dry_run_and_apply_preserve_current_and_latest_backup(workspace):
    old, backup, current = generation(1), generation(3), generation(2)
    publish(current)
    preview = docs.cleanup()
    assert preview["protected"] == [current.name, backup.name]
    assert preview["candidates"] == [old.name]
    assert preview["removed"] == []
    assert old.is_dir()
    applied = docs.cleanup(dry_run=False)
    assert applied["removed"] == [old.name]
    assert not old.exists()
    assert current.is_dir() and backup.is_dir()
    assert json.loads((docs.STORE_DIR / "current.json").read_text())["generation"] == current.name


def test_keep_one_cannot_remove_current_even_when_its_timestamp_is_oldest(workspace):
    current, newer = generation(1), generation(2)
    publish(current)
    assert docs.cleanup(keep=1, dry_run=False)["removed"] == [newer.name]
    assert current.is_dir()


@pytest.mark.parametrize("keep", [0, -1, True, 1.5])
def test_invalid_retention_never_deletes(workspace, keep):
    current, other = generation(), generation()
    publish(current)
    with pytest.raises(ValueError):
        docs.cleanup(keep=keep, dry_run=False)
    assert current.is_dir() and other.is_dir()


@pytest.mark.parametrize("problem", ["missing_manifest", "malformed", "traversal", "missing_file", "changed_file", "legacy"])
def test_unverifiable_current_index_prevents_all_deletion(workspace, problem):
    current, old = generation(), generation()
    publish(current)
    pointer = docs.STORE_DIR / "current.json"
    if problem == "missing_manifest":
        pointer.unlink()
    elif problem == "malformed":
        pointer.write_text("not JSON", encoding="utf-8")
    elif problem == "traversal":
        pointer.write_text(json.dumps({"generation": "../outside"}), encoding="utf-8")
    elif problem == "missing_file":
        (current / "index.pkl").unlink()
    elif problem == "changed_file":
        (current / "index.faiss").write_bytes(b"corrupted")
    else:
        pointer.write_text(json.dumps({"generation": current.name, "fingerprint": "test"}), encoding="utf-8")
    with pytest.raises((OSError, ValueError)):
        docs.cleanup(keep=1, dry_run=False)
    assert current.is_dir() and old.is_dir()


def test_unknown_files_symlinks_and_original_documents_are_preserved(workspace):
    current = generation()
    publish(current)
    unknown = generation()
    (unknown / "notes.md").write_text("keep me", encoding="utf-8")
    outside = workspace / "outside"
    outside.mkdir()
    (outside / "index.faiss").write_bytes(b"outside")
    link = docs.STORE_DIR / f"index-{uuid4().hex}"
    link.symlink_to(outside, target_is_directory=True)
    linked_file = generation()
    (linked_file / "index.faiss").unlink()
    (linked_file / "index.faiss").symlink_to(outside / "index.faiss")
    unrelated = docs.STORE_DIR / "manual-backup"
    unrelated.mkdir()
    original = add("Nanite")
    report = docs.cleanup(keep=1, dry_run=False)
    assert set(report["skipped"]) == {unknown.name, link.name, linked_file.name}
    assert report["removed"] == []
    assert (unknown / "notes.md").read_text() == "keep me"
    assert (outside / "index.faiss").read_bytes() == b"outside"
    assert unrelated.is_dir()
    assert knowledge.list_entries()[0]["id"] == original["id"]


def test_partial_build_is_removed_without_taking_a_backup_slot(workspace):
    current, backup, partial = generation(2), generation(1), generation(3)
    publish(current)
    (partial / "index.pkl").unlink()
    report = docs.cleanup(dry_run=False)
    assert report["protected"] == [current.name, backup.name]
    assert report["removed"] == [partial.name]


def test_failed_cleanup_is_reported_and_preserves_current(workspace, monkeypatch):
    current, old = generation(), generation()
    publish(current)
    unlink = Path.unlink
    def fail_old_file(path, *args, **kwargs):
        if path.parent == old:
            raise PermissionError("cannot remove")
        return unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, "unlink", fail_old_file)
    report = docs.cleanup(keep=1, dry_run=False)
    assert report["failed"] == [old.name]
    assert current.is_dir() and old.is_dir()


def test_real_faiss_rebuilds_keep_two_generations_and_survive_cleanup_failure(workspace, monkeypatch):
    monkeypatch.setattr(docs, "_embeddings", LocalEmbeddings)
    add("Nanite")
    for _ in range(3):
        docs.build()
    directories = [path for path in docs.STORE_DIR.glob("index-*") if path.is_dir()]
    assert len(directories) == 2
    def fail_cleanup(*args, **kwargs):
        raise OSError("cleanup failure")
    monkeypatch.setattr(docs, "_cleanup", fail_cleanup)
    docs.build()
    monkeypatch.setattr(docs, "_store", None)
    assert docs.retrieve("Nanite")[0]["title"] == "Nanite"


def test_failed_publish_never_prunes_previous_generations(workspace, monkeypatch):
    monkeypatch.setattr(docs, "_embeddings", LocalEmbeddings)
    add("Nanite")
    docs.build()
    previous = (docs.STORE_DIR / "current.json").read_text()
    old = generation()
    def fail_publish(*args):
        raise OSError("publish failure")
    monkeypatch.setattr(docs.os, "replace", fail_publish)
    with pytest.raises(OSError):
        docs.build()
    assert (docs.STORE_DIR / "current.json").read_text() == previous
    assert old.is_dir()


def test_changed_documents_can_rebuild_when_previous_generation_is_missing(workspace, monkeypatch):
    monkeypatch.setattr(docs, "_embeddings", LocalEmbeddings)
    add("Nanite")
    docs.build()
    current = docs.STORE_DIR / json.loads((docs.STORE_DIR / "current.json").read_text())["generation"]
    for file in current.iterdir():
        file.unlink()
    current.rmdir()
    add("shader")
    assert docs.retrieve("shader")[0]["title"] == "shader"


def test_index_file_lock_is_shared_across_processes(workspace):
    with docs._file_lock():
        script = ("from filelock import FileLock, Timeout\n"
                  "import sys\n"
                  "try:\n"
                  "    with FileLock(sys.argv[1], timeout=0.1): sys.exit(2)\n"
                  "except Timeout: sys.exit(0)\n")
        result = subprocess.run([sys.executable, "-c", script, str(docs.STORE_DIR / ".index.lock")],
                                capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    with FileLock(docs.STORE_DIR / ".index.lock", timeout=0.1):
        pass


def test_cleanup_cli_defaults_to_preview_and_apply_is_explicit(workspace, monkeypatch, capsys):
    current, old = generation(), generation()
    publish(current)
    monkeypatch.setattr(sys, "argv", ["agents.docs", "--cleanup", "--keep", "1"])
    assert docs.main() == 0
    assert json.loads(capsys.readouterr().out)["dry_run"] is True
    assert old.is_dir()
    monkeypatch.setattr(sys, "argv", ["agents.docs", "--cleanup", "--keep", "1", "--apply"])
    assert docs.main() == 0
    assert json.loads(capsys.readouterr().out)["removed"] == [old.name]
    assert current.is_dir()


@pytest.mark.parametrize("arguments", [["--apply"], ["--keep", "1"]])
def test_cleanup_flags_cannot_accidentally_start_a_rebuild(workspace, monkeypatch, arguments):
    monkeypatch.setattr(sys, "argv", ["agents.docs", *arguments])
    with pytest.raises(SystemExit) as error:
        docs.main()
    assert error.value.code == 2
    assert not docs.STORE_DIR.exists()
