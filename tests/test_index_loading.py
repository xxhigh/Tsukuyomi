import json

import pytest

from agents import docs
from tests.test_retrieval import LocalEmbeddings, add


@pytest.fixture
def stored_index(workspace, monkeypatch):
    monkeypatch.setattr(docs, "_embeddings", LocalEmbeddings)
    record = add("Nanite")
    docs.build()
    pointer = docs.STORE_DIR / "current.json"
    manifest = json.loads(pointer.read_text())
    monkeypatch.setattr(docs, "_store", None)
    monkeypatch.setattr(docs, "_store_fingerprint", None)
    return record, pointer, docs.STORE_DIR / manifest["generation"]


def test_changed_pickle_is_rejected_even_when_still_deserializable(stored_index):
    _, pointer, generation = stored_index
    original_pointer = pointer.read_bytes()
    pickle_path = generation / "index.pkl"
    pickle_path.write_bytes(pickle_path.read_bytes() + b"changed trailing bytes")
    changed_pickle = pickle_path.read_bytes()

    with pytest.raises(ValueError):
        docs.retrieve("Nanite")

    assert pointer.read_bytes() == original_pointer
    assert pickle_path.read_bytes() == changed_pickle
    assert docs._store is None


def test_verified_index_loads_after_restart(stored_index):
    record, pointer, _ = stored_index
    original_pointer = pointer.read_bytes()

    assert docs.retrieve("Nanite")[0]["doc_id"] == record["id"]
    assert pointer.read_bytes() == original_pointer
    assert docs._store_fingerprint == docs._fingerprint()


@pytest.mark.parametrize("problem", [
    "changed_faiss", "changed_pickle", "missing_faiss", "missing_pickle",
    "linked_faiss", "linked_pickle", "linked_generation", "missing_generation",
    "legacy_manifest", "missing_hash", "wrong_hash",
])
def test_invalid_artifacts_never_reach_model_or_deserializer(stored_index, monkeypatch, problem):
    from langchain_community.vectorstores import FAISS

    _, pointer, generation = stored_index
    manifest = json.loads(pointer.read_text())
    filename = "index.faiss" if problem.endswith("faiss") else "index.pkl"
    artifact = generation / filename
    if problem.startswith("changed_"):
        artifact.write_bytes(artifact.read_bytes() + b"changed")
    elif problem.startswith("missing_") and problem != "missing_hash":
        if problem == "missing_generation":
            generation.rename(generation.with_name("preserved-generation"))
        else:
            artifact.unlink()
    elif problem.startswith("linked_"):
        target = generation.with_name("preserved-target")
        if problem == "linked_generation":
            generation.rename(target)
            generation.symlink_to(target, target_is_directory=True)
        else:
            artifact.rename(target)
            artifact.symlink_to(target)
    else:
        if problem == "legacy_manifest":
            manifest.pop("files")
        elif problem == "missing_hash":
            manifest["files"].pop("index.pkl")
        else:
            manifest["files"]["index.pkl"] = "0" * 64
        pointer.write_text(json.dumps(manifest))
    pointer_before = pointer.read_bytes()
    entries_before = sorted(path.name for path in docs.STORE_DIR.iterdir())

    def unexpected(*args, **kwargs):
        pytest.fail("검증 실패 후 모델 로드·역직렬화에 도달하면 안 됩니다")

    monkeypatch.setattr(docs, "_embeddings", unexpected)
    monkeypatch.setattr(FAISS, "load_local", unexpected)
    with pytest.raises(ValueError):
        docs.retrieve("Nanite")

    assert docs._store is None
    assert docs._store_fingerprint is None
    assert pointer.read_bytes() == pointer_before
    assert sorted(path.name for path in docs.STORE_DIR.iterdir()) == entries_before


def test_explicit_rebuild_recovers_from_invalid_artifacts(stored_index, monkeypatch):
    record, pointer, generation = stored_index
    artifact = generation / "index.pkl"
    artifact.write_bytes(artifact.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        docs.retrieve("Nanite")

    assert docs.build() > 0
    replacement = json.loads(pointer.read_text())
    assert replacement["generation"] != generation.name
    assert docs._artifact_hashes(docs.STORE_DIR / replacement["generation"]) == replacement["files"]
    monkeypatch.setattr(docs, "_store", None)
    assert docs.retrieve("Nanite")[0]["doc_id"] == record["id"]
    assert generation.is_dir()
