import pytest

from agents import docs, knowledge


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge, "DOCS_DIR", tmp_path / "docs")
    monkeypatch.setattr(docs, "DOCS_DIR", tmp_path / "docs")
    monkeypatch.setattr(docs, "STORE_DIR", tmp_path / "indexes")
    monkeypatch.setattr(docs, "_store", None)
    monkeypatch.setattr(docs, "_store_fingerprint", None)
    monkeypatch.setenv("RAG_MIN_SIMILARITY", "0.45")
    return tmp_path
