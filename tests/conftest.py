import pytest

from agents import docs, knowledge


@pytest.fixture(autouse=True)
def isolate_tavily_key(monkeypatch):
    # Tests must not use the server administrator's real Tavily credentials.
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setattr(knowledge, "DOCS_DIR", tmp_path / "docs")
    monkeypatch.setattr(docs, "DOCS_DIR", tmp_path / "docs")
    monkeypatch.setattr(docs, "STORE_DIR", tmp_path / "indexes")
    monkeypatch.setattr(docs, "_store", None)
    monkeypatch.setattr(docs, "_store_fingerprint", None)
    monkeypatch.setenv("RAG_MIN_SIMILARITY", "0.45")
    return tmp_path
