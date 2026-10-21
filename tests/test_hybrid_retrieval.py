import pytest
from langchain_core.embeddings import Embeddings

from agents import docs, documents

QUERY = "UE_BUILD_FAILURE_7391"


class IdentifierEmbeddings(Embeddings):
    """오류 코드가 포함된 자료의 벡터 순위를 의도적으로 31위로 만든다."""

    def __init__(self):
        self.queries = []

    @staticmethod
    def vector(text):
        if text == "zzzzzz":
            return [-1.0, 0.0]
        return [0.8, 0.6] if text.startswith("EXACT_SOURCE") else [1.0, 0.0]

    def embed_query(self, text):
        self.queries.append(text)
        return self.vector(text)

    def embed_documents(self, texts):
        return [self.vector(text) for text in texts]


@pytest.fixture
def library(workspace, monkeypatch):
    embeddings = IdentifierEmbeddings()
    monkeypatch.setattr(docs, "_embeddings", lambda: embeddings)
    docs.DOCS_DIR.mkdir()
    for number in range(docs.CANDIDATES):
        (docs.DOCS_DIR / f"note-{number}.md").write_text("DISTRACTOR unrelated reference", encoding="utf-8")
    (docs.DOCS_DIR / "exact.md").write_text(f"EXACT_SOURCE {QUERY}", encoding="utf-8")
    return embeddings


@pytest.mark.parametrize("version", [None, "5.5"])
@pytest.mark.parametrize("reload", [False, True])
def test_exact_error_code_outside_vector_candidates_can_rank_first(library, monkeypatch, version, reload):
    if version:
        for path in docs.DOCS_DIR.glob("*.md"):
            documents.update(path.name, documents.DocumentUpdate(
                title=path.stem, engine_version=version, expected_revision=0))
        (docs.DOCS_DIR / "000-wrong-version.md").write_text(f"EXACT_SOURCE {QUERY}", encoding="utf-8")
        documents.update("000-wrong-version.md", documents.DocumentUpdate(
            title="Wrong version", engine_version="5.6", expected_revision=0))
    if reload:
        docs.build()
        monkeypatch.setattr(docs, "_store", None)

    found = docs.retrieve(QUERY, k=1, engine_version=version)

    assert library.queries == [QUERY]
    assert [item["doc_id"] for item in found] == ["exact"]
    assert found[0]["score"] == pytest.approx(0.8 + docs.KEYWORD_WEIGHT + 0.05)
    if version:
        assert found[0]["engine_version"] == version


def test_overlapping_vector_and_keyword_candidates_are_not_duplicated(library, monkeypatch):
    monkeypatch.setattr(docs, "CANDIDATES", docs.CANDIDATES + 1)

    found = docs.retrieve(QUERY, k=10)

    assert found[0]["doc_id"] == "exact"
    identities = [(item["doc_id"], item["chunk_id"]) for item in found]
    assert len(identities) == len(set(identities)) == 10
    assert library.queries == [QUERY]


def test_each_candidate_channel_is_bounded_before_final_ranking(library, monkeypatch):
    monkeypatch.setattr(docs, "CANDIDATES", 3)
    for number in range(5):
        (docs.DOCS_DIR / f"exact-{number}.md").write_text(f"EXACT_SOURCE {QUERY}", encoding="utf-8")

    found = docs.retrieve(QUERY, k=10)

    assert len(found) == 6  # 벡터 3개 + 별도 문자열 후보 3개.
    assert sum(item["doc_id"].startswith("exact") for item in found) == 3
    assert library.queries == [QUERY]


def test_irrelevant_vector_and_keyword_results_are_still_rejected(library):
    assert docs.retrieve("zzzzzz") == []
    assert library.queries == ["zzzzzz"]
