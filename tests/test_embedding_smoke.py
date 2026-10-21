"""Optional real-model check; see README for the offline opt-in command."""

import math
import os

import pytest

from agents import docs, knowledge

pytestmark = pytest.mark.skipif(os.getenv("RUN_BGE_SMOKE") != "1", reason="실제 bge-m3 검증은 선택 실행")


def test_real_bge_embeddings_and_versioned_faiss_round_trip(workspace, monkeypatch):
    assert os.getenv("HF_HUB_OFFLINE") == "1", "캐시만 사용하도록 HF_HUB_OFFLINE=1을 설정하세요."
    docs._embeddings.cache_clear()
    try:
        embeddings = docs._embeddings()
        query = "Nanite 오프라인 임베딩 통합 검증"
        vector = embeddings.embed_query(query)
        assert len(vector) > 0
        assert all(math.isfinite(value) for value in vector)
        assert math.sqrt(sum(value * value for value in vector)) == pytest.approx(1.0, abs=1e-5)

        first = knowledge.save(knowledge.KnowledgeInput(
            title=query, kind="technique", engine_version="smoke-A", content=query))
        second = knowledge.save(knowledge.KnowledgeInput(
            title=query, kind="technique", engine_version="smoke-B", content=query))
        found = docs.retrieve(query, engine_version="smoke-A")
        assert [item["doc_id"] for item in found] == [first["id"]]
        assert all(math.isfinite(item["score"]) for item in found)
        assert docs._store.index.d == len(vector)
        stored_vector = docs._store.index.reconstruct(0)
        assert all(math.isfinite(value) for value in stored_vector)
        assert math.sqrt(sum(float(value) ** 2 for value in stored_vector)) == pytest.approx(1.0, abs=1e-5)
        assert docs.retrieve(query, engine_version="absent") == []
        assert [item["doc_id"] for item in docs.retrieve(query, engine_version="smoke-B")] == [second["id"]]

        # FAISS 파일을 실제로 다시 읽고 임베딩 어댑터와의 검색 호환성을 확인한다.
        monkeypatch.setattr(docs, "_store", None)
        assert [item["doc_id"] for item in docs.retrieve(query, engine_version="smoke-A")] == [first["id"]]
        fingerprint = docs._store_fingerprint
        knowledge.update(first["id"], knowledge.KnowledgeUpdate(
            title=query, kind="technique", engine_version="smoke-C", content=query, expected_revision=1))
        assert docs.retrieve(query, engine_version="smoke-A") == []
        assert [item["doc_id"] for item in docs.retrieve(query, engine_version="smoke-C")] == [first["id"]]
        assert docs._store_fingerprint != fingerprint
        print(f"Real bge-m3 check passed: {len(vector)} dimensions, normalized vectors, "
              "FAISS save/reload, version filters and reindexing.")
    finally:
        docs._embeddings.cache_clear()
