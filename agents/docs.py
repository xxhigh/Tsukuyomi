"""Unreal 문서의 bge-m3 임베딩, FAISS + bigram 검색. 투자 자료와 분리한다."""

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from functools import lru_cache
from uuid import uuid4

from filelock import FileLock
from langchain_core.documents import Document

from . import documents, knowledge

DOCS_DIR = knowledge.DOCS_DIR
STORE_DIR = knowledge.DATA_DIR / "vectorstores" / "BAAI--bge-m3"
EMBED_MODEL = "BAAI/bge-m3"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
KEYWORD_WEIGHT = 0.5
MIN_KEYWORD_SCORE = 0.2
CANDIDATES = 30
INDEX_FILES = {"index.faiss", "index.pkl"}
KEEP_GENERATIONS = 2
_logger = logging.getLogger(__name__)
_store = None
_store_fingerprint = None


class CorpusChangedError(RuntimeError):
    """질문 처리 중 검색 원본이 변경되어 답변을 다시 만들어야 한다."""


class DocumentIdConflictError(ValueError):
    """서로 다른 원본을 같은 검색 문서 ID로 처리하지 않는다."""

    def __init__(self):
        super().__init__("문서 ID가 중복되었습니다. 서버 문서 폴더의 PDF·Markdown 파일명이 "
                         "확장자를 제외해도 서로 다른 지식·문서 ID가 되도록 수정해 주세요.")


@lru_cache(maxsize=1)
def _embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL,
                                encode_kwargs={"normalize_embeddings": True})


def _bigrams(text: str) -> set[str]:
    text = re.sub(r"[^\w]", "", text.lower())
    return {text[index:index + 2] for index in range(len(text) - 1)}


def _keyword_score(query: str, text: str) -> float:
    grams = _bigrams(query)
    return len(grams & _bigrams(text)) / len(grams) if grams else 0.0


def _corpus_paths():
    paths = [path for path in sorted(DOCS_DIR.glob("*"))
             if path.suffix.lower() in {".pdf", ".md", ".json"} and not path.is_symlink() and path.is_file()]
    document_ids = set()
    for path in paths:
        if path.name != "metadata.json" and (path.suffix == ".json" or path.suffix.lower() in {".pdf", ".md"}):
            if path.stem in document_ids:
                raise DocumentIdConflictError()
            document_ids.add(path.stem)
    return paths


def _fingerprint() -> str:
    digest = hashlib.sha256(f"{EMBED_MODEL}:{CHUNK_SIZE}:{CHUNK_OVERLAP}".encode())
    for path in _corpus_paths():
        for value in (path.name.encode(), path.read_bytes()):
            digest.update(len(value).to_bytes(8, "big"))
            digest.update(value)
    return digest.hexdigest()


def _file_lock():
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    return FileLock(STORE_DIR / ".index.lock", timeout=60)


def _artifact_hashes(directory) -> dict[str, str]:
    hashes = {}
    for name in sorted(INDEX_FILES):
        path = directory / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("인덱스 파일 쌍을 확인할 수 없습니다.")
        with path.open("rb") as file:
            hashes[name] = hashlib.file_digest(file, "sha256").hexdigest()
    return hashes


def _read_manifest(verify=False) -> dict:
    manifest = json.loads((STORE_DIR / "current.json").read_text(encoding="utf-8"))
    generation = manifest.get("generation") if isinstance(manifest, dict) else None
    if not isinstance(generation, str) or not re.fullmatch(r"index-[0-9a-f]{32}", generation):
        raise ValueError("잘못된 인덱스 경로입니다.")
    if verify:
        _verify_artifacts(manifest)
    return manifest


def _verify_artifacts(manifest: dict) -> None:
    directory = STORE_DIR / manifest["generation"]
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("현재 인덱스 디렉터리를 확인할 수 없습니다.")
    if not manifest.get("files") or manifest["files"] != _artifact_hashes(directory):
        raise ValueError("현재 인덱스 검증에 실패했습니다. python -m agents.docs로 다시 생성하세요.")


def _generation_files(path):
    if path.is_symlink() or not path.is_dir():
        return None
    files = list(path.iterdir())
    if any(file.name not in INDEX_FILES or file.is_symlink() or not file.is_file() for file in files):
        return None
    return files


def _cleanup(keep: int, dry_run: bool) -> dict:
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ValueError("보관 개수는 1 이상인 정수여야 합니다.")
    current = _read_manifest(verify=True)["generation"]
    complete, incomplete, skipped = [], [], []
    for path in sorted(STORE_DIR.iterdir()):
        if not re.fullmatch(r"index-[0-9a-f]{32}", path.name) or path.name == current:
            continue
        files = _generation_files(path)
        if files is None:
            skipped.append(path.name)
            continue
        (complete if {file.name for file in files} == INDEX_FILES else incomplete).append(path)
    complete.sort(key=lambda path: (path.stat().st_mtime_ns, path.name), reverse=True)
    protected = [current, *(path.name for path in complete[:keep - 1])]
    candidates = [*complete[keep - 1:], *incomplete]
    report = {"dry_run": dry_run, "protected": protected, "candidates": [path.name for path in candidates],
              "removed": [], "failed": [], "skipped": skipped}
    if not dry_run:
        for path in candidates:
            try:
                # 원본이나 알 수 없는 파일에는 접근하지 않고 생성된 파일 쌍만 삭제한다.
                files = _generation_files(path)
                if files is None:
                    report["skipped"].append(path.name)
                    continue
                for file in files:
                    file.unlink()
                path.rmdir()
                report["removed"].append(path.name)
            except OSError:
                report["failed"].append(path.name)
    return report


def cleanup(keep: int = KEEP_GENERATIONS, dry_run: bool = True) -> dict:
    """현재 세대를 보호한다. 기본 동작은 삭제 없는 미리보기다."""
    with knowledge.locked(), _file_lock():
        return _cleanup(keep, dry_run)


def _chunks() -> list[Document]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = []
    source_metadata = documents.read_metadata(DOCS_DIR)

    def reference_metadata(path, title=None, page=0):
        entry = source_metadata.get(path.name, {})
        return {"doc_id": path.stem, "title": entry.get("title", title or path.stem), "kind": "reference",
                "engine_version": entry.get("engine_version", "미지정"),
                "url": entry.get("source_url") or "", "page": page}

    def append(text, metadata):
        for piece in splitter.split_text(text):
            chunks.append(Document(page_content=piece,
                                   metadata={**metadata, "chunk_id": len(chunks)}))

    for path in _corpus_paths():
        if path.name == "metadata.json":
            continue
        if path.suffix == ".json":
            record = knowledge.read_record(path)
            entry = knowledge.KnowledgeInput.model_validate(
                {key: value for key, value in record.items() if key in knowledge.KnowledgeInput.model_fields})
            metadata = {"doc_id": record["id"], "title": entry.title, "kind": entry.kind,
                        "engine_version": entry.engine_version, "url": str(entry.source_url or ""),
                        "date": record["updated_at"] or record["created_at"], "page": 0}
            text = (f"{entry.title}\nUE {entry.engine_version}\n환경: {entry.context}\n"
                    f"내용: {entry.content}\n재현: {entry.reproduction}\n해결: {entry.solution}")
            append(text, metadata)
        elif path.suffix.lower() == ".md":
            append(path.read_text(encoding="utf-8"), reference_metadata(path))
        elif path.suffix.lower() == ".pdf":
            import pymupdf
            with pymupdf.open(path) as pdf:
                for number, page in enumerate(pdf, 1):
                    text = page.get_text().strip()
                    if text:
                        append(text, reference_metadata(path, pdf.metadata.get("title"), number))
    return chunks


def _build(fingerprint: str):
    from langchain_community.vectorstores import FAISS
    global _store, _store_fingerprint
    chunks = _chunks()
    if not chunks:
        raise FileNotFoundError("등록된 언리얼 지식이 없습니다. 먼저 지식을 등록하세요.")
    store = FAISS.from_documents(chunks, _embeddings())
    # 완성된 세대만 pointer로 공개한다. 실패하면 이전 세대는 그대로 유지된다.
    generation = f"index-{uuid4().hex}"
    store.save_local(str(STORE_DIR / generation))
    pointer = STORE_DIR / f"{generation}.tmp"
    pointer.write_text(json.dumps({"generation": generation, "fingerprint": fingerprint,
                                  "files": _artifact_hashes(STORE_DIR / generation)}), encoding="utf-8")
    os.replace(pointer, STORE_DIR / "current.json")
    _store, _store_fingerprint = store, fingerprint
    try:
        report = _cleanup(KEEP_GENERATIONS, dry_run=False)
        if report["failed"]:
            _logger.warning("이전 인덱스 %s개를 정리하지 못했습니다.", len(report["failed"]))
    except (OSError, ValueError):
        _logger.warning("새 인덱스를 저장했으나 이전 세대 정리를 보류했습니다.")
    return store


def _load():
    from langchain_community.vectorstores import FAISS
    global _store, _store_fingerprint
    fingerprint = _fingerprint()
    if _store is not None and _store_fingerprint == fingerprint:
        return _store
    pointer = STORE_DIR / "current.json"
    if pointer.exists():
        manifest = _read_manifest()
        generation = manifest["generation"]
        if manifest["fingerprint"] == fingerprint:
            _verify_artifacts(manifest)
            # 이 앱이 직접 생성한 로컬 인덱스만 읽는다.
            _store = FAISS.load_local(str(STORE_DIR / generation), _embeddings(),
                                      allow_dangerous_deserialization=True)
            _store_fingerprint = fingerprint
            return _store
    return _build(fingerprint)


def _search_candidates(store, query: str, engine_version: str | None):
    embedding = store.embeddings.embed_query(query)
    options = {}
    if engine_version:
        # FAISS는 후보 추출 후 필터링하므로 전체에서 조회해야 버전별 후보를 놓치지 않는다.
        options = {"filter": lambda metadata: documents.version_matches(metadata.get("engine_version"), engine_version),
                   "fetch_k": store.index.ntotal}
    candidates = store.similarity_search_with_score_by_vector(
        embedding, k=min(CANDIDATES, store.index.ntotal), **options)
    seen = {(doc.metadata["doc_id"], doc.metadata["chunk_id"]) for doc, _ in candidates}
    lexical = []
    for position, document_id in store.index_to_docstore_id.items():
        doc = store.docstore.search(document_id)
        if engine_version and not documents.version_matches(doc.metadata.get("engine_version"), engine_version):
            continue
        keyword = _keyword_score(query, doc.page_content)
        if keyword >= MIN_KEYWORD_SCORE:
            lexical.append((keyword, position, doc))
    lexical.sort(key=lambda item: item[0], reverse=True)
    for _, position, doc in lexical[:CANDIDATES]:
        identity = (doc.metadata["doc_id"], doc.metadata["chunk_id"])
        if identity in seen:
            continue
        # 새 문자열 후보에도 같은 정규화 벡터의 실제 L2 거리를 사용한다.
        vector = store.index.reconstruct(position)
        distance = sum((float(value) - expected) ** 2 for value, expected in zip(vector, embedding))
        candidates.append((doc, distance))
        seen.add(identity)
    return candidates


def retrieve(query: str, k: int = 5, kind: str | None = None,
             engine_version: str | None = None) -> list[dict]:
    """관련 근거만 반환한다. 새 문서는 다음 검색에서 자동으로 인덱싱한다."""
    with knowledge.locked():
        if not any(path.suffix.lower() in {".json", ".md", ".pdf"} and path.name != "metadata.json"
                   and not path.is_symlink() and path.is_file()
                   for path in DOCS_DIR.glob("*")):
            return []
        with _file_lock():
            store = _load()
            engine_version = engine_version.strip() if engine_version else None
            candidates = _search_candidates(store, query, engine_version)
    cutoff = float(os.getenv("RAG_MIN_SIMILARITY", "0.45"))
    results = []
    for doc, distance in candidates:
        similarity = 1 - float(distance) / 2
        keyword = _keyword_score(query, doc.page_content)
        if similarity < cutoff and keyword < MIN_KEYWORD_SCORE:
            continue
        preferred = doc.metadata.get("kind") in {kind, "reference"}
        results.append({**doc.metadata, "content": doc.page_content,
                        "score": similarity + KEYWORD_WEIGHT * keyword + (0.05 if preferred else 0)})
    return sorted(results, key=lambda item: item["score"], reverse=True)[:max(1, min(k, 10))]


def search(query: str, k: int = 5) -> str:
    return json.dumps(retrieve(query, k), ensure_ascii=False)


def build() -> int:
    with knowledge.locked(), _file_lock():
        return _build(_fingerprint()).index.ntotal


def main() -> int:
    parser = argparse.ArgumentParser(description="언리얼 문서 인덱싱 및 이전 세대 정리")
    parser.add_argument("--cleanup", action="store_true", help="이전 세대 정리 목록 미리보기")
    parser.add_argument("--apply", action="store_true", help="--cleanup과 함께 사용해 삭제 실행")
    parser.add_argument("--keep", type=int, help="현재 세대를 포함한 보관 개수 (기본 2)")
    args = parser.parse_args()
    if (args.apply or args.keep is not None) and not args.cleanup:
        parser.error("--apply와 --keep은 --cleanup과 함께 사용하세요.")
    try:
        if args.cleanup:
            report = cleanup(KEEP_GENERATIONS if args.keep is None else args.keep, dry_run=not args.apply)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 1 if report["failed"] else 0
        print(f"문서 풀 인덱싱: {DOCS_DIR}")
        print(f"완료: chunk {build()}개 → {STORE_DIR}")
        return 0
    except (OSError, ValueError) as error:
        print(f"인덱스 작업 실패: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
