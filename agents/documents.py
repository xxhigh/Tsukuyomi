"""PDF·Markdown 원본을 유지하며 별도 메타데이터를 관리한다."""

import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from . import knowledge


class DocumentMetadata(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    # 여러 버전은 "5.4, 5.6"처럼 쉼표로 구분해 저장한다.
    engine_version: str = Field(min_length=1, max_length=200)
    source_url: HttpUrl | None = None


class DocumentUpdate(DocumentMetadata):
    expected_revision: int = Field(ge=0, strict=True)


class StoredMetadata(DocumentMetadata):
    revision: int = Field(ge=1, strict=True)


def version_matches(stored: str | None, wanted: str) -> bool:
    """쉼표로 구분된 버전 목록 중 하나라도 정확히 일치하면 True."""
    return wanted.strip() in {version.strip() for version in (stored or "").split(",")}


def _valid_filename(filename: str) -> bool:
    return (bool(filename) and "/" not in filename and "\\" not in filename and "\x00" not in filename
            and Path(filename).suffix.lower() in {".md", ".pdf"})


def read_metadata(directory: Path) -> dict:
    path = directory / "metadata.json"
    if path.is_symlink():
        raise ValueError("메타데이터 파일은 심볼릭 링크를 지원하지 않습니다.")
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or any(not _valid_filename(name) for name in data):
        raise ValueError("문서 메타데이터 형식을 확인해 주세요.")
    return {name: StoredMetadata.model_validate(value).model_dump(mode="json") for name, value in data.items()}


def _record(path: Path, metadata: dict) -> dict:
    return {"filename": path.name, "doc_id": path.stem, "title": path.stem,
            "engine_version": "미지정", "source_url": None, "revision": 0, **metadata.get(path.name, {})}


def list_documents() -> list[dict]:
    with knowledge.locked():
        metadata = read_metadata(knowledge.DOCS_DIR)
        return [_record(path, metadata) for path in sorted(knowledge.DOCS_DIR.glob("*"))
                if _valid_filename(path.name) and not path.is_symlink() and path.is_file()]


def update(filename: str, entry: DocumentUpdate) -> dict:
    with knowledge.locked():
        if not _valid_filename(filename):
            raise FileNotFoundError("문서를 찾을 수 없습니다.")
        path = knowledge.DOCS_DIR / filename
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError("문서를 찾을 수 없습니다.")
        metadata = read_metadata(knowledge.DOCS_DIR)
        if _record(path, metadata)["revision"] != entry.expected_revision:
            raise knowledge.RevisionConflict()
        metadata[filename] = {**entry.model_dump(mode="json", exclude={"expected_revision"}),
                              "revision": entry.expected_revision + 1}
        fd, temporary = tempfile.mkstemp(dir=knowledge.DOCS_DIR, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(metadata, file, ensure_ascii=False, indent=2)
            os.replace(temporary, knowledge.DOCS_DIR / "metadata.json")
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return _record(path, metadata)
