"""언리얼 기술/오류 해결 사례를 로컬 JSON 문서로 저장한다."""

import json
import os
import re
import tempfile
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Literal
from uuid import uuid4

from filelock import FileLock
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "unreal"
DOCS_DIR = DATA_DIR / "docs"
LOCK_TIMEOUT = 60
_lock = threading.RLock()


@lru_cache(maxsize=1)
def _file_lock(root: Path) -> FileLock:
    # 같은 객체를 재사용해야 중첩 호출에서도 파일 잠금에 재진입할 수 있다.
    return FileLock(root / ".documents.lock")


@contextmanager
def locked():
    """같은 로컬 저장소의 조회·수정·인덱싱을 프로세스 간 직렬화한다."""
    deadline = time.monotonic() + LOCK_TIMEOUT
    if not _lock.acquire(timeout=LOCK_TIMEOUT):
        raise TimeoutError("문서 저장소 잠금 대기 시간을 초과했습니다.")
    try:
        root = DOCS_DIR.resolve().parent
        root.mkdir(parents=True, exist_ok=True)
        with _file_lock(root).acquire(timeout=max(0, deadline - time.monotonic())):
            yield
    finally:
        _lock.release()


class KnowledgeInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    title: str = Field(min_length=1, max_length=200)
    kind: Literal["technique", "error"]
    engine_version: str = Field(min_length=1, max_length=50)
    context: str = Field(default="", max_length=3000)
    content: str = Field(min_length=1, max_length=20000)
    reproduction: str = Field(default="", max_length=5000)
    solution: str = Field(default="", max_length=10000)
    source_url: HttpUrl | None = None

    @model_validator(mode="after")
    def require_solution(self):
        if self.kind == "error" and (not self.reproduction or not self.solution):
            raise ValueError("오류 사례에는 재현 방법과 해결 방법이 필요합니다.")
        return self


class KnowledgeRecord(KnowledgeInput):
    id: str = Field(pattern=r"^[0-9a-f]{32}$")
    created_at: str
    updated_at: str | None = None
    revision: int = Field(default=1, ge=1)


class KnowledgeUpdate(KnowledgeInput):
    expected_revision: int = Field(ge=1)


class RevisionConflict(Exception):
    """조회 이후 변경된 문서를 덮어쓰거나 삭제하지 않는다."""


def _path(entry_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", entry_id):
        raise FileNotFoundError("지식이 없습니다.")
    return DOCS_DIR / f"{entry_id}.json"


def read_record(path: Path) -> dict:
    record = KnowledgeRecord.model_validate_json(path.read_text(encoding="utf-8"))
    if record.id != path.stem:
        raise ValueError("지식 ID와 파일 이름이 일치하지 않습니다.")
    return record.model_dump(mode="json")


def _write(record: dict) -> dict:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    # 완성된 문서만 검색에 노출한다.
    fd, temporary = tempfile.mkstemp(dir=DOCS_DIR, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(record, file, ensure_ascii=False, indent=2)
        os.replace(temporary, DOCS_DIR / f"{record['id']}.json")
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return record


def save(entry: KnowledgeInput) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    record = KnowledgeRecord(**entry.model_dump(), id=uuid4().hex,
                             created_at=now, updated_at=now).model_dump(mode="json")
    with locked():
        return _write(record)


def update(entry_id: str, entry: KnowledgeUpdate) -> dict:
    with locked():
        current = read_record(_path(entry_id))
        if current["revision"] != entry.expected_revision:
            raise RevisionConflict()
        record = {**current, **entry.model_dump(mode="json", exclude={"expected_revision"}),
                  "updated_at": datetime.now(timezone.utc).isoformat(), "revision": current["revision"] + 1}
        return _write(record)


def delete(entry_id: str, expected_revision: int) -> None:
    with locked():
        path = _path(entry_id)
        current = read_record(path)
        if current["revision"] != expected_revision:
            raise RevisionConflict()
        path.unlink()


def list_entries() -> list[dict]:
    with locked():
        entries = [read_record(path) for path in DOCS_DIR.glob("*.json") if path.name != "metadata.json"]
    return sorted(entries, key=lambda entry: entry["created_at"], reverse=True)
