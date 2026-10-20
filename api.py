"""로컬 내부용 Unreal RAG API. Vue 개발 서버는 /api 요청을 이 서버로 전달한다."""

import os
import re
from hmac import compare_digest
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from markdown_it import MarkdownIt
from pydantic import BaseModel, ConfigDict, Field

load_dotenv()

from agents import build_graph, documents, knowledge, trace  # noqa: E402
from agents.docs import CorpusChangedError, DocumentIdConflictError  # noqa: E402
from agents.graph import run_config  # noqa: E402
from agents.llm import check_connection, create_llm  # noqa: E402
from openai import APIConnectionError, APIStatusError, AuthenticationError, RateLimitError  # noqa: E402

_bearer = HTTPBearer(auto_error=False, scheme_name="InternalAccess")


def require_internal_access(credentials: HTTPAuthorizationCredentials | None = Depends(_bearer)):
    expected = os.getenv("INTERNAL_API_TOKEN", "").strip()
    if not expected:
        return
    supplied = credentials.credentials if credentials else ""
    if len(supplied) > 4096 or not compare_digest(supplied.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(401, detail={"code": "internal_access_required",
                            "message": "내부 접근 토큰을 확인한 뒤 다시 연결해 주세요."},
                            headers={"WWW-Authenticate": "Bearer"})


app = FastAPI(title="Tsukuyomi for Unreal", version="0.1.0", dependencies=[Depends(require_internal_access)])


class Message(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=8000)


class LLMCheckRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    model: str | None = Field(default=None, min_length=1, max_length=100)


class ChatRequest(LLMCheckRequest):
    question: str = Field(min_length=1, max_length=8000)
    history: list[Message] = Field(default_factory=list, max_length=10)
    engine_version: str | None = Field(default=None, min_length=1, max_length=50)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, error: RequestValidationError):
    # 검증 오류에 입력 본문·헤더·키가 되돌아가지 않도록 한다.
    return JSONResponse(status_code=422, content={"detail": "입력 항목을 확인해 주세요.",
                        "fields": [".".join(map(str, item["loc"])) for item in error.errors()]})


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/config")
def config():
    return {"model": os.getenv("OPENAI_MODEL", "gpt-5-mini"),
            "has_server_key": bool(os.getenv("OPENAI_API_KEY", "").strip())}


@app.get("/api/knowledge")
def list_knowledge():
    try:
        entries = knowledge.list_entries()
    except (ValueError, OSError, KeyError, TypeError):
        raise HTTPException(503, "저장된 지식 파일을 읽을 수 없습니다. 서버 문서를 확인해 주세요.") from None
    return {"entries": entries, "count": len(entries)}


@app.post("/api/knowledge", status_code=201)
def save_knowledge(entry: knowledge.KnowledgeInput):
    try:
        record = knowledge.save(entry)
    except OSError:
        raise HTTPException(503, "지식을 저장하지 못했습니다. 저장 경로 권한을 확인해 주세요.") from None
    return {"entry": record, "message": "저장했습니다. 다음 질문에서 검색 인덱스에 반영합니다."}


@app.put("/api/knowledge/{entry_id}")
def update_knowledge(entry_id: str, entry: knowledge.KnowledgeUpdate):
    try:
        record = knowledge.update(entry_id, entry)
    except FileNotFoundError:
        raise HTTPException(404, "지식을 찾을 수 없습니다. 목록을 새로고침해 주세요.") from None
    except knowledge.RevisionConflict:
        raise HTTPException(409, "다른 작업에서 지식이 변경되었습니다. 목록을 새로고침한 뒤 다시 편집해 주세요.") from None
    except (ValueError, OSError):
        raise HTTPException(503, "지식을 수정하지 못했습니다. 저장 파일과 경로 권한을 확인해 주세요.") from None
    return {"entry": record, "message": "수정했습니다. 다음 질문에서 검색 인덱스에 반영합니다."}


@app.delete("/api/knowledge/{entry_id}", status_code=204)
def delete_knowledge(entry_id: str, expected_revision: int = Query(ge=1)):
    try:
        knowledge.delete(entry_id, expected_revision)
    except FileNotFoundError:
        raise HTTPException(404, "지식을 찾을 수 없습니다. 목록을 새로고침해 주세요.") from None
    except knowledge.RevisionConflict:
        raise HTTPException(409, "다른 작업에서 지식이 변경되었습니다. 목록을 새로고침한 뒤 다시 삭제해 주세요.") from None
    except (ValueError, OSError):
        raise HTTPException(503, "지식을 삭제하지 못했습니다. 저장 파일과 경로 권한을 확인해 주세요.") from None
    return Response(status_code=204)


@app.get("/api/documents")
def list_documents():
    try:
        entries = documents.list_documents()
    except (ValueError, OSError):
        raise HTTPException(503, "문서 목록과 메타데이터를 읽을 수 없습니다. 서버 파일을 확인해 주세요.") from None
    return {"entries": entries, "count": len(entries)}


@app.put("/api/documents/{filename}")
def update_document(filename: str, entry: documents.DocumentUpdate):
    try:
        record = documents.update(filename, entry)
    except FileNotFoundError:
        raise HTTPException(404, "문서를 찾을 수 없습니다. 목록을 새로고침해 주세요.") from None
    except knowledge.RevisionConflict:
        raise HTTPException(409, "다른 작업에서 문서 정보가 변경되었습니다. 목록을 새로고침한 뒤 다시 편집해 주세요.") from None
    except (ValueError, OSError):
        raise HTTPException(503, "문서 정보를 저장하지 못했습니다. 메타데이터와 경로 권한을 확인해 주세요.") from None
    return {"entry": record, "message": "문서 정보를 저장했습니다. 다음 질문에서 검색에 반영합니다."}


def _llm_key(value: str | None) -> str:
    api_key = (value or os.getenv("OPENAI_API_KEY", "")).strip()
    if not api_key:
        raise HTTPException(400, "연결 설정에서 LLM API 키를 입력해 주세요.")
    if len(api_key) > 4096:
        raise HTTPException(400, "API 키 길이를 확인해 주세요.")
    return api_key


@app.post("/api/llm/check")
def check_llm(payload: LLMCheckRequest, x_llm_api_key: str | None = Header(default=None)):
    api_key = _llm_key(x_llm_api_key)
    try:
        check_connection(create_llm(api_key, payload.model))
    except AuthenticationError:
        raise HTTPException(401, "LLM API 키가 거부되었습니다. 연결 설정을 확인해 주세요.") from None
    except RateLimitError:
        raise HTTPException(429, "LLM 사용 한도를 확인한 뒤 다시 요청해 주세요.") from None
    except (APIConnectionError, APIStatusError):
        raise HTTPException(502, "외부 LLM 연결과 모델의 tool calling 지원을 확인해 주세요.") from None
    except (ValueError, TypeError, KeyError):
        raise HTTPException(502, "모델이 올바른 tool calling 응답을 반환하지 않았습니다. 모델 설정을 확인해 주세요.") from None
    except Exception:
        raise HTTPException(503, "LLM 연결을 확인하지 못했습니다. 서버 설정을 확인해 주세요.") from None
    return {"status": "ok", "message": "LLM 키·모델과 tool calling 응답을 확인했습니다."}


def _answer_blocks(text: str) -> list[dict]:
    # Markdown의 줄 번호와 맞추되 코드 밖 원문과 줄바꿈은 유지한다.
    lines = re.findall(r"[^\r\n]*(?:\r\n|\r|\n)|[^\r\n]+$", text)
    blocks, cursor = [], 0
    for token in MarkdownIt("commonmark", {"html": False}).parse(text):
        if token.type not in {"fence", "code_block"} or not token.map:
            continue
        start, end = token.map
        before = "".join(lines[cursor:start])
        if before:
            blocks.append({"type": "text", "content": before})
        language = token.info.split(maxsplit=1)
        blocks.append({"type": "code", "content": token.content,
                       "language": language[0][:50] if language else ""})
        cursor = end
    if blocks:
        after = "".join(lines[cursor:])
        if after:
            blocks.append({"type": "text", "content": after})
    return blocks


@app.post("/api/chat")
def chat(payload: ChatRequest, x_llm_api_key: str | None = Header(default=None)):
    api_key = _llm_key(x_llm_api_key)
    run_id = uuid4().hex
    try:
        llm = create_llm(api_key, payload.model)
        graph = build_graph(llm)
        result = graph.invoke({"question": payload.question,
                               "engine_version": payload.engine_version,
                               "history": [message.model_dump() for message in payload.history]},
                              # LangSmith trace ID를 응답의 run_id와 맞춰 로컬 트레이스와 연결한다.
                              config=run_config("api", run_id=UUID(run_id), engine_version=payload.engine_version))
    except CorpusChangedError:
        raise HTTPException(409, "답변을 만들던 중 지식이 변경되었습니다. 같은 질문을 다시 보내 주세요.") from None
    except DocumentIdConflictError as error:
        raise HTTPException(503, str(error)) from None
    except AuthenticationError:
        raise HTTPException(401, "LLM API 키가 거부되었습니다. 연결 설정을 확인해 주세요.") from None
    except RateLimitError:
        raise HTTPException(429, "LLM 사용 한도를 확인한 뒤 다시 요청해 주세요.") from None
    except (APIConnectionError, APIStatusError):
        raise HTTPException(502, "외부 LLM 연결 또는 모델 설정을 확인해 주세요.") from None
    except Exception:
        # 제공자 오류에는 키나 요청 내용이 포함될 수 있으므로 원문을 노출하지 않는다.
        raise HTTPException(503, "답변을 생성하지 못했습니다. LLM 설정과 문서 인덱싱 환경을 확인해 주세요.") from None
    # 질문·답변·키·검색 본문은 trace에 저장하지 않는다.
    try:
        trace.log(run_id, "chat_done", status=result["status"], rounds=result.get("rounds", 0),
                  workers=[decision["worker"] for decision in result["decisions"]])
    except OSError:
        pass  # 부가적인 trace 저장 실패 때문에 완성된 답변을 잃지 않는다.
    return {"run_id": run_id, "answer": result["answer"], "answer_blocks": _answer_blocks(result["answer"]),
            "sources": result["sources"],
            "engine_version": payload.engine_version,
            "status": result["status"], "decisions": result["decisions"]}


# 빌드된 Vue를 동일 출처에서 제공한다. 개발 시에는 Vite proxy를 사용한다.
dist = Path(__file__).resolve().parent / "frontend" / "dist"
if dist.is_dir():
    app.mount("/", StaticFiles(directory=dist, html=True), name="frontend")
