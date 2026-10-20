# Tsukuyomi for Unreal

언리얼 엔진 기술과 개발 오류·해결 사례를 저장하고 검색하는 내부용 RAG 챗봇입니다.


## 구성

- **Vue.js**: 대화, 요청별 LLM 키 입력, 기술·오류 해결 사례 관리, PDF·Markdown 정보 편집.
- **FastAPI**: `/api/chat`, `/api/llm/check`, `/api/knowledge`, `/api/knowledge/{id}`, `/api/documents`, `/api/documents/{filename}`, `/api/config`, `/api/health`.
- **Supervisor**: 질문에 따라 기술/오류 검색 담당을 선택하고 최대 2회 검색 후 답변합니다.
- **RAG**: 로컬 `BAAI/bge-m3` 정규화 임베딩 + FAISS + bigram 재정렬.
- **외부 LLM**: OpenAI 또는 tool calling을 지원하는 OpenAI 호환 chat completions API.

```text
Vue → FastAPI → Supervisor ⇄ 기술 검색 / 오류 검색 → 근거 기반 답변
                    ↓              ↓
               외부 LLM       data/unreal/docs
                               bge-m3 + FAISS
```

## 실행

Python 3.11 이상, Node.js 22.12 이상을 사용합니다. 프로젝트 루트에서:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn api:app --host 127.0.0.1 --port 8000
```

다른 터미널에서:

```bash
cd frontend
npm ci
npm run dev
```

Vite가 출력하는 주소(기본 `http://127.0.0.1:5173`)를 엽니다. 사이드바의 **설정 → LLM 연결**에서 API 키를 입력하거나 `.env`에 `OPENAI_API_KEY`를 설정합니다. `OPENAI_MODEL`로 기본 모델을 변경할 수 있습니다. 다른 호환 제공자는 서버의 `OPENAI_BASE_URL`로 지정합니다. 제공자별 tool calling 호환성은 별도 확인이 필요합니다.

**설정 → LLM 연결 → LLM 연결 확인**을 누르면 키·모델과 Supervisor의 `Route` tool calling 응답 형식을 검사합니다. 입력한 키를 우선 사용하고, 비어 있으면 서버 키를 사용합니다. 확인은 선택 사항이며 자동으로 실행하지 않습니다. 누를 때마다 외부 LLM 사용량이 발생할 수 있고, SDK는 일시적 오류를 한 번 재시도할 수 있습니다.

검사에는 고정된 짧은 요청만 보내며 팀 문서·질문·대화와 임베딩을 사용하지 않습니다. 결과와 키는 트레이스에 저장하지 않습니다. 키·모델을 바꾸거나 내부 연결을 해제하면 이전 결과를 지웁니다. 임베딩 준비와 RAG 답변 품질은 별도로 검증합니다.

API는 `POST /api/llm/check`에 `{"model": "모델 이름"}` 또는 `{}`를 보냅니다. LLM 키는 `X-LLM-API-Key`, 내부 토큰은 기존 `Authorization` 헤더를 사용합니다. 성공은 `200`, LLM 키 거부는 `401`, 사용 한도는 `429`, 제공자 연결/모델/응답 형식 문제는 `502`로 반환합니다. 요청의 `tools`·`tool_choice` 형식은 [OpenAI Docs](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)를 참고했습니다.

프런트엔드를 빌드한 뒤 API 서버를 재시작하면 `http://127.0.0.1:8000`에서 함께 제공합니다.

```bash
npm --prefix frontend run build
```

CLI도 사용할 수 있습니다. 키가 없으면 숨김 입력으로 받습니다.

```bash
python app.py
python app.py "UE 5.5 셰이더 오류 관련 기록을 찾아줘"
python app.py --engine-version 5.5 "셰이더 오류 관련 기록을 찾아줘"
```

`--engine-version`은 등록 버전과 정확히 일치하는 자료만 검색합니다. 생략하면 전체 검색이며 `python app.py --engine-version 5.5`로 대화 모드를 시작하면 세션 동안 같은 필터를 유지합니다. 버전은 앞뒤 공백 제거 후 1~50자여야 합니다. `python app.py --help`는 키 입력 없이 도움말을 표시하며, `--`로 시작하는 질문은 `python app.py -- "--로 시작하는 질문"`처럼 전달합니다.

## LangSmith 추적

Supervisor·검색·답변은 LangChain/LangGraph로 실행되므로 `.env`에 아래를 설정하고 서버나 CLI를 다시 시작하면 LangSmith에 기록됩니다. 기본값은 꺼져 있습니다.

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=tsukuyomi-for-unreal
```

- 실행 이름은 `tsukuyomi_chat`이며 `api`/`cli` 태그와 `engine_version` metadata가 붙습니다. 각 노드(`supervisor`, `technique`, `troubleshooting`, `answer`) 아래에 `supervisor_prompt`/`writer_prompt`와 LLM 호출이 표시됩니다. 연결 확인은 `llm_connection_check`로 기록됩니다.
- API 응답의 `run_id`가 LangSmith trace ID입니다. 하이픈 없는 UUID이므로 LangSmith에서는 하이픈을 넣은 형식으로 검색합니다.
- **켜면 질문·대화 이력·검색 원문(팀 문서 내용)·답변이 LangSmith 서버로 전송됩니다.** LLM API 키와 내부 접근 토큰은 기록되지 않습니다. 실행 구조와 시간만 남기려면 `LANGSMITH_HIDE_INPUTS=true`, `LANGSMITH_HIDE_OUTPUTS=true`를 함께 설정합니다.

## 내부 API 접근 설정

`.env`의 `INTERNAL_API_TOKEN`에 임의의 공유 토큰을 설정하고 서버를 재시작하면 모든 `/api/*` 요청에 접근 확인을 적용합니다. health/config 조회도 포함합니다. 빈 값이면 기존 로컬 개발 방식으로 실행합니다.

토큰은 아래처럼 생성할 수 있습니다. 생성한 값은 `.env`와 허용된 팀원에게만 전달합니다.

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

사이드바의 **설정 → 내부 연결**에서 **내부 접근 토큰**을 입력하고 **연결**을 누릅니다. 토큰이 필요하거나 거부되면 설정 화면이 자동으로 열립니다. API 호출에는 `Authorization: Bearer <내부 접근 토큰>`을 사용합니다. 외부 LLM 키는 기존 `X-LLM-API-Key` 헤더로 별도 전달하며 내부 토큰을 LLM에 전달하지 않습니다. `/docs`에서는 `InternalAccess` 인증 설정을 사용할 수 있습니다.

설정·대화·지식 저장소 메뉴를 이동해도 키·모델·대화와 작성 중인 지식·문서 정보는 유지됩니다. 검색 엔진 버전은 사이드바에서 선택합니다. 키와 토큰은 브라우저 저장소에 저장하지 않습니다.

**연결 해제**와 재연결은 현재 화면의 키·대화·지식 목록·편집 상태를 비웁니다. 접근 토큰도 메모리에만 보관하며 새로고침하면 지워집니다. 서버의 공유 토큰 자체는 유지되며, 교체하려면 서버 설정을 바꾸고 재시작합니다. CLI와 서버 파일에 대한 로컬 접근은 이 HTTP 인증의 적용 대상에 포함하지 않습니다.

현재는 공유 토큰을 통한 기본 API 접근 확인을 제공합니다. 프런트엔드 정적 화면과 API 스키마는 열람할 수 있습니다. 팀 배포 전 HTTPS, SSO와 사용자별 권한 정책을 정해야 하며, 기본 실행 주소는 계속 loopback입니다.

## 지식 등록과 검색

화면의 **검색 엔진 버전**에 `5.5` 등을 입력하면 등록 버전과 정확히 일치하는 자료만 검색합니다. 비우면 전체를 검색합니다. `5.5`와 `5.5.4`, `UE 5.5`는 서로 다른 값이며 호환성을 자동으로 판단하지 않습니다. 버전이 미지정인 PDF/Markdown 자료는 특정 버전 검색에서 제외됩니다. 해당 버전의 근거가 없으면 다른 버전 자료로 대체하지 않습니다.

검색은 벡터 후보와 문자쌍(bigram) 기반 문자열 후보를 각각 최대 30개 구해 중복을 합친 뒤 점수를 계산합니다. 오류 코드·API 이름이 벡터 후보 밖에 있어도 문자열 후보로 검색할 수 있습니다. 두 경로에 같은 엔진 버전 필터를 적용하며, 쿼리 임베딩은 한 번 계산해 재사용합니다.

각 대화에는 검색 버전을 표시하며, 선택한 버전이 다른 대화 이력은 후속 요청에 전달하지 않습니다. API에서는 `/api/chat` 요청의 선택 필드 `engine_version`으로 지정합니다. 생략하거나 `null`을 보내면 전체 검색이며, CLI는 선택 옵션 `--engine-version`을 사용합니다.

질문은 최대 8,000자까지 입력할 수 있습니다. Supervisor가 첫 검색을 생략하려 하면 질문 원문으로 검색하며, 이때 긴 질문을 별도의 검색어 길이 제한으로 거부하거나 잘라내지 않습니다. API에서 8,000자를 넘는 질문은 `422`로 반환합니다.

답변의 출처 아래 **검색 원문 보기**를 펼치면 답변 당시 검색된 원문을 최대 1,000자까지 확인할 수 있습니다. 출처 번호 검증을 통과한 인용 자료만 표시하며, 긴 원문은 잘림 안내를 표시합니다. 엔진 버전과 PDF 쪽수를 함께 확인하세요. 원문은 모델이 생성한 요약이 아니며 HTML·스크립트도 일반 텍스트로 표시합니다. 출처 번호 검증 자체가 답변의 사실성을 보증하지는 않습니다.

코드 예시는 코드 블록이나 인라인 백틱으로 감싸고, 출처는 코드·링크 밖의 설명 문장에 `[1]`, `[2]`처럼 표기합니다. 코드의 배열 인덱스 `Values[0]`이나 링크 안의 번호는 인용으로 계산하지 않습니다. [markdown-it-py 토큰 스트림](https://markdown-it-py.readthedocs.io/en/latest/using.html#the-token-stream)으로 설명 문장의 번호를 검사하며, 범위 밖 번호나 `[01]`처럼 형식이 다른 번호는 거부합니다. 답변 본문은 모델이 반환한 형식을 유지합니다.

웹 화면은 fenced/들여쓰기 코드 블록을 언어 표시·들여쓰기·가로 스크롤과 함께 보여줍니다. 코드 영역은 키보드로도 포커스할 수 있으며 HTML·스크립트도 일반 텍스트로 표시합니다. `/api/chat`의 `answer_blocks`는 표시용 `text`/`code` 블록과 선택적 언어를 제공하고, 코드가 없으면 빈 배열입니다. 후속 이력에는 `answer` 원문만 전달합니다.

출처 검증 실패(`invalid_citations`) 시 화면에는 재질문 안내와 질문 원문을 복원합니다. 실패 턴은 후속 이력에서 제외하며 자동 재시도하지 않습니다. CLI 단일 질문은 종료 코드 `1`로 끝나고 대화 모드는 계속 입력받습니다. 근거 부족(`insufficient`) 응답은 정상 이력으로 유지해 후속 질문에서 문맥을 이어갈 수 있습니다.

`/api/chat`의 `sources`에는 `excerpt`와 `excerpt_truncated`가 포함됩니다. 펼쳐보기는 추가 검색·LLM 호출 없이 동작합니다. 원문도 대화와 함께 브라우저 메모리에만 보관하며 후속 질문의 대화 이력과 서버 트레이스에는 추가하지 않습니다. 문서를 수정해도 이전 답변의 원문은 당시 내용으로 남고, 새 질문부터 변경 내용을 사용합니다.

첫 검색부터 답변 검증까지 자료가 변경되면 결과를 반환하지 않고 재질문을 안내합니다. API는 `409`를 반환하며, 화면에는 입력한 질문이 복원됩니다. 같은 질문을 다시 보내면 변경된 자료로 새로 검색합니다. 자동 재시도는 하지 않으며, 실패한 질문은 후속 대화 이력에서 제외합니다. CLI는 안내 후 종료 코드 `1`로 끝납니다.

이 검사는 기본 로컬 검색 경로에서 저장소 전체 지문을 비교하므로 인용하지 않은 문서의 추가·수정·삭제나 메타데이터 변경도 감지합니다. LLM 호출 중에는 저장소를 잠그지 않습니다. 이미 완료된 답변은 당시의 검색 원문을 유지합니다.

지문 계산은 파일명과 내용의 길이도 구분합니다. 이전 형식으로 만든 인덱스는 다음 검색에서 한 번 다시 생성됩니다.

화면의 **지식 저장소**에서 기술 내용 또는 오류 증상·재현 방법·해결 방법을 등록합니다. 엔진 버전은 필수이며 환경과 참고 URL도 남길 수 있습니다. 문서는 `data/unreal/docs/*.json`에 저장됩니다.

**등록된 기록 검색**에서 제목·버전·환경·내용·재현·해결의 일부를 입력하면 불러온 목록을 필터링합니다. 앞뒤 공백과 대소문자는 무시하고 검색어를 문자 그대로 비교합니다. 결과 수와 전체 수를 표시하며 **검색 초기화**로 전체 목록을 복원합니다. 추가 API·LLM 호출 없이 처리하고, 새로고침에서는 검색어를 유지하며 연결 해제·접근 거부 시 비웁니다.

등록된 기록을 펼치면 **수정**과 **삭제**를 선택할 수 있습니다. 수정은 같은 폼에서 저장하거나 취소하며, 삭제는 확인 버튼을 한 번 더 눌러 실행합니다. 조회 이후 다른 작업에서 수정된 기록은 변경을 거부합니다. 이때 목록을 새로고침하고 다시 편집하세요. 수정·삭제 결과는 다음 질문부터 검색에 반영합니다. 기존 형식의 JSON 기록도 수정할 수 있습니다.

API 수정은 `PUT /api/knowledge/{id}`에 등록 필드와 `expected_revision`을 보내고, 삭제는 `DELETE /api/knowledge/{id}?expected_revision=번호`를 사용합니다. revision은 목록 응답에서 확인합니다. 오래된 revision은 `409`, 없는 기록은 `404`로 반환합니다.

서버에서 `.pdf`, `.md` 파일을 같은 디렉터리에 직접 넣을 수도 있습니다. PDF는 텍스트 추출 가능한 문서만 지원하며 OCR·첨부 업로드·URL 수집은 구현하지 않았습니다. **지식 저장소 → PDF · Markdown 문서 → 정보 편집**에서 제목·엔진 버전·출처 URL을 저장합니다. 목록의 **새로고침**은 두 종류의 기록을 함께 갱신합니다. 실제 문서 버전을 확인하기 전에는 기본값인 `미지정`을 유지하세요.

정보는 `data/unreal/docs/metadata.json`에 별도로 저장하며 원본 내용과 PDF 쪽수, 검색 평가용 `doc_id`(확장자를 제외한 파일명)는 유지합니다. 저장한 버전은 검색 버전 선택 목록과 다음 질문의 검색·인용에 반영합니다. 문서 본문 편집·이름 변경·삭제는 화면에서 제공하지 않습니다.

원본마다 고유한 문서 ID가 필요합니다. `Guide.md`와 `Guide.pdf`처럼 확장자를 제외한 이름이 같거나 폼 등록 지식의 ID와 겹치면 검색·인덱싱·평가를 중단하고, 채팅 API는 수정 안내와 `503`을 반환합니다. 임베딩 로드 전에 검사하며 기존 캐시도 검사 대상입니다. 서버에서 PDF/Markdown 파일명을 구분해 주세요. 이름을 바꾸면 문서 ID도 바뀌므로 문서 정보와 평가 정답 ID를 갱신해야 합니다.

API 목록은 `GET /api/documents`, 수정은 `PUT /api/documents/{filename}`을 사용합니다. 파일명 전체(확장자 포함)를 URL 인코딩하고 `title`, `engine_version`, 선택적 `source_url`, 목록에서 받은 `expected_revision`을 보냅니다. 초기 revision은 `0`이며 저장할 때 증가합니다. 충돌 시 `409`를 반환하므로 목록을 새로고침한 뒤 해당 문서의 **정보 편집**을 다시 선택하세요. 손상된 메타데이터는 덮어쓰지 않고 오류를 반환합니다.

문서 변경은 다음 검색에서 자동으로 인덱싱합니다. 미리 준비하려면:

```bash
python -m agents.docs
```

첫 인덱싱은 bge-m3 모델 다운로드(약 2.3GB)와 로딩으로 오래 걸릴 수 있습니다. 지식이 없으면 먼저 등록하세요. 투자 자료인 `data/docs`와 기존 `data/vectorstores`는 검색하지 않습니다. 새 인덱스는 `data/unreal/vectorstores/BAAI--bge-m3/`에 세대별로 저장합니다. 갱신 실패 시 이전 세대 파일은 보존되며, 검색 요청에는 실패를 반환합니다. 현재는 단일 API worker로 실행합니다.

서버 재시작 등으로 저장 인덱스를 다시 읽을 때는 `current.json`에 기록한 파일 쌍·SHA-256과 세대 경로를 먼저 확인합니다. 손상·누락·심볼릭 링크 또는 해시 없는 이전 형식이면 모델 로드와 역직렬화를 중단합니다. 원본 자료를 확인한 뒤 `python -m agents.docs`로 다시 생성하세요. 자료가 변경되어 새 인덱스가 필요한 경우에는 기존 자동 재생성을 유지합니다. 해시는 파일 일관성 확인용이며, 메모리에 로드한 인덱스는 검색마다 다시 검사하지 않습니다.

지식 등록·조회·수정·삭제, 문서 정보 관리, 검색·인덱싱·정리는 `data/unreal/.documents.lock`으로 같은 로컬 저장소를 공유하는 앱 프로세스 간에 직렬화합니다. 프로세스 내부 잠금과 이 파일 잠금의 총 대기는 최대 60초이며, 초과하면 API는 `503`을 반환합니다. 다른 작업이 끝난 뒤 다시 요청하세요. 인덱스 잠금은 그 뒤에 획득하며 별도로 최대 60초 대기합니다.

서버 계정에는 잠금 파일을 만들 `data/unreal/`의 쓰기 권한도 필요합니다. 잠금 파일은 해제 후에도 남습니다. 이 잠금은 앱의 파일 작업에 적용하며, 편집기로 직접 수정하는 작업은 보호하지 않습니다. 여러 worker 실행 시 모델·인덱스 메모리를 각각 사용하므로 worker 수와 성능은 별도 확인이 필요합니다.

새 인덱스를 정상 저장한 뒤 현재 세대와 최근 이전 세대 1개를 자동 보관하고 나머지는 정리합니다. 이전 세대의 순서는 디렉터리 수정 시간 기준입니다. 정리 실패 시에도 저장한 인덱스는 사용할 수 있습니다. 인덱스 파일 작업은 파일 잠금으로 직렬화하며, 잠금 대기 상한은 60초입니다.

수동 정리는 아래 명령으로 확인합니다. 기본은 미리보기이며, `--apply`로 실행합니다. `--keep`은 현재 세대를 포함한 보관 개수이고 해당 실행에만 적용됩니다.

```bash
python -m agents.docs --cleanup
python -m agents.docs --cleanup --keep 1 --apply
```

정리 결과에는 `protected`, `candidates`, `removed`, `failed`, `skipped` 목록이 표시됩니다. 현재 세대는 항상 보존하며, 이름이나 파일 구성이 다른 자료와 심볼릭 링크는 삭제하지 않습니다. 현재 세대의 경로·파일 쌍·해시를 검증할 수 없으면 정리를 중단합니다. 해시가 없는 이전 형식 인덱스는 `python -m agents.docs`로 다시 생성한 뒤 정리하세요.

## 검증

```bash
pip install -r requirements-dev.txt
python -m pytest -q
npm --prefix frontend run build
```

기본 테스트는 실제 LangGraph·FAISS 저장/로드·문서 파싱·API를 실행하며 외부 LLM과 임베딩 모델만 대체합니다. LLM 연결 확인 테스트는 실제 SDK·structured output 파서와 모의 HTTP 응답을 사용합니다. 출처 번호 검사는 인용의 존재·범위를 확인하며 답변 내용의 사실성을 보증하지 않습니다.

실제 bge-m3 검증은 전체 모델이 Hugging Face 캐시에 있는 환경에서 선택 실행합니다. 기본 테스트에서는 건너뛰며, 아래 명령은 오프라인으로 실행하므로 캐시가 없으면 다운로드 없이 실패합니다.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 TOKENIZERS_PARALLELISM=false RUN_BGE_SMOKE=1 python -m pytest -q -s tests/test_embedding_smoke.py
```

임시 기록으로 실제 쿼리/문서 벡터 정규화, FAISS 저장·재로드, 버전 필터와 수정 후 재색인을 확인합니다. 운영 원본·인덱스와 외부 LLM은 사용하지 않습니다. 실제 LLM까지 연결한 종단 검증, 팀 자료의 검색 임계값·답변 정확도 평가는 별도로 필요합니다.

선택적으로 Playwright와 Chromium을 설치하고, 빌드된 화면을 API 서버로 제공한 상태에서 `python tests/browser_smoke.py`를 실행할 수 있습니다. 이 검사는 채팅·지식·문서 정보 응답을 모의 처리하므로 실제 키나 테스트 지식을 저장하지 않습니다.

검증 서버 주소는 `BROWSER_BASE_URL`로 지정합니다. 기본값은 `http://127.0.0.1:8000`이며 다른 포트를 사용하면 `BROWSER_BASE_URL=http://127.0.0.1:8001 python tests/browser_smoke.py`처럼 실행합니다.

정답 문서 ID가 붙은 질문 목록으로 검색 품질을 평가할 수 있습니다. 외부 LLM 호출 없이 Hit Rate·Recall·MRR과 근거 없음 처리를 측정합니다.

```bash
python -m agents.evaluate outputs/evaluation/cases.json --k 5 --output outputs/retrieval_eval.json
```

평가 파일은 직접 준비해야 합니다. 형식과 지표 정의는 [검색 평가 안내](docs/EVALUATION.md)를 참고하세요. 실제 팀 질문의 정답 목록·목표 점수와 생성 답변 정확도 평가는 TODO입니다.

## 저장 범위와 미정 사항

입력한 접근 토큰·LLM 키와 대화는 브라우저 메모리에만 두며 새로고침 시 지워집니다. 서버는 요청 중에만 입력 키를 사용하고, 기본 트레이스에는 실행 ID·상태·담당 이름만 기록합니다. 질문과 검색 문맥은 선택한 외부 LLM으로 전송되며, LangSmith 추적을 켜면 LangSmith에도 전송됩니다. `outputs/traces/`는 운영 상태 기록용입니다.

공유 접근 토큰은 선택적으로 설정할 수 있으며, SSO·사용자별 권한은 미구현입니다. 제공자 확정, 지식 승인·사용자별 편집 권한, 버전 호환성 판정, 검색 평가 등은 [작업 계획](docs/WORK_PLAN.md)에 TODO로 남겼습니다.

지식 삭제는 원본 JSON을 지우고 다음 검색에서 제외합니다. 보관 중인 인덱스 세대와 이미 표시한 대화에는 이전 내용이 남을 수 있습니다. 마지막 문서를 삭제한 경우에는 새 인덱스를 만들지 않으므로 현재 세대도 남습니다. 완전한 데이터 삭제·보존 정책은 별도로 확정해야 합니다.

기존 투자 평가 전용 Python 모듈과 미사용 프롬프트는 삭제했습니다. `prompts/`에는 현재 사용하는 `supervisor.md`, `writer.md`와 로더만 유지합니다. 기존 투자 원본 자료는 남아 있지만 새 실행 경로에서는 사용하지 않습니다.
