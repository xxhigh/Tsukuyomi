<script setup>
import { computed, nextTick, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import DocumentLibrary from './DocumentLibrary.vue'

const tab = ref('chat')
const apiKey = ref('')
const model = ref('')
const serverKey = ref(false)
const accessToken = ref('')
const activeToken = ref('')
const accessReady = ref(false)
const connecting = ref(false)
const accessError = ref('')
const configured = computed(() => accessReady.value && Boolean(apiKey.value.trim() || serverKey.value))
const question = ref('')
const messages = ref([])
const entries = ref([])
const entrySearch = ref('')
const filteredEntries = computed(() => {
  const query = entrySearch.value.trim().toLowerCase()
  if (!query) return entries.value
  return entries.value.filter(item => ['title', 'engine_version', 'context', 'content', 'reproduction', 'solution']
    .some(field => (item[field] || '').toLowerCase().includes(query)))
})
const documents = ref([])
const workspaceEpoch = ref(0)
const documentSaving = ref(false)
const documentsError = ref('')
const engineVersion = ref('')
const engineVersions = computed(() => [...new Set([...entries.value, ...documents.value].map(item => item.engine_version))]
  .sort((left, right) => left.localeCompare(right, undefined, { numeric: true })))
const busy = ref(false)
const checking = ref(false)
const connectionNotice = ref('')
const connectionError = ref('')
const saving = ref(false)
const editing = ref(null)
const deletingId = ref('')
const deleteTarget = ref('')
const mutating = computed(() => saving.value || documentSaving.value || checking.value || Boolean(deletingId.value))
const loading = ref(true)
const chatError = ref('')
const knowledgeError = ref('')
const notice = ref('')
const conversation = ref(null)
let chatController
watch([apiKey, model], () => { connectionNotice.value = ''; connectionError.value = '' })
const emptyEntry = () => ({ title: '', kind: 'technique', engine_version: '', context: '', content: '', reproduction: '', solution: '', source_url: '' })
const entry = reactive(emptyEntry())

async function request(path, options = {}) {
  const headers = new Headers(options.headers)
  if (activeToken.value) headers.set('Authorization', `Bearer ${activeToken.value}`)
  const response = await fetch(`/api${path}`, { ...options, headers })
  const data = await response.json().catch(() => ({}))
  if (!response.ok) {
    const error = new Error(typeof data.detail === 'string' ? data.detail : data.detail?.message || `요청에 실패했습니다 (${response.status}).`)
    error.accessDenied = data.detail?.code === 'internal_access_required'
    if (error.accessDenied) {
      accessReady.value = false
      activeToken.value = ''
      clearWorkspace()
      accessError.value = error.message
      tab.value = 'settings'
    }
    throw error
  }
  return data
}

async function loadKnowledge() {
  if (!accessReady.value) return
  const epoch = workspaceEpoch.value
  loading.value = true
  knowledgeError.value = ''
  documentsError.value = ''
  await Promise.all([
    loadList('/knowledge', entries, knowledgeError, epoch),
    loadList('/documents', documents, documentsError, epoch),
  ])
  loading.value = false
}

async function loadList(path, target, errorTarget, epoch) {
  try {
    const result = await request(path)
    if (accessReady.value && workspaceEpoch.value === epoch) target.value = result.entries
  } catch (error) {
    if (!error.accessDenied && workspaceEpoch.value === epoch) errorTarget.value = error.message
  }
}

function documentSaved(record) {
  documents.value = documents.value.map(item => item.filename === record.filename ? record : item)
}

function clearWorkspace() {
  checking.value = false
  connectionNotice.value = ''
  connectionError.value = ''
  messages.value = []
  entries.value = []
  entrySearch.value = ''
  documents.value = []
  documentsError.value = ''
  documentSaving.value = false
  workspaceEpoch.value += 1
  question.value = ''
  apiKey.value = ''
  model.value = ''
  serverKey.value = false
  engineVersion.value = ''
  resetEntry()
  deleteTarget.value = ''
  chatError.value = ''
  knowledgeError.value = ''
  notice.value = ''
}

async function connect() {
  if (connecting.value || busy.value || mutating.value) return
  connecting.value = true
  accessReady.value = false
  clearWorkspace()
  accessError.value = ''
  activeToken.value = accessToken.value.trim()
  try {
    const config = await request('/config')
    model.value = config.model
    serverKey.value = config.has_server_key
    accessReady.value = true
    await loadKnowledge()
  } catch (error) {
    accessError.value = error.message
    activeToken.value = ''
    tab.value = 'settings'
  } finally {
    connecting.value = false
    loading.value = false
  }
}

function disconnect() {
  accessToken.value = ''
  activeToken.value = ''
  accessReady.value = false
  accessError.value = ''
  clearWorkspace()
}

onMounted(connect)
onUnmounted(() => chatController?.abort())

async function checkLLM() {
  if (!configured.value || busy.value || mutating.value || loading.value) return
  const epoch = workspaceEpoch.value
  checking.value = true
  connectionNotice.value = ''
  connectionError.value = ''
  try {
    const headers = { 'Content-Type': 'application/json' }
    if (apiKey.value.trim()) headers['X-LLM-API-Key'] = apiKey.value.trim()
    const result = await request('/llm/check', {
      method: 'POST', headers, body: JSON.stringify({ model: model.value.trim() || null }),
    })
    if (accessReady.value && workspaceEpoch.value === epoch) connectionNotice.value = result.message
  } catch (error) {
    if (!error.accessDenied && workspaceEpoch.value === epoch) connectionError.value = error.message
  } finally {
    checking.value = false
  }
}

async function scrollToEnd() {
  await nextTick()
  conversation.value?.scrollIntoView({ block: 'end', behavior: 'instant' })
}

async function ask() {
  const text = question.value.trim()
  if (!text || busy.value || checking.value || !configured.value) return
  const selectedVersion = engineVersion.value.trim() || null
  const history = messages.value.filter(message => !message.failed && message.engineVersion === selectedVersion).slice(-10)
    .map(({ role, content }) => ({ role, content: content.slice(0, 8000) }))
  const current = { role: 'user', content: text, engineVersion: selectedVersion }
  messages.value.push(current)
  question.value = ''
  chatError.value = ''
  busy.value = true
  chatController = new AbortController()
  await scrollToEnd()
  try {
    const headers = { 'Content-Type': 'application/json' }
    if (apiKey.value.trim()) headers['X-LLM-API-Key'] = apiKey.value.trim()
    const result = await request('/chat', {
      method: 'POST', headers, signal: chatController.signal,
      body: JSON.stringify({ question: text, history, model: model.value.trim() || null, engine_version: selectedVersion }),
    })
    if (result.status === 'invalid_citations') throw new Error(result.answer)
    messages.value.push({ role: 'assistant', content: result.answer, answerBlocks: result.answer_blocks,
      sources: result.sources, status: result.status, engineVersion: selectedVersion })
  } catch (error) {
    // 실패한 질문을 후속 요청의 대화 이력에 포함하지 않는다.
    if (!error.accessDenied) {
      messages.value[messages.value.length - 1].failed = true
      question.value = text
      chatError.value = error.name === 'AbortError' ? '응답 대기를 취소했습니다.' : error.message
    }
  } finally {
    busy.value = false
    await scrollToEnd()
  }
}

async function saveEntry() {
  if (mutating.value || !accessReady.value) return
  saving.value = true
  knowledgeError.value = ''
  notice.value = ''
  try {
    const payload = { ...entry, source_url: entry.source_url.trim() || null }
    if (payload.kind === 'technique') {
      payload.reproduction = ''
      payload.solution = ''
    }
    if (editing.value) payload.expected_revision = editing.value.revision
    const path = editing.value ? `/knowledge/${encodeURIComponent(editing.value.id)}` : '/knowledge'
    const result = await request(path, {
      method: editing.value ? 'PUT' : 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload),
    })
    resetEntry()
    notice.value = result.message
    await loadKnowledge()
  } catch (error) {
    if (!error.accessDenied) knowledgeError.value = error.message
  } finally {
    saving.value = false
  }
}

function resetEntry() {
  editing.value = null
  Object.assign(entry, emptyEntry())
}

async function editEntry(item) {
  if (mutating.value || !accessReady.value) return
  editing.value = { id: item.id, revision: item.revision ?? 1 }
  for (const key of Object.keys(emptyEntry())) entry[key] = item[key] ?? ''
  notice.value = ''
  knowledgeError.value = ''
  deleteTarget.value = ''
  await nextTick()
  document.getElementById('title')?.focus()
}

async function deleteEntry(item) {
  if (mutating.value || !accessReady.value) return
  deletingId.value = item.id
  knowledgeError.value = ''
  notice.value = ''
  try {
    await request(`/knowledge/${encodeURIComponent(item.id)}?expected_revision=${item.revision ?? 1}`, { method: 'DELETE' })
    if (editing.value?.id === item.id) resetEntry()
    deleteTarget.value = ''
    notice.value = '삭제했습니다. 다음 질문의 검색 결과에서 제외됩니다.'
    await loadKnowledge()
  } catch (error) {
    if (!error.accessDenied) knowledgeError.value = error.message
  } finally {
    deletingId.value = ''
  }
}

function safeUrl(value) {
  try {
    const url = new URL(value)
    return ['https:', 'http:'].includes(url.protocol) ? url.href : null
  } catch { return null }
}
function onQuestionEnter(event) {
  if (event.isComposing) return
  event.preventDefault()
  ask()
}
</script>

<template>
  <a class="skip-link" href="#workspace">작업 영역으로 이동</a>
  <div class="shell">
    <aside class="sidebar" aria-label="도구 메뉴">
      <div class="identity"><span class="symbol" aria-hidden="true">T / U</span><div><strong>Tsukuyomi for Unreal</strong><span>팀 기술 노트</span></div></div>
      <p class="eyebrow">WORKSPACE</p>
      <nav aria-label="작업 선택">
        <button :class="{ selected: tab === 'chat' }" :aria-pressed="tab === 'chat'" :disabled="mutating" @click="tab = 'chat'">대화<span aria-hidden="true">↗</span></button>
        <button :class="{ selected: tab === 'knowledge' }" :aria-pressed="tab === 'knowledge'" :disabled="mutating" @click="tab = 'knowledge'">지식 저장소<span>{{ entries.length }}</span></button>
        <button :class="{ selected: tab === 'settings' }" :aria-pressed="tab === 'settings'" :disabled="mutating" @click="tab = 'settings'">설정<span aria-hidden="true">↗</span></button>
      </nav>
      <section class="search-settings" aria-label="검색 범위">
        <label for="search-version">검색 엔진 버전</label>
        <input id="search-version" v-model="engineVersion" list="engine-versions" maxlength="50" placeholder="전체 버전" :disabled="busy || !accessReady" aria-describedby="version-help" />
        <datalist id="engine-versions"><option v-for="version in engineVersions" :key="version" :value="version"></option></datalist>
        <p id="version-help" class="hint">입력한 버전과 정확히 일치하는 기록만 검색합니다. 비우면 전체를 검색합니다.</p>
      </section>
      <footer class="sidebar-footer">INTERNAL TOOL <span>Unreal Engine RAG</span></footer>
    </aside>

    <main id="workspace" :class="{ 'idle-workspace': tab === 'chat' && !messages.length }" tabindex="-1">
      <header v-if="tab !== 'chat' || messages.length" class="workspace-header"><span>{{ tab === 'chat' ? '대화 / 기술 지원' : tab === 'knowledge' ? '지식 / 팀 기록' : '설정 / 연결 관리' }}</span><span class="subtle">UNREAL ENGINE</span></header>
      <section v-if="tab === 'chat'" class="chat-workspace" :class="{ 'is-empty': !messages.length }" aria-label="대화">
        <div v-if="messages.length" class="page-title"><div><p class="eyebrow">ASK YOUR KNOWLEDGE</p><h1>기록에서 답을 찾으세요.</h1><p>언리얼 기술과 팀이 해결한 오류를 근거와 함께 확인합니다.</p></div><button class="quiet" :disabled="busy" @click="messages = []; chatError = ''">새 대화</button></div>
        <p v-if="messages.length && !configured" class="setup-notice">{{ accessReady ? '대화를 시작하려면 설정에서 LLM API 키를 입력하세요.' : '설정에서 내부 연결을 확인해 주세요.' }} <button class="quiet" :disabled="mutating" @click="tab = 'settings'">연결 설정 열기</button></p>
        <div v-if="messages.length" class="messages" aria-live="polite" aria-relevant="additions" :aria-busy="busy">
          <article v-for="(message, index) in messages" :key="index" :class="['message', message.role, { failed: message.failed }]">
            <p class="message-role">{{ message.role === 'user' ? '나' : 'Tsukuyomi for Unreal' }} · {{ message.engineVersion ? `UE ${message.engineVersion}` : '전체 버전' }}<span v-if="message.failed"> · 처리 미완료</span></p>
            <div class="message-content">
              <template v-if="message.answerBlocks?.length">
                <template v-for="(block, blockIndex) in message.answerBlocks" :key="blockIndex">
                  <div v-if="block.type === 'code'" class="answer-code">
                    <p class="code-language">{{ block.language || '코드' }}</p>
                    <pre tabindex="0" aria-label="코드 블록"><code>{{ block.content }}</code></pre>
                  </div>
                  <span v-else>{{ block.content }}</span>
                </template>
              </template>
              <template v-else>{{ message.content }}</template>
            </div>
            <ul v-if="message.sources?.length" class="sources" aria-label="답변 근거">
              <li v-for="source in message.sources" :key="source.number">
                <span class="source-number">[{{ source.number }}]</span>
                <a v-if="safeUrl(source.url)" :href="safeUrl(source.url)" target="_blank" rel="noopener noreferrer">{{ source.title }}</a><span v-else>{{ source.title }}</span>
                <small>UE {{ source.engine_version || '미지정' }}{{ source.page ? ` · p.${source.page}` : '' }}</small>
                <details v-if="source.excerpt" class="source-excerpt">
                  <summary>검색 원문 보기 <span class="sr-only">출처 [{{ source.number }}] {{ source.title }}</span></summary>
                  <p class="hint">답변 당시 검색한 원문의 일부입니다.</p>
                  <blockquote>{{ source.excerpt }}</blockquote>
                  <p v-if="source.excerpt_truncated" class="hint">원문이 길어 앞 1,000자만 표시했습니다.</p>
                </details>
              </li>
            </ul>
          </article>
        </div>
        <div v-if="busy" class="pending" role="status">기록을 검색하고 답변을 작성하고 있습니다. 첫 검색은 모델 준비로 오래 걸릴 수 있습니다.<button class="quiet" @click="chatController?.abort()">대기 취소</button></div>
        <p v-if="chatError" class="error" role="alert">{{ chatError }}</p>
        <div ref="conversation"></div>
        <form class="composer" @submit.prevent="ask">
          <label class="sr-only" for="question">언리얼 기술 질문</label>
          <textarea id="question" v-model="question" rows="3" maxlength="8000" placeholder="엔진 버전, 오류 메시지 또는 궁금한 기술을 입력하세요." :disabled="busy" required @keydown.enter.exact="onQuestionEnter"></textarea>
          <div class="composer-bottom"><span v-if="messages.length">Enter 전송 · Shift + Enter 줄바꿈</span><button class="primary" type="submit" aria-label="질문 보내기" :disabled="busy || checking || !question.trim() || !configured"><template v-if="messages.length">{{ busy ? '답변 작성 중' : '질문 보내기' }} </template><span aria-hidden="true">↑</span></button></div>
        </form>
        <p v-if="messages.length" class="footnote">검색 근거와 엔진 버전을 확인한 뒤 프로젝트에 적용하세요.</p>
      </section>

      <section v-show="tab === 'knowledge'" class="knowledge-workspace" aria-labelledby="knowledge-title">
        <div class="page-title"><div><p class="eyebrow">SHARED KNOWLEDGE</p><h1 id="knowledge-title">다음에 다시 찾을 수 있도록.</h1><p>새로 익힌 기술과 해결한 오류를 팀의 기록으로 남깁니다.</p></div></div>
        <div class="knowledge-layout">
          <form class="knowledge-form" aria-labelledby="entry-form-title" @submit.prevent="saveEntry">
            <h2 id="entry-form-title">{{ editing ? '지식 수정' : '지식 등록' }}</h2>
            <fieldset :disabled="mutating || !accessReady">
            <label for="kind">기록 종류</label><select id="kind" v-model="entry.kind"><option value="technique">기술 / 개발 방법</option><option value="error">오류 / 해결 사례</option></select>
            <label for="title">제목</label><input id="title" v-model="entry.title" required maxlength="200" placeholder="나중에 찾기 쉬운 제목" />
            <label for="version">엔진 버전</label><input id="version" v-model="entry.engine_version" required maxlength="50" placeholder="예: 5.5.4" />
            <label for="context">환경 / 전제 조건 <span class="optional">선택</span></label><textarea id="context" v-model="entry.context" rows="2" maxlength="3000" placeholder="플랫폼, 플러그인, 빌드 설정 등"></textarea>
            <label for="content">{{ entry.kind === 'error' ? '오류 메시지 / 증상' : '기술 내용 / 적용 방법' }}</label><textarea id="content" v-model="entry.content" rows="5" required maxlength="20000"></textarea>
            <template v-if="entry.kind === 'error'">
              <label for="reproduction">재현 방법</label><textarea id="reproduction" v-model="entry.reproduction" rows="3" required maxlength="5000"></textarea>
              <label for="solution">해결 방법 / 확인 결과</label><textarea id="solution" v-model="entry.solution" rows="4" required maxlength="10000"></textarea>
            </template>
            <label for="source">참고 URL <span class="optional">선택</span></label><input id="source" v-model="entry.source_url" type="url" maxlength="2000" placeholder="https://…" />
            <p v-if="knowledgeError" class="error" role="alert">{{ knowledgeError }}</p>
            <p v-if="notice" class="success" role="status">{{ notice }}</p>
            <div class="entry-actions">
              <button class="primary" type="submit">{{ saving ? '저장 중…' : editing ? '수정 저장' : '지식 저장' }}</button>
              <button v-if="editing" type="button" @click="resetEntry(); knowledgeError = ''; notice = ''">편집 취소</button>
            </div>
            </fieldset>
          </form>
          <aside class="entry-list" aria-labelledby="entries-title">
            <div class="list-heading"><h2 id="entries-title">등록된 기록 <span>{{ entries.length }}</span></h2><button class="quiet" :disabled="loading || mutating || !accessReady" @click="loadKnowledge">새로고침</button></div>
            <label for="entry-search">등록된 기록 검색</label>
            <input id="entry-search" v-model="entrySearch" type="search" maxlength="200" :disabled="loading || !accessReady" aria-describedby="entry-search-help" placeholder="기술명, 오류 메시지 또는 버전" />
            <p id="entry-search-help" class="hint">제목·버전·환경·내용·재현·해결에서 검색합니다.</p>
            <button v-if="entrySearch" class="quiet" type="button" @click="entrySearch = ''">검색 초기화</button>
            <p v-if="loading" role="status">기록을 불러오는 중…</p>
            <p v-else-if="!entries.length" class="hint">등록된 기록이 없습니다. 왼쪽 양식으로 첫 지식을 저장하세요.</p>
            <p v-if="entrySearch.trim() && !loading" class="hint" role="status">검색 결과 {{ filteredEntries.length }}개 / 전체 {{ entries.length }}개</p>
            <p v-if="!loading && entries.length && !filteredEntries.length" class="hint">검색어와 일치하는 기록이 없습니다.</p>
            <details v-for="item in filteredEntries" :key="item.id" class="entry">
              <summary><span class="entry-meta">{{ item.kind === 'error' ? '오류 해결' : '기술' }} · UE {{ item.engine_version }}</span><strong>{{ item.title }}</strong></summary>
              <p v-if="item.context">{{ item.context }}</p><p>{{ item.content }}</p>
              <template v-if="item.kind === 'error'"><h3>재현 방법</h3><p>{{ item.reproduction }}</p><h3>해결 방법</h3><p>{{ item.solution }}</p></template>
              <a v-if="safeUrl(item.source_url)" :href="safeUrl(item.source_url)" target="_blank" rel="noopener noreferrer">참고 자료 열기 ↗</a>
              <div class="entry-actions">
                <button type="button" :disabled="mutating" @click="editEntry(item)">수정</button>
                <button type="button" :disabled="mutating" @click="deleteTarget = item.id">삭제</button>
              </div>
              <div v-if="deleteTarget === item.id" class="delete-confirm" role="group" :aria-label="`${item.title} 삭제 확인`">
                <p>이 기록을 삭제할까요? 삭제하면 다음 검색에서 제외됩니다.</p>
                <div class="entry-actions">
                  <button class="danger" type="button" :disabled="mutating" @click="deleteEntry(item)">{{ deletingId === item.id ? '삭제 중…' : '삭제 확인' }}</button>
                  <button type="button" :disabled="mutating" @click="deleteTarget = ''">삭제 취소</button>
                </div>
              </div>
            </details>
          </aside>
        </div>
        <DocumentLibrary :key="workspaceEpoch" :entries="documents" :request="request"
          :disabled="!accessReady || mutating || loading" :error="documentsError"
          @busy="documentSaving = $event" @saved="documentSaved" />
      </section>

      <section v-if="tab === 'settings'" class="settings-workspace" aria-labelledby="settings-title">
        <div class="page-title"><div><p class="eyebrow">SETTINGS</p><h1 id="settings-title">연결 설정</h1><p>내부 접근과 답변에 사용할 LLM을 설정합니다.</p></div></div>
        <div class="settings-layout">
          <section class="settings-panel access-settings" aria-labelledby="access-title">
            <h2 id="access-title">내부 연결</h2>
            <p class="connection-state"><span class="status-dot" :class="{ ready: accessReady }"></span>{{ connecting ? '연결 중…' : accessReady ? '연결됨' : '연결 필요' }}</p>
            <form @submit.prevent="connect">
              <label for="access-token">내부 접근 토큰</label>
              <input id="access-token" v-model="accessToken" type="password" autocomplete="off" spellcheck="false" maxlength="4096" placeholder="서버에 설정된 경우 입력" :disabled="connecting || busy || mutating || loading" />
              <p class="hint">서버에 토큰을 설정한 경우에만 입력하세요. LLM API 키와 별도입니다.</p>
              <p class="hint">다시 연결하거나 연결을 해제하면 대화와 작성 중인 내용을 비웁니다.</p>
              <div class="entry-actions">
                <button type="submit" :disabled="connecting || busy || mutating || loading">{{ connecting ? '연결 중…' : '연결' }}</button>
                <button type="button" :disabled="!accessReady || connecting || busy || mutating || loading" @click="disconnect">연결 해제</button>
              </div>
            </form>
            <p v-if="accessError" class="error" role="alert">{{ accessError }}</p>
          </section>
          <section class="settings-panel connection" aria-labelledby="connection-title">
            <h2 id="connection-title">LLM 연결</h2>
            <p class="connection-state"><span class="status-dot" :class="{ ready: configured }"></span>{{ apiKey.trim() ? '입력한 키 사용' : serverKey ? '서버에 설정된 키 사용' : 'API 키 입력 필요' }}</p>
            <label for="api-key">API 키</label>
            <input id="api-key" v-model="apiKey" type="password" autocomplete="off" spellcheck="false" placeholder="API 키 입력" maxlength="4096" :disabled="!accessReady || checking" />
            <p class="hint">입력한 키와 토큰은 새로고침하거나 연결을 해제하면 지워집니다. 메뉴를 이동해도 유지됩니다.</p>
            <label for="model">모델</label>
            <input id="model" v-model="model" placeholder="모델 이름" maxlength="100" autocomplete="off" :disabled="!accessReady || checking" />
            <div class="llm-check">
              <button type="button" :disabled="!configured || busy || mutating || loading" @click="checkLLM">{{ checking ? '연결 확인 중…' : 'LLM 연결 확인' }}</button>
              <p class="hint">누르면 짧은 LLM 요청이 전송되며 사용량이 발생합니다.</p>
              <p v-if="connectionNotice" class="success" role="status">{{ connectionNotice }}</p>
              <p v-if="connectionError" class="error" role="alert">{{ connectionError }}</p>
            </div>
          </section>
        </div>
      </section>
    </main>
  </div>
</template>
