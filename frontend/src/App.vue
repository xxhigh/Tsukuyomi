<script setup>
import { computed, nextTick, onMounted, onUnmounted, reactive, ref, watch } from 'vue'
import DocumentLibrary from './DocumentLibrary.vue'
import { ENGINE_VERSIONS as engineVersions } from './engineVersions.js'
import { createConversationStore } from './conversations.js'

const tab = ref('chat')
const apiKey = ref('')
const showApiKey = ref(false)
const model = ref('')
const serverKey = ref(false)
const accessToken = ref('')
const activeToken = ref('')
const accessReady = ref(false)
const connecting = ref(false)
const accessError = ref('')
const configured = computed(() => accessReady.value && Boolean(apiKey.value.trim() || serverKey.value))
const chatHistory = createConversationStore()
const { question, messages, engineVersion, chatError, recent: recentConversations,
  activeId: activeConversationId, storageWarning } = chatHistory
const conversationDeleteTarget = ref('')
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
const versionOptions = ['', ...engineVersions]
const versionMenuOpen = ref(false)
const versionPicker = ref(null)
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
  chatHistory.clearVisible()
  conversationDeleteTarget.value = ''
  entries.value = []
  entrySearch.value = ''
  documents.value = []
  documentsError.value = ''
  documentSaving.value = false
  workspaceEpoch.value += 1
  question.value = ''
  apiKey.value = ''
  showApiKey.value = false
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
    chatHistory.load()
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
onMounted(() => document.addEventListener('pointerdown', onOutsidePointer))
onUnmounted(() => {
  chatController?.abort()
  document.removeEventListener('pointerdown', onOutsidePointer)
})

function versionItems() {
  return [...(versionPicker.value?.querySelectorAll('[role="menuitemradio"]') ?? [])]
}

async function openVersionMenu() {
  versionMenuOpen.value = true
  await nextTick()
  const items = versionItems()
  ;(items[versionOptions.indexOf(engineVersion.value)] ?? items[0])?.focus()
}

function closeVersionMenu(refocus = true) {
  versionMenuOpen.value = false
  if (refocus) versionPicker.value?.querySelector('.version-trigger')?.focus()
}

function chooseVersion(version) {
  engineVersion.value = version
  closeVersionMenu()
}

function onOutsidePointer(event) {
  if (versionMenuOpen.value && !versionPicker.value?.contains(event.target)) closeVersionMenu(false)
}

function onVersionKeydown(event) {
  if (!versionMenuOpen.value) return
  if (event.key === 'Escape') {
    event.preventDefault()
    closeVersionMenu()
  } else if (event.key === 'Tab') {
    closeVersionMenu(false)
  } else if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
    event.preventDefault()
    const items = versionItems()
    const current = items.indexOf(document.activeElement)
    const next = { ArrowDown: current + 1, ArrowUp: current - 1, Home: 0, End: items.length - 1 }[event.key]
    items[(next + items.length) % items.length]?.focus()
  }
}

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

function startNewConversation() {
  if (busy.value || mutating.value || !accessReady.value) return
  chatHistory.startNew()
  conversationDeleteTarget.value = ''
  versionMenuOpen.value = false
  tab.value = 'chat'
}

function openConversation(id) {
  if (busy.value || mutating.value || !accessReady.value) return
  chatHistory.open(id)
  conversationDeleteTarget.value = ''
  versionMenuOpen.value = false
  tab.value = 'chat'
  scrollToEnd()
}

function deleteConversation(id) {
  if (busy.value || mutating.value || !accessReady.value) return
  chatHistory.remove(id)
  conversationDeleteTarget.value = ''
}

async function ask() {
  const text = question.value.trim()
  if (!text || busy.value || checking.value || !configured.value) return
  const selectedVersion = engineVersion.value.trim() || null
  const history = messages.value.filter(message => !message.failed && message.engineVersion === selectedVersion).slice(-10)
    .map(({ role, content }) => ({ role, content: content.slice(0, 8000) }))
  const epoch = workspaceEpoch.value
  const activeConversation = chatHistory.ensureConversation(text)
  const current = { role: 'user', content: text, engineVersion: selectedVersion }
  activeConversation.messages.push(current)
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
    if (epoch !== workspaceEpoch.value || !accessReady.value) return
    if (result.status === 'invalid_citations') throw new Error(result.answer)
    activeConversation.messages.push({ role: 'assistant', content: result.answer, answerBlocks: result.answer_blocks,
      sources: result.sources, status: result.status, engineVersion: selectedVersion })
  } catch (error) {
    // 실패한 질문을 후속 요청의 대화 이력에 포함하지 않는다.
    if (!error.accessDenied && epoch === workspaceEpoch.value) {
      activeConversation.messages[activeConversation.messages.length - 1].failed = true
      question.value = text
      chatError.value = error.name === 'AbortError' ? '응답 대기를 취소했습니다.' : error.message
    }
  } finally {
    busy.value = false
    if (epoch === workspaceEpoch.value) await scrollToEnd()
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
      <div class="identity"><svg class="brand-mark" viewBox="0 0 40 40" fill="none" aria-hidden="true" focusable="false"><path d="M20.00 4.00 C24.19 4.21 28.17 6.14 30.86 9.14 C33.56 12.13 34.94 16.14 34.73 20.00 C34.52 23.86 32.72 27.51 29.96 29.96 C27.21 32.42 23.52 33.67 20.00 33.45 C16.48 33.24 13.16 31.59 10.94 29.06 C8.71 26.54 7.61 23.19 7.82 20.00 C8.03 16.81 9.55 13.82 11.84 11.84 C14.12 9.85 17.14 8.88 20.00 9.09 C22.86 9.30 25.51 10.68 27.26 12.74 C29.02 14.79 29.85 17.48 29.64 20.00 C29.42 22.52 28.18 24.85 26.36 26.36 C24.55 27.88 22.19 28.58 20.00 28.36 C17.81 28.15 15.82 27.04 14.54 25.46 C13.26 23.88 12.70 21.86 12.91 20.00 C13.12 18.14 14.09 16.48 15.44 15.44 C16.78 14.39 18.48 13.97 20.00 14.18 C21.52 14.39 22.85 15.23 23.66 16.34 C24.47 17.45 24.76 18.81 24.55 20.00 C24.33 21.19 23.64 22.19 22.76 22.76 C21.89 23.34 20.86 23.48 20.00 23.27 C19.14 23.06 18.47 22.50 18.14 21.86 C17.80 21.23 17.79 20.52 18.00 20.00" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" /></svg><div><strong><span class="wordmark-accent">Tsuk</span>uyomi</strong><span class="brand-caption">for Unreal Engine</span></div></div>
      <p class="eyebrow">WORKSPACE</p>
      <nav aria-label="작업 선택">
        <button :class="{ selected: tab === 'chat' }" :aria-pressed="tab === 'chat'" :disabled="mutating" @click="tab = 'chat'"><svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M5 4h14a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H9l-6 3V6a2 2 0 0 1 2-2Z"/><path d="M7 9h10M7 13h6"/></svg>대화<span aria-hidden="true">↗</span></button>
        <button :class="{ selected: tab === 'knowledge' }" :aria-pressed="tab === 'knowledge'" :disabled="mutating" @click="tab = 'knowledge'"><svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><rect x="4" y="3" width="16" height="18" rx="2"/><path d="M8 3v18M11 8h5M11 12h5"/></svg>지식 저장소<span>{{ entries.length }}</span></button>
        <button :class="{ selected: tab === 'settings' }" :aria-pressed="tab === 'settings'" :disabled="mutating" @click="tab = 'settings'"><svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><path d="M4 7h16M4 17h16"/><circle cx="9" cy="7" r="3"/><circle cx="15" cy="17" r="3"/></svg>설정<span aria-hidden="true">↗</span></button>
      </nav>
      <section v-if="accessReady" class="conversation-history" aria-labelledby="conversation-history-title">
        <div class="history-heading"><h2 id="conversation-history-title">최근 대화</h2><button type="button" class="quiet" aria-label="새 대화 시작" :disabled="busy || mutating" @click="startNewConversation">＋</button></div>
        <p v-if="storageWarning" class="history-warning" role="alert">{{ storageWarning }}</p>
        <ul class="conversation-list">
          <li v-for="item in recentConversations" :key="item.id">
            <div class="conversation-row" :class="{ active: activeConversationId === item.id }">
              <button type="button" class="conversation-link" :aria-label="`대화 열기: ${item.title}`" :aria-current="activeConversationId === item.id ? 'true' : undefined" :title="item.title" :disabled="busy || mutating" @click="openConversation(item.id)">{{ item.title }}</button>
              <button type="button" class="conversation-delete" :aria-label="`대화 삭제: ${item.title}`" :disabled="busy || mutating" @click="conversationDeleteTarget = item.id">×</button>
            </div>
            <div v-if="conversationDeleteTarget === item.id" class="conversation-delete-confirm" role="group" :aria-label="`${item.title} 대화 삭제 확인`">
              <p>이 브라우저에서 대화를 삭제할까요?</p><div class="entry-actions"><button type="button" class="danger" :disabled="busy || mutating" @click="deleteConversation(item.id)">삭제 확인</button><button type="button" @click="conversationDeleteTarget = ''">취소</button></div>
            </div>
          </li>
        </ul>
      </section>
      <footer class="sidebar-footer">COPYRIGHT 2026. xxhigh</footer>
    </aside>

    <main id="workspace" :class="{ 'idle-workspace': tab === 'chat' && !messages.length, 'chat-mode': tab === 'chat' }" tabindex="-1">
      <header v-if="tab !== 'chat'" class="workspace-header"><span>{{ tab === 'knowledge' ? '지식' : '설정 / 연결 관리' }}</span><span class="subtle">UNREAL ENGINE</span></header>
      <header v-else-if="messages.length" class="chat-header"><span>워크스페이스 <span aria-hidden="true">/</span> <strong>기술 지원 대화</strong></span><button class="quiet" :disabled="busy || mutating" @click="startNewConversation">새 대화 <span aria-hidden="true">＋</span></button></header>
      <section v-if="tab === 'chat'" class="chat-workspace" :class="{ 'is-empty': !messages.length }" aria-label="대화">
        <h1 v-if="!messages.length" class="start-wordmark"><span class="start-wordmark-name"><span class="wordmark-accent">Tsuk</span>uyomi</span><span class="start-wordmark-subtitle">for Unreal</span></h1>
        <p v-if="messages.length && !configured" class="setup-notice">{{ accessReady ? '대화를 시작하려면 설정에서 LLM API 키를 입력하세요.' : '설정에서 내부 연결을 확인해 주세요.' }} <button class="quiet" :disabled="mutating" @click="tab = 'settings'">연결 설정 열기</button></p>
        <div v-if="messages.length" class="messages" aria-live="polite" aria-relevant="additions" :aria-busy="busy">
          <article v-for="(message, index) in messages" :key="index" :class="['message', message.role, { failed: message.failed }]">
            <p class="message-role"><span :class="{ 'sr-only': message.role === 'user' }">{{ message.role === 'user' ? '나' : 'Tsukuyomi' }} · </span>{{ message.engineVersion ? `UE ${message.engineVersion}` : '전체 버전' }}<span v-if="message.failed"> · 처리 미완료</span></p>
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
        <div v-if="busy" class="pending" role="status"><span class="pulse" aria-hidden="true"></span>기록을 검색하고 답변을 작성하고 있습니다. 첫 검색은 모델 준비로 오래 걸릴 수 있습니다.</div>
        <p v-if="chatError" class="error" role="alert">{{ chatError }}</p>
        <div ref="conversation"></div>
        <div class="composer-dock">
          <form class="composer" @submit.prevent="ask">
            <label class="sr-only" for="question">언리얼 기술 질문</label>
            <textarea id="question" v-model="question" rows="2" maxlength="8000" placeholder="엔진 버전, 오류 메시지 또는 궁금한 기술을 입력하세요." :disabled="busy" required @keydown.enter.exact="onQuestionEnter"></textarea>
            <div class="composer-bottom">
              <div ref="versionPicker" class="version-picker" @keydown="onVersionKeydown">
                <button type="button" class="version-trigger" aria-haspopup="menu" :aria-expanded="versionMenuOpen" aria-controls="version-menu"
                  :disabled="busy || !accessReady" @click="versionMenuOpen ? closeVersionMenu() : openVersionMenu()">
                  엔진 버전: <strong>{{ engineVersion || '전체' }}</strong><span class="chevron" aria-hidden="true">⌄</span>
                </button>
                <div v-if="versionMenuOpen" id="version-menu" class="version-menu" role="menu" aria-label="엔진 버전 선택">
                  <p class="version-menu-title" aria-hidden="true">마이너 버전 · 패치 기록 포함</p>
                  <button v-for="version in versionOptions" :key="version" type="button" role="menuitemradio" tabindex="-1"
                    :aria-checked="engineVersion === version" @click="chooseVersion(version)">
                    <span><strong>{{ version ? `UE ${version}` : '전체 버전' }}</strong><small v-if="!version">모든 버전의 기록을 검색</small></span>
                    <span class="check" aria-hidden="true">✓</span>
                  </button>
                </div>
              </div>
              <button v-if="busy" class="send stop" type="button" aria-label="응답 대기 취소" @click="chatController?.abort()"><span aria-hidden="true">■</span></button>
              <button v-else class="send" type="submit" aria-label="질문 보내기" :disabled="checking || !question.trim() || !configured"><span aria-hidden="true">↑</span></button>
            </div>
          </form>
          <p v-if="messages.length" class="footnote">검색 근거와 엔진 버전을 확인한 뒤 프로젝트에 적용하세요.</p>
        </div>
      </section>

      <section v-show="tab === 'knowledge'" class="knowledge-workspace" aria-labelledby="knowledge-title">
        <div class="page-title"><div><p class="eyebrow">SHARED KNOWLEDGE</p><h1 id="knowledge-title">프로젝트 지식 저장소</h1><p>한 번의 해결이 팀의 다음 답이 되도록, 개발 경험을 기록하세요.</p></div><span class="page-index" aria-hidden="true">Library ↗</span></div>
        <div class="knowledge-layout">
          <form class="knowledge-form" aria-labelledby="entry-form-title" @submit.prevent="saveEntry">
            <h2 id="entry-form-title">{{ editing ? '지식 수정' : '지식 등록' }}</h2>
            <fieldset :disabled="mutating || !accessReady">
            <label for="kind">기록 종류</label><select id="kind" v-model="entry.kind"><option value="technique">기술 / 개발 방법</option><option value="error">오류 / 해결 사례</option></select>
            <label for="title">제목</label><input id="title" v-model="entry.title" required maxlength="200" placeholder="나중에 찾기 쉬운 제목" />
            <label for="version">엔진 버전</label><input id="version" v-model="entry.engine_version" required maxlength="50" placeholder="예: 5.5.4" />
            <p class="hint">확인한 패치 버전까지 기록하세요. 검색은 같은 마이너 계열로 묶고 출처에는 입력한 버전을 유지합니다.</p>
            <label for="context">환경 / 전제 조건 <span class="optional">선택</span></label><textarea id="context" v-model="entry.context" rows="2" maxlength="3000" placeholder="플랫폼, 플러그인, 빌드 설정 등"></textarea>
            <label for="content">{{ entry.kind === 'error' ? '오류 메시지 / 증상' : '기술 내용 / 적용 방법' }}</label><textarea id="content" v-model="entry.content" rows="5" required maxlength="20000" placeholder="패치별 영향이나 수정 시점이 확인되면 해당 버전과 근거를 함께 기록하세요."></textarea>
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
          <section class="settings-panel" aria-labelledby="access-title">
            <div class="panel-head">
              <div class="panel-title">
                <h2 id="access-title">내부 연결</h2>
                <span class="help">
                  <button type="button" class="help-button" aria-label="내부 연결 도움말" aria-describedby="access-help">?</button>
                  <span id="access-help" role="tooltip" class="help-tip">LLM API 키와 별도이며, 서버에 토큰을 설정한 경우에만 입력합니다. 다시 연결하거나 해제하면 대화와 작성 중인 내용을 비웁니다.</span>
                </span>
              </div>
              <p class="connection-state"><span class="status-dot" :class="{ ready: accessReady }"></span>{{ connecting ? '연결 중…' : accessReady ? '연결됨' : '연결 필요' }}</p>
            </div>
            <form class="panel-row" aria-describedby="access-help" @submit.prevent="connect">
              <div class="connection-field"><label for="access-token">접근 토큰</label>
              <input id="access-token" v-model="accessToken" type="password" autocomplete="off" spellcheck="false" maxlength="4096" placeholder="서버에 설정된 경우 입력" :disabled="connecting || busy || mutating || loading" />
              </div><button class="primary" type="submit" :disabled="connecting || busy || mutating || loading">{{ connecting ? '연결 중…' : '연결' }}</button>
              <button type="button" :disabled="!accessReady || connecting || busy || mutating || loading" @click="disconnect">연결 해제</button>
            </form>
            <p v-if="accessError" class="panel-note error" role="alert">{{ accessError }}</p>
          </section>
          <section class="settings-panel" aria-labelledby="connection-title">
            <div class="panel-head">
              <div class="panel-title">
                <h2 id="connection-title">LLM 연결</h2>
                <span class="help">
                  <button type="button" class="help-button" aria-label="LLM 연결 도움말" aria-describedby="llm-help">?</button>
                  <span id="llm-help" role="tooltip" class="help-tip">키는 새로고침하거나 연결을 해제하면 지워집니다. 연결 확인은 짧은 LLM 요청을 보내 사용량이 발생합니다.</span>
                </span>
              </div>
              <p class="connection-state"><span class="status-dot" :class="{ ready: configured }"></span>{{ apiKey.trim() ? '입력한 키 사용' : serverKey ? '서버에 설정된 키 사용' : 'API 키 입력 필요' }}</p>
            </div>
            <div class="panel-row">
              <div class="connection-field"><label for="api-key">API 키</label>
              <span class="secret-field">
                <input id="api-key" v-model="apiKey" :type="showApiKey ? 'text' : 'password'" autocomplete="off" spellcheck="false" placeholder="API 키 입력" maxlength="4096" :disabled="!accessReady || checking" aria-describedby="llm-help" />
                <button type="button" class="reveal-button" :aria-label="showApiKey ? 'API 키 숨기기' : 'API 키 보기'" :aria-pressed="showApiKey" :disabled="!accessReady" @click="showApiKey = !showApiKey">
                  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
                    <path d="M2 12s3.6-7 10-7 10 7 10 7-3.6 7-10 7S2 12 2 12Z" /><circle cx="12" cy="12" r="3" />
                    <path v-if="showApiKey" d="M3 3l18 18" />
                  </svg>
                </button>
              </span>
              </div><div class="connection-field"><label for="model">모델</label>
              <input id="model" v-model="model" class="model-input" placeholder="모델 이름" maxlength="100" autocomplete="off" :disabled="!accessReady || checking" />
              </div><button class="primary" type="button" :disabled="!configured || busy || mutating || loading" aria-describedby="llm-help" @click="checkLLM">{{ checking ? '확인 중…' : '연결 확인' }}</button>
            </div>
            <p v-if="connectionError" class="panel-note error" role="alert">{{ connectionError }}</p>
            <p v-else-if="connectionNotice" class="panel-note success" role="status">{{ connectionNotice }}</p>
          </section>
        </div>
      </section>
    </main>
  </div>
</template>
