import { computed, ref, watch } from 'vue'

export const CONVERSATION_STORAGE_KEY = 'tsukuyomi.conversations.v1'
const emptyDraft = (engineVersion = '') => ({ draft: '', engineVersion, error: '' })
const sourceFields = ['number', 'title', 'engine_version', 'page', 'url', 'excerpt', 'excerpt_truncated', 'source_type', 'published_date', 'retrieved_at']

function storedMessage(message) {
  if (!['user', 'assistant'].includes(message?.role) || typeof message.content !== 'string') {
    throw new Error('Invalid conversation message')
  }
  return {
    role: message.role, content: message.content,
    engineVersion: typeof message.engineVersion === 'string' ? message.engineVersion : null,
    failed: Boolean(message.failed),
    warnings: Array.isArray(message.warnings) ? message.warnings.filter(value => typeof value === 'string') : [],
    status: ['done', 'insufficient', 'invalid_citations'].includes(message.status) ? message.status : undefined,
    sources: Array.isArray(message.sources) ? message.sources.filter(source => source && typeof source.title === 'string')
      .map(source => Object.fromEntries(sourceFields.filter(key => key in source).map(key => [key, source[key]]))) : [],
    answerBlocks: Array.isArray(message.answerBlocks) ? message.answerBlocks
      .filter(block => ['text', 'code'].includes(block?.type) && typeof block.content === 'string')
      .map(block => ({ type: block.type, content: block.content, language: typeof block.language === 'string' ? block.language : '' })) : [],
  }
}

function storedConversation(item) {
  if (typeof item?.id !== 'string' || !item.id || !Array.isArray(item.messages)) {
    throw new Error('Invalid conversation')
  }
  return {
    id: item.id,
    title: typeof item.title === 'string' ? item.title : '이전 대화',
    messages: item.messages.map(storedMessage),
    draft: typeof item.draft === 'string' ? item.draft : '',
    engineVersion: typeof item.engineVersion === 'string' ? item.engineVersion : '',
    error: typeof item.error === 'string' ? item.error : '',
    updatedAt: Number.isFinite(item.updatedAt) ? item.updatedAt : 0,
  }
}

// The store receives conversation data only; connection keys and tokens stay in App.vue.
export function createConversationStore(getStorage = () => window.localStorage) {
  const conversations = ref([])
  const activeId = ref(null)
  const blank = ref(emptyDraft())
  const storageWarning = ref('')
  let storageEnabled = false
  const active = computed(() => conversations.value.find(item => item.id === activeId.value) || null)
  const current = computed(() => active.value || blank.value)
  const messages = computed(() => active.value?.messages || [])
  const recent = computed(() => [...conversations.value].sort((a, b) => b.updatedAt - a.updatedAt))
  const field = key => computed({ get: () => current.value[key], set: value => { current.value[key] = value } })

  function persist() {
    if (!storageEnabled) return
    try {
      getStorage().setItem(CONVERSATION_STORAGE_KEY, JSON.stringify({
        version: 1, activeId: activeId.value, conversations: conversations.value.map(storedConversation),
      }))
      storageWarning.value = ''
    } catch {
      storageWarning.value = '대화 기록을 이 브라우저에 저장하지 못했습니다. 저장 공간과 브라우저 설정을 확인하세요. 현재 탭에서는 계속 대화할 수 있습니다.'
    }
  }
  // Synchronous writes retain the latest draft even when reloading immediately after typing.
  watch([conversations, activeId], persist, { deep: true, flush: 'sync' })

  function clearVisible() {
    storageEnabled = false
    conversations.value = []
    activeId.value = null
    blank.value = emptyDraft()
    storageWarning.value = ''
  }

  function load() {
    clearVisible()
    try {
      const raw = getStorage().getItem(CONVERSATION_STORAGE_KEY)
      if (raw) {
        const saved = JSON.parse(raw)
        if (saved?.version !== 1 || !Array.isArray(saved.conversations)) throw new Error('Invalid saved conversations')
        const loaded = saved.conversations.map(storedConversation)
        if (new Set(loaded.map(item => item.id)).size !== loaded.length) throw new Error('Duplicate conversation IDs')
        for (const item of loaded) {
          const last = item.messages.at(-1)
          if (last?.role === 'user' && !last.failed) {
            last.failed = true
            item.draft ||= last.content
            item.error = '이전 응답을 받기 전에 대화가 중단되었습니다. 질문을 다시 보내 주세요.'
          }
        }
        conversations.value = loaded
        activeId.value = loaded.some(item => item.id === saved.activeId) ? saved.activeId : null
      }
      storageEnabled = true
    } catch {
      storageWarning.value = '저장된 대화 기록을 읽지 못했습니다. 브라우저 저장소를 확인하세요. 새 대화는 현재 탭에서 계속할 수 있습니다.'
    }
  }

  function ensureConversation(text) {
    if (!active.value) {
      const id = globalThis.crypto?.randomUUID?.() || `chat-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`
      const item = { id, title: text.replace(/\s+/g, ' ').slice(0, 60),
        messages: [], ...blank.value, updatedAt: Date.now() }
      conversations.value.unshift(item)
      activeId.value = item.id
    }
    active.value.updatedAt = Date.now()
    return active.value
  }

  function startNew() {
    const version = current.value.engineVersion
    activeId.value = null
    blank.value = emptyDraft(version)
  }

  function open(id) {
    if (conversations.value.some(item => item.id === id)) activeId.value = id
  }

  function remove(id) {
    if (activeId.value === id) startNew()
    conversations.value = conversations.value.filter(item => item.id !== id)
  }

  return { conversations, activeId, messages, recent, storageWarning,
    question: field('draft'), engineVersion: field('engineVersion'), chatError: field('error'),
    load, clearVisible, ensureConversation, startNew, open, remove }
}
