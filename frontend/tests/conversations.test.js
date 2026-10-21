import test from 'node:test'
import assert from 'node:assert/strict'
import { createConversationStore, CONVERSATION_STORAGE_KEY } from '../src/conversations.js'

function storage() {
  const entries = new Map()
  return { getItem: key => entries.get(key) ?? null, setItem: (key, value) => entries.set(key, value) }
}

function ask(store, title, version = '5.4') {
  store.engineVersion.value = version
  store.question.value = title
  const conversation = store.ensureConversation(title)
  conversation.messages.push({ role: 'user', content: title, engineVersion: version })
  store.question.value = ''
  return conversation
}

function answer(conversation, content = '확인한 기록 [1]') {
  conversation.messages.push({ role: 'assistant', content, engineVersion: conversation.engineVersion,
    status: 'done', sources: [{ number: 1, title: 'Patch source', engine_version: '5.4.2', excerpt: '검색 원문', excerpt_truncated: false }],
    answerBlocks: [{ type: 'code', content: 'Values[0]', language: 'cpp' }] })
}

test('new conversations preserve previous messages, draft, version and error', () => {
  const disk = storage()
  const store = createConversationStore(() => disk)
  store.load()
  const first = ask(store, '첫 번째 대화')
  answer(first)
  store.question.value = '이어서 작성하는 질문'
  store.chatError.value = '이전 오류 안내'
  store.startNew()
  assert.deepEqual(store.messages.value, [])
  assert.equal(store.question.value, '')
  const second = ask(store, '두 번째 대화', '5.6')
  answer(second)
  assert.notEqual(first.id, second.id)
  store.open(first.id)
  assert.equal(store.question.value, '이어서 작성하는 질문')
  assert.equal(store.engineVersion.value, '5.4')
  assert.equal(store.chatError.value, '이전 오류 안내')
  assert.equal(store.messages.value.length, 2)
  assert.equal(store.messages.value[0].content, '첫 번째 대화')
})

test('reload restores active conversation, draft, original source and code', () => {
  const disk = storage()
  const store = createConversationStore(() => disk)
  store.load()
  const first = ask(store, '질문')
  answer(first)
  store.question.value = '저장할 초안'
  const restored = createConversationStore(() => disk)
  restored.load()
  assert.equal(restored.activeId.value, first.id)
  assert.equal(restored.question.value, '저장할 초안')
  assert.equal(restored.messages.value[1].sources[0].engine_version, '5.4.2')
  assert.equal(restored.messages.value[1].sources[0].excerpt, '검색 원문')
  assert.deepEqual(restored.messages.value[1].answerBlocks, [{ type: 'code', content: 'Values[0]', language: 'cpp' }])
})

test('starting a new conversation and reloading retains the old list without reopening it', () => {
  const disk = storage()
  const store = createConversationStore(() => disk)
  store.load()
  const item = ask(store, '완료한 대화')
  answer(item)
  store.startNew()
  const restored = createConversationStore(() => disk)
  restored.load()
  assert.equal(restored.activeId.value, null)
  assert.deepEqual(restored.messages.value, [])
  assert.equal(restored.recent.value[0].id, item.id)
})

test('hiding records on disconnect does not overwrite saved data', () => {
  const disk = storage()
  const store = createConversationStore(() => disk)
  store.load()
  answer(ask(store, '보관할 대화'))
  const saved = disk.getItem(CONVERSATION_STORAGE_KEY)
  store.clearVisible()
  assert.deepEqual(store.recent.value, [])
  assert.equal(disk.getItem(CONVERSATION_STORAGE_KEY), saved)
  store.load()
  assert.equal(store.messages.value[0].content, '보관할 대화')
})

test('deleting the active conversation persists deletion and keeps other conversations', () => {
  const disk = storage()
  const store = createConversationStore(() => disk)
  store.load()
  const first = ask(store, '남길 대화')
  answer(first)
  store.startNew()
  const second = ask(store, '지울 대화')
  answer(second)
  store.remove(second.id)
  assert.equal(store.activeId.value, null)
  const restored = createConversationStore(() => disk)
  restored.load()
  assert.deepEqual(restored.recent.value.map(item => item.id), [first.id])
})

test('reload turns an interrupted request into a failed turn and restores its question', () => {
  const disk = storage()
  const store = createConversationStore(() => disk)
  store.load()
  ask(store, '전송 중 중단된 질문')
  const restored = createConversationStore(() => disk)
  restored.load()
  assert.equal(restored.messages.value[0].failed, true)
  assert.equal(restored.question.value, '전송 중 중단된 질문')
  assert.match(restored.chatError.value, /중단/)
})

test('failed turns remain failed after reload', () => {
  const disk = storage()
  const store = createConversationStore(() => disk)
  store.load()
  const item = ask(store, '실패한 질문')
  item.messages[0].failed = true
  store.question.value = '실패한 질문'
  const restored = createConversationStore(() => disk)
  restored.load()
  assert.equal(restored.messages.value[0].failed, true)
  assert.equal(restored.question.value, '실패한 질문')
})

test('saving whitelists conversation fields and drops unrelated connection credentials', () => {
  const disk = storage()
  const store = createConversationStore(() => disk)
  store.load()
  const item = ask(store, '안전한 질문')
  item.apiKey = 'provider-secret'
  item.accessToken = 'access-secret'
  item.messages[0].apiKey = 'message-secret'
  answer(item)
  item.messages[1].sources[0].apiKey = 'source-secret'
  const saved = disk.getItem(CONVERSATION_STORAGE_KEY)
  assert.ok(!saved.includes('secret'))
})

test('quota errors keep in-memory records and recover after a later successful write', () => {
  const disk = storage()
  let full = true
  const adapter = { getItem: disk.getItem, setItem: (key, value) => {
    if (full) throw new Error('QuotaExceededError')
    disk.setItem(key, value)
  } }
  const store = createConversationStore(() => adapter)
  store.load()
  const item = ask(store, '탭에서 계속 대화')
  answer(item)
  assert.match(store.storageWarning.value, /저장하지 못했습니다/)
  assert.equal(store.messages.value.length, 2)
  full = false
  store.question.value = '다시 저장'
  assert.equal(store.storageWarning.value, '')
  assert.ok(disk.getItem(CONVERSATION_STORAGE_KEY))
})

for (const raw of ['broken', '{"version":2,"conversations":[]}', '{"version":1,"conversations":[{}]}']) {
  test(`invalid saved records show a warning and are not silently overwritten: ${raw}`, () => {
    const disk = storage()
    disk.setItem(CONVERSATION_STORAGE_KEY, raw)
    const store = createConversationStore(() => disk)
    store.load()
    assert.match(store.storageWarning.value, /읽지 못했습니다/)
    answer(ask(store, '새 대화'))
    assert.equal(disk.getItem(CONVERSATION_STORAGE_KEY), raw)
  })
}

test('disabled browser storage does not prevent using the current tab', () => {
  const store = createConversationStore(() => { throw new Error('SecurityError') })
  store.load()
  answer(ask(store, '현재 탭에서 대화'))
  assert.equal(store.messages.value.length, 2)
  assert.ok(store.storageWarning.value)
})

test('conversation creation also works when randomUUID is unavailable on an HTTP origin', t => {
  t.mock.method(globalThis.crypto, 'randomUUID', () => undefined)
  const disk = storage()
  const store = createConversationStore(() => disk)
  store.load()
  const first = ask(store, '첫 대화')
  answer(first)
  store.startNew()
  const second = ask(store, '다음 대화')
  assert.ok(first.id)
  assert.notEqual(first.id, second.id)
})

test('web source provenance and search warnings survive reload', () => {
  const disk = storage()
  const store = createConversationStore(() => disk)
  store.load()
  const item = ask(store, '공식 웹 문서 검색')
  answer(item)
  item.messages[1].sources[0].source_type = 'web'
  item.messages[1].sources[0].url = 'https://dev.epicgames.com/documentation'
  item.messages[1].sources[0].retrieved_at = '2026-10-21T00:00:00+00:00'
  item.messages[1].sources[0].published_date = '2026-01-01'
  item.messages[1].warnings = ['웹 검색 일부 실패']
  const restored = createConversationStore(() => disk)
  restored.load()
  assert.equal(restored.messages.value[1].sources[0].source_type, 'web')
  assert.equal(restored.messages.value[1].sources[0].retrieved_at, '2026-10-21T00:00:00+00:00')
  assert.equal(restored.messages.value[1].sources[0].published_date, '2026-01-01')
  assert.deepEqual(restored.messages.value[1].warnings, ['웹 검색 일부 실패'])
})
