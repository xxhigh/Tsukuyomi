<script setup>
import { computed, nextTick, reactive, ref } from 'vue'
import { ENGINE_VERSIONS } from './engineVersions.js'

const props = defineProps({ entries: Array, request: Function, disabled: Boolean, error: String })
const emit = defineEmits(['busy', 'saved'])
const editing = ref(null)
const saving = ref(false)
const error = ref('')
const notice = ref('')
const UNSPECIFIED = '미지정'
const form = reactive({ title: '', versions: [], source_url: '' })
// 목록에 없는 기존 버전(예: 5.5.4)도 선택지로 남겨 편집 중에 사라지지 않게 한다.
const versionOptions = computed(() => [...ENGINE_VERSIONS, ...form.versions.filter(version => !ENGINE_VERSIONS.includes(version))])

function splitVersions(value) {
  return (value || '').split(',').map(version => version.trim()).filter(version => version && version !== UNSPECIFIED)
}

async function edit(item) {
  editing.value = { filename: item.filename, revision: item.revision }
  Object.assign(form, { title: item.title, versions: splitVersions(item.engine_version), source_url: item.source_url || '' })
  error.value = ''
  notice.value = ''
  await nextTick()
  document.getElementById('document-title')?.focus()
}

function cancel() {
  editing.value = null
  Object.assign(form, { title: '', versions: [], source_url: '' })
  error.value = ''
}

async function save() {
  if (props.disabled || saving.value || !editing.value) return
  saving.value = true
  emit('busy', true)
  error.value = ''
  notice.value = ''
  try {
    const result = await props.request(`/documents/${encodeURIComponent(editing.value.filename)}`, {
      method: 'PUT', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        title: form.title,
        engine_version: versionOptions.value.filter(version => form.versions.includes(version)).join(', ') || UNSPECIFIED,
        source_url: form.source_url.trim() || null,
        expected_revision: editing.value.revision,
      }),
    })
    emit('saved', result.entry)
    cancel()
    notice.value = result.message
  } catch (failure) {
    if (!failure.accessDenied) error.value = failure.message
  } finally {
    saving.value = false
    emit('busy', false)
  }
}
</script>

<template>
  <section class="document-library" aria-labelledby="documents-title">
    <h2 id="documents-title">PDF · Markdown 문서 <span>{{ entries.length }}</span></h2>
    <p class="hint">서버에 추가한 참고 문서입니다. 원본은 유지하고 검색에 사용할 제목·버전·출처를 편집합니다.</p>
    <p v-if="props.error" class="error" role="alert">{{ props.error }}</p>
    <p v-else-if="!entries.length" class="hint">표시할 문서가 없습니다. 연결 후 서버의 문서 폴더를 확인하세요.</p>
    <ul class="document-list">
      <li v-for="item in entries" :key="item.filename" class="document-entry">
        <div><strong>{{ item.title }}</strong><p class="hint">{{ item.filename }} · UE {{ item.engine_version }}</p></div>
        <button type="button" :disabled="disabled" :aria-label="`${item.filename} 정보 편집`" @click="edit(item)">정보 편집</button>
      </li>
    </ul>
    <form v-if="editing" class="knowledge-form document-form" aria-label="문서 정보 편집" @submit.prevent="save">
      <h3>{{ editing.filename }}</h3>
      <fieldset :disabled="disabled">
        <label for="document-title">문서 제목</label><input id="document-title" v-model="form.title" required maxlength="200" />
        <fieldset class="version-group" aria-describedby="document-version-help">
          <legend>문서 엔진 버전 <span class="optional">여러 개 선택 가능</span></legend>
          <div class="version-options">
            <label v-for="version in versionOptions" :key="version" class="version-option">
              <input v-model="form.versions" type="checkbox" :value="version" /><span>UE {{ version }}</span>
            </label>
          </div>
          <p id="document-version-help" class="hint">확인된 버전만 선택하세요. 선택하지 않으면 ‘미지정’으로 저장합니다.</p>
        </fieldset>
        <label for="document-source">문서 출처 URL <span class="optional">선택</span></label><input id="document-source" v-model="form.source_url" type="url" maxlength="2000" placeholder="https://…" />
        <div class="entry-actions">
          <button class="primary" type="submit">{{ saving ? '저장 중…' : '문서 정보 저장' }}</button>
          <button type="button" @click="cancel">문서 편집 취소</button>
        </div>
      </fieldset>
    </form>
    <p v-if="error" class="error" role="alert">{{ error }}</p>
    <p v-if="notice" class="success" role="status">{{ notice }}</p>
  </section>
</template>

<style scoped>
.document-library { margin-top: 40px; padding-top: 24px; border-top: 1px solid var(--color-rule); }
.document-library h2 span { font-size: 12px; color: var(--color-muted); }
.document-list { list-style: none; margin: 16px 0; padding: 0; }
.document-entry { display: flex; align-items: start; gap: 16px; justify-content: space-between; padding: 14px 0; border-bottom: 1px solid var(--color-rule); }
.document-entry div { min-width: 0; overflow-wrap: anywhere; }
.document-entry button { flex-shrink: 0; font-size: 12px; }
.document-form { max-width: 600px; }
.document-form h3 { overflow-wrap: anywhere; }
.version-group { margin: 16px 0 0; }
.version-group legend { padding: 0; margin-bottom: 8px; font-size: 13px; font-weight: 500; }
.version-options { display: flex; flex-wrap: wrap; gap: 8px; }
.version-option { display: inline-flex; align-items: center; gap: 6px; margin: 0; padding: 6px 12px; border: 1px solid var(--color-rule); border-radius: 999px; font-size: 12px; font-weight: 400; cursor: pointer; transition: background 150ms ease-out, border-color 150ms ease-out; }
.version-option:hover { border-color: var(--color-muted); }
.version-option:has(input:checked) { border-color: var(--color-accent); background: var(--color-accent-soft); color: var(--color-accent); }
.version-option input { width: auto; margin: 0; accent-color: var(--color-accent); }
@media (max-width: 480px) { .document-entry { flex-direction: column; gap: 4px; } }
</style>
