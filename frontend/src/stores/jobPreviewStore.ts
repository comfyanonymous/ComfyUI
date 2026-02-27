import { defineStore } from 'pinia'
import { computed, readonly, ref, watch } from 'vue'

import { useSettingStore } from '@/platform/settings/settingStore'
import {
  releaseSharedObjectUrl,
  retainSharedObjectUrl
} from '@/utils/objectUrlUtil'

type PromptPreviewMap = Record<string, string>

export const useJobPreviewStore = defineStore('jobPreview', () => {
  const settingStore = useSettingStore()
  const previewsByPromptId = ref<PromptPreviewMap>({})
  const readonlyPreviewsByPromptId = readonly(previewsByPromptId)

  const previewMethod = computed(() =>
    settingStore.get('Comfy.Execution.PreviewMethod')
  )
  const isPreviewEnabled = computed(() => previewMethod.value !== 'none')

  function setPreviewUrl(promptId: string | undefined, url: string) {
    if (!promptId || !isPreviewEnabled.value) return
    const current = previewsByPromptId.value[promptId]
    if (current === url) return
    if (current) releaseSharedObjectUrl(current)
    retainSharedObjectUrl(url)
    previewsByPromptId.value = {
      ...previewsByPromptId.value,
      [promptId]: url
    }
  }

  function clearPreview(promptId: string | undefined) {
    if (!promptId) return
    const current = previewsByPromptId.value[promptId]
    if (!current) return
    releaseSharedObjectUrl(current)
    const next = { ...previewsByPromptId.value }
    delete next[promptId]
    previewsByPromptId.value = next
  }

  function clearAllPreviews() {
    Object.values(previewsByPromptId.value).forEach((url) => {
      releaseSharedObjectUrl(url)
    })
    previewsByPromptId.value = {}
  }

  watch(isPreviewEnabled, (enabled) => {
    if (!enabled) clearAllPreviews()
  })

  return {
    previewsByPromptId: readonlyPreviewsByPromptId,
    isPreviewEnabled,
    setPreviewUrl,
    clearPreview,
    clearAllPreviews
  }
})
