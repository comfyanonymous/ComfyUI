import type { Ref } from 'vue'
import { computed } from 'vue'

const LEGACY_MANAGER_KEYWORDS = [
  'manager',
  'comfyui-manager',
  'manager comfyui',
  'comfyui manager'
] as const

export function useLegacySearchTip(
  searchQuery: Ref<string>,
  isNewManagerUI: Ref<boolean>
) {
  const isLegacyManagerSearch = computed(() => {
    if (!isNewManagerUI.value) return false
    const query = searchQuery.value.toLowerCase().trim()
    if (!query) return false
    return LEGACY_MANAGER_KEYWORDS.some((keyword) => query.includes(keyword))
  })

  return {
    isLegacyManagerSearch
  }
}
