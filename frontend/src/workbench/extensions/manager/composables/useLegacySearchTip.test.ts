import type { Ref } from 'vue'
import { beforeEach, describe, expect, it } from 'vitest'
import { nextTick, ref } from 'vue'

import { useLegacySearchTip } from './useLegacySearchTip'

describe('useLegacySearchTip', () => {
  let searchQuery: Ref<string>
  let isNewManagerUI: Ref<boolean>

  beforeEach(() => {
    searchQuery = ref('')
    isNewManagerUI = ref(true)
  })

  describe('isLegacyManagerSearch', () => {
    it('returns true when searching "manager" in new manager UI', async () => {
      const { isLegacyManagerSearch } = useLegacySearchTip(
        searchQuery,
        isNewManagerUI
      )

      searchQuery.value = 'manager'
      await nextTick()

      expect(isLegacyManagerSearch.value).toBe(true)
    })

    it('returns true when searching "comfyui-manager"', async () => {
      const { isLegacyManagerSearch } = useLegacySearchTip(
        searchQuery,
        isNewManagerUI
      )

      searchQuery.value = 'comfyui-manager'
      await nextTick()

      expect(isLegacyManagerSearch.value).toBe(true)
    })

    it('returns true when searching "comfyui manager" (space variant)', async () => {
      const { isLegacyManagerSearch } = useLegacySearchTip(
        searchQuery,
        isNewManagerUI
      )

      searchQuery.value = 'comfyui manager'
      await nextTick()

      expect(isLegacyManagerSearch.value).toBe(true)
    })

    it('returns true when keyword is part of larger query', async () => {
      const { isLegacyManagerSearch } = useLegacySearchTip(
        searchQuery,
        isNewManagerUI
      )

      searchQuery.value = 'where is manager'
      await nextTick()

      expect(isLegacyManagerSearch.value).toBe(true)
    })

    it('matches case-insensitively', async () => {
      const { isLegacyManagerSearch } = useLegacySearchTip(
        searchQuery,
        isNewManagerUI
      )

      searchQuery.value = 'MANAGER'
      await nextTick()

      expect(isLegacyManagerSearch.value).toBe(true)
    })

    it('returns false when query is empty', async () => {
      const { isLegacyManagerSearch } = useLegacySearchTip(
        searchQuery,
        isNewManagerUI
      )

      searchQuery.value = ''
      await nextTick()

      expect(isLegacyManagerSearch.value).toBe(false)
    })

    it('returns false when query is whitespace only', async () => {
      const { isLegacyManagerSearch } = useLegacySearchTip(
        searchQuery,
        isNewManagerUI
      )

      searchQuery.value = '   '
      await nextTick()

      expect(isLegacyManagerSearch.value).toBe(false)
    })

    it('returns false when searching unrelated terms', async () => {
      const { isLegacyManagerSearch } = useLegacySearchTip(
        searchQuery,
        isNewManagerUI
      )

      searchQuery.value = 'controlnet'
      await nextTick()

      expect(isLegacyManagerSearch.value).toBe(false)
    })

    it('returns false when isNewManagerUI is false', async () => {
      isNewManagerUI.value = false
      const { isLegacyManagerSearch } = useLegacySearchTip(
        searchQuery,
        isNewManagerUI
      )

      searchQuery.value = 'manager'
      await nextTick()

      expect(isLegacyManagerSearch.value).toBe(false)
    })
  })
})
