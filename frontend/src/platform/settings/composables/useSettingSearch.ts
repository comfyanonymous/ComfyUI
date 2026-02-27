import { computed, ref, watch } from 'vue'

import { st } from '@/i18n'
import type { SettingTreeNode } from '@/platform/settings/settingStore'
import {
  getSettingInfo,
  useSettingStore
} from '@/platform/settings/settingStore'
import type { ISettingGroup, SettingParams } from '@/platform/settings/types'
import { normalizeI18nKey } from '@/utils/formatUtil'
import { useVueFeatureFlags } from '@/composables/useVueFeatureFlags'

interface SearchableNavItem {
  key: string
  label: string
}

export function useSettingSearch() {
  const settingStore = useSettingStore()
  const { shouldRenderVueNodes } = useVueFeatureFlags()

  const searchQuery = ref<string>('')
  const filteredSettingIds = ref<string[]>([])
  const matchedNavItemKeys = ref<Set<string>>(new Set())
  const searchInProgress = ref<boolean>(false)

  watch(searchQuery, () => (searchInProgress.value = true))

  /**
   * Settings categories that contains at least one setting in search results.
   */
  const searchResultsCategories = computed<Set<string>>(() => {
    return new Set(
      filteredSettingIds.value.map(
        (id) => getSettingInfo(settingStore.settingsById[id]).category
      )
    )
  })

  /**
   * Check if the search query is empty
   */
  const queryIsEmpty = computed(() => searchQuery.value.length === 0)

  /**
   * Check if we're in search mode
   */
  const inSearch = computed(
    () => !queryIsEmpty.value && !searchInProgress.value
  )

  /**
   * Handle search functionality
   */
  const handleSearch = (query: string, navItems?: SearchableNavItem[]) => {
    matchedNavItemKeys.value = new Set()

    if (!query) {
      filteredSettingIds.value = []
      return
    }

    const queryLower = query.toLocaleLowerCase()
    const allSettings = Object.values(settingStore.settingsById)
    const filteredSettings = allSettings.filter((setting) => {
      // Filter out hidden and deprecated settings, just like in normal settings tree
      if (
        setting.type === 'hidden' ||
        setting.deprecated ||
        (shouldRenderVueNodes.value && setting.hideInVueNodes)
      ) {
        return false
      }

      const idLower = setting.id.toLowerCase()
      const nameLower = setting.name.toLowerCase()
      const translatedName = st(
        `settings.${normalizeI18nKey(setting.id)}.name`,
        setting.name
      ).toLocaleLowerCase()
      const info = getSettingInfo(setting)
      const translatedCategory = st(
        `settingsCategories.${normalizeI18nKey(info.category)}`,
        info.category
      ).toLocaleLowerCase()
      const translatedSubCategory = st(
        `settingsCategories.${normalizeI18nKey(info.subCategory)}`,
        info.subCategory
      ).toLocaleLowerCase()

      return (
        idLower.includes(queryLower) ||
        nameLower.includes(queryLower) ||
        translatedName.includes(queryLower) ||
        translatedCategory.includes(queryLower) ||
        translatedSubCategory.includes(queryLower)
      )
    })

    if (navItems) {
      for (const item of navItems) {
        if (
          item.key.toLocaleLowerCase().includes(queryLower) ||
          item.label.toLocaleLowerCase().includes(queryLower)
        ) {
          matchedNavItemKeys.value.add(item.key)
        }
      }
    }

    filteredSettingIds.value = filteredSettings.map((x) => x.id)
    searchInProgress.value = false
  }

  /**
   * Get search results grouped by category
   */
  const getSearchResults = (
    activeCategory: SettingTreeNode | null
  ): ISettingGroup[] => {
    const groupedSettings: {
      [key: string]: { category: string; settings: SettingParams[] }
    } = {}

    filteredSettingIds.value.forEach((id) => {
      const setting = settingStore.settingsById[id]
      const info = getSettingInfo(setting)
      const groupKey =
        activeCategory === null
          ? `${info.category}/${info.subCategory}`
          : info.subCategory

      if (activeCategory === null || activeCategory.label === info.category) {
        if (!groupedSettings[groupKey]) {
          groupedSettings[groupKey] = {
            category: info.category,
            settings: []
          }
        }
        groupedSettings[groupKey].settings.push(setting)
      }
    })

    return Object.entries(groupedSettings).map(
      ([key, { category, settings }]) => ({
        label: activeCategory === null ? key.split('/')[1] : key,
        ...(activeCategory === null ? { category } : {}),
        settings
      })
    )
  }

  return {
    searchQuery,
    filteredSettingIds,
    matchedNavItemKeys,
    searchInProgress,
    searchResultsCategories,
    queryIsEmpty,
    inSearch,
    handleSearch,
    getSearchResults
  }
}
