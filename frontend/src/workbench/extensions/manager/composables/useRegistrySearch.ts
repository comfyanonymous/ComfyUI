import { watchDebounced } from '@vueuse/core'
import { orderBy } from 'es-toolkit/compat'
import { computed, ref, watch } from 'vue'

import { DEFAULT_PAGE_SIZE } from '@/constants/searchConstants'
import { useRegistrySearchGateway } from '@/services/gateway/registrySearchGateway'
import type { SearchAttribute } from '@/types/algoliaTypes'
import type { components } from '@/types/comfyRegistryTypes'
import type { QuerySuggestion, SearchMode } from '@/types/searchServiceTypes'
import { SortableAlgoliaField } from '@/workbench/extensions/manager/types/comfyManagerTypes'

type RegistryNodePack = components['schemas']['Node']

const SEARCH_DEBOUNCE_TIME = 320
const DEFAULT_SORT_FIELD = SortableAlgoliaField.Downloads // Set in the index configuration

/**
 * Composable for managing UI state of Comfy Node Registry search.
 */
export function useRegistrySearch(
  options: {
    initialSortField?: string
    initialSearchMode?: SearchMode
    initialSearchQuery?: string
    initialPageNumber?: number
  } = {}
) {
  const {
    initialSortField = DEFAULT_SORT_FIELD,
    initialSearchMode = 'packs',
    initialSearchQuery = '',
    initialPageNumber = 0
  } = options

  const isLoading = ref(false)
  const sortField = ref<string>(initialSortField)
  const searchMode = ref<SearchMode>(initialSearchMode)
  const pageSize = ref(DEFAULT_PAGE_SIZE)
  const pageNumber = ref(initialPageNumber)
  const searchQuery = ref(initialSearchQuery)
  const searchResults = ref<RegistryNodePack[]>([])
  const suggestions = ref<QuerySuggestion[]>([])

  const searchAttributes = computed<SearchAttribute[]>(() =>
    searchMode.value === 'nodes' ? ['comfy_nodes'] : ['name', 'description']
  )

  const searchGateway = useRegistrySearchGateway()

  const { searchPacks, clearSearchCache, getSortValue, getSortableFields } =
    searchGateway

  const updateSearchResults = async (options: { append?: boolean }) => {
    isLoading.value = true
    if (!options.append) {
      pageNumber.value = 0
    }
    const { nodePacks, querySuggestions } = await searchPacks(
      searchQuery.value,
      {
        pageSize: pageSize.value,
        pageNumber: pageNumber.value,
        restrictSearchableAttributes: searchAttributes.value
      }
    )

    let sortedPacks = nodePacks

    // Results are sorted by the default field to begin with -- so don't manually sort again
    if (sortField.value && sortField.value !== DEFAULT_SORT_FIELD) {
      // Get the sort direction from the provider's sortable fields
      const sortableFields = getSortableFields()
      const fieldConfig = sortableFields.find((f) => f.id === sortField.value)
      const direction = fieldConfig?.direction || 'desc'

      sortedPacks = orderBy(
        nodePacks,
        [(pack) => getSortValue(pack, sortField.value)],
        [direction]
      )
    }

    if (options.append && searchResults.value?.length) {
      searchResults.value = searchResults.value.concat(sortedPacks)
    } else {
      searchResults.value = sortedPacks
    }
    suggestions.value = querySuggestions
    isLoading.value = false
  }

  const onQueryChange = () => void updateSearchResults({ append: false })
  const onPageChange = () => {
    if (pageNumber.value === 0) return
    void updateSearchResults({ append: true })
  }

  watch([sortField, searchMode], onQueryChange)
  watch(pageNumber, onPageChange)
  watchDebounced(searchQuery, onQueryChange, {
    debounce: SEARCH_DEBOUNCE_TIME,
    immediate: true
  })

  const sortOptions = computed(() => {
    return getSortableFields()
  })

  return {
    isLoading,
    pageNumber,
    pageSize,
    sortField,
    searchMode,
    searchQuery,
    suggestions,
    searchResults,
    sortOptions,
    clearCache: clearSearchCache
  }
}
