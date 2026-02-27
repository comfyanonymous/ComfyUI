import { computed, ref, shallowRef, watch } from 'vue'
import type { Ref } from 'vue'

interface LazyPaginationOptions {
  itemsPerPage?: number
  initialPage?: number
}

export function useLazyPagination<T>(
  items: Ref<T[]> | T[],
  options: LazyPaginationOptions = {}
) {
  const { itemsPerPage = 12, initialPage = 1 } = options

  const currentPage = ref(initialPage)
  const isLoading = ref(false)
  const loadedPages = shallowRef(new Set<number>([]))

  // Get reactive items array
  const itemsArray = computed(() => {
    const itemData = 'value' in items ? items.value : items
    return Array.isArray(itemData) ? itemData : []
  })

  // Simulate pagination by slicing the items
  const paginatedItems = computed(() => {
    const itemData = itemsArray.value
    if (itemData.length === 0) {
      return []
    }

    const loadedPageNumbers = Array.from(loadedPages.value).sort(
      (a, b) => a - b
    )
    const maxLoadedPage = Math.max(...loadedPageNumbers, 0)
    const endIndex = maxLoadedPage * itemsPerPage
    return itemData.slice(0, endIndex)
  })

  const hasMoreItems = computed(() => {
    const itemData = itemsArray.value
    if (itemData.length === 0) {
      return false
    }

    const loadedPagesArray = Array.from(loadedPages.value)
    const maxLoadedPage = Math.max(...loadedPagesArray, 0)
    return maxLoadedPage * itemsPerPage < itemData.length
  })

  const totalPages = computed(() => {
    const itemData = itemsArray.value
    if (itemData.length === 0) {
      return 0
    }
    return Math.ceil(itemData.length / itemsPerPage)
  })

  const loadNextPage = async () => {
    if (isLoading.value || !hasMoreItems.value) return

    isLoading.value = true
    const loadedPagesArray = Array.from(loadedPages.value)
    const nextPage = Math.max(...loadedPagesArray, 0) + 1

    // Simulate network delay
    // await new Promise((resolve) => setTimeout(resolve, 5000))

    const newLoadedPages = new Set(loadedPages.value)
    newLoadedPages.add(nextPage)
    loadedPages.value = newLoadedPages
    currentPage.value = nextPage
    isLoading.value = false
  }

  // Initialize with first page
  watch(
    () => itemsArray.value.length,
    (length) => {
      if (length > 0 && loadedPages.value.size === 0) {
        loadedPages.value = new Set([1])
      }
    },
    { immediate: true }
  )

  const reset = () => {
    currentPage.value = initialPage
    loadedPages.value = new Set([])
    isLoading.value = false

    // Immediately load first page if we have items
    const itemData = itemsArray.value
    if (itemData.length > 0) {
      loadedPages.value = new Set([1])
    }
  }

  return {
    paginatedItems,
    isLoading,
    hasMoreItems,
    currentPage,
    totalPages,
    loadNextPage,
    reset
  }
}
