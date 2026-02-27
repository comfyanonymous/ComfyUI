import { refDebounced } from '@vueuse/core'
import { sortBy as sortByUtil } from 'es-toolkit'
import Fuse from 'fuse.js'
import { computed, ref } from 'vue'
import type { Ref } from 'vue'

import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import { getMediaTypeFromFilename } from '@/utils/formatUtil'

type SortOption = 'newest' | 'oldest' | 'longest' | 'fastest'

/**
 * Get timestamp from asset (either create_time or created_at)
 */
const getAssetTime = (asset: AssetItem): number => {
  return (
    (asset.user_metadata?.create_time as number) ??
    (asset.created_at ? new Date(asset.created_at).getTime() : 0)
  )
}

/**
 * Get execution time from asset user_metadata
 */
const getAssetExecutionTime = (asset: AssetItem): number => {
  return (asset.user_metadata?.executionTimeInSeconds as number) ?? 0
}

/**
 * Media Asset Filtering composable
 * Manages search, filter, and sort for media assets
 */
export function useMediaAssetFiltering(assets: Ref<AssetItem[]>) {
  const searchQuery = ref('')
  const debouncedSearchQuery = refDebounced(searchQuery, 50)
  const sortBy = ref<SortOption>('newest')
  const mediaTypeFilters = ref<string[]>([])

  const fuseOptions = {
    keys: ['name'],
    threshold: 0.4,
    includeScore: true
  }

  const fuse = computed(() => new Fuse(assets.value, fuseOptions))

  const searchFiltered = computed(() => {
    if (!debouncedSearchQuery.value.trim()) {
      return assets.value
    }

    const results = fuse.value.search(debouncedSearchQuery.value)
    return results.map((result) => result.item)
  })

  const typeFiltered = computed(() => {
    // Apply media type filter
    if (mediaTypeFilters.value.length === 0) {
      return searchFiltered.value
    }

    return searchFiltered.value.filter((asset) => {
      const mediaType = getMediaTypeFromFilename(asset.name)
      // Convert '3D' to '3d' for comparison
      const normalizedType = mediaType.toLowerCase()
      return mediaTypeFilters.value.includes(normalizedType)
    })
  })

  const filteredAssets = computed(() => {
    // Sort by create_time (output assets) or created_at (input assets)
    switch (sortBy.value) {
      case 'oldest':
        // Ascending order (oldest first)
        return sortByUtil(typeFiltered.value, [getAssetTime])
      case 'longest':
        // Descending order (longest execution time first)
        return sortByUtil(typeFiltered.value, [
          (asset) => -getAssetExecutionTime(asset)
        ])
      case 'fastest':
        // Ascending order (fastest execution time first)
        return sortByUtil(typeFiltered.value, [getAssetExecutionTime])
      case 'newest':
      default:
        // Descending order (newest first) - negate for descending
        return sortByUtil(typeFiltered.value, [(asset) => -getAssetTime(asset)])
    }
  })

  return {
    searchQuery,
    sortBy,
    mediaTypeFilters,
    filteredAssets
  }
}
