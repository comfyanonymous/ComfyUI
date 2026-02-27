import { useAsyncState, whenever } from '@vueuse/core'
import { difference } from 'es-toolkit'
import { defineStore } from 'pinia'
import { computed, reactive, ref, shallowReactive } from 'vue'
import {
  mapInputFileToAssetItem,
  mapTaskOutputToAssetItem
} from '@/platform/assets/composables/media/assetMappers'
import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import { assetService } from '@/platform/assets/services/assetService'
import type { PaginationOptions } from '@/platform/assets/services/assetService'
import { isCloud } from '@/platform/distribution/types'
import type { JobListItem } from '@/platform/remote/comfyui/jobs/jobTypes'
import { api } from '@/scripts/api'

import { TaskItemImpl } from './queueStore'
import { useAssetDownloadStore } from './assetDownloadStore'
import { useModelToNodeStore } from './modelToNodeStore'

const INPUT_LIMIT = 100

/**
 * Fetch input files from the internal API (OSS version)
 */
async function fetchInputFilesFromAPI(): Promise<AssetItem[]> {
  const response = await fetch(api.internalURL('/files/input'), {
    headers: {
      'Comfy-User': api.user
    }
  })

  if (!response.ok) {
    throw new Error('Failed to fetch input files')
  }

  const filenames: string[] = await response.json()
  return filenames.map((name, index) =>
    mapInputFileToAssetItem(name, index, 'input')
  )
}

/**
 * Fetch input files from cloud service
 */
async function fetchInputFilesFromCloud(): Promise<AssetItem[]> {
  return await assetService.getAssetsByTag('input', false, {
    limit: INPUT_LIMIT
  })
}

/**
 * Convert history job items to asset items
 */
function mapHistoryToAssets(historyItems: JobListItem[]): AssetItem[] {
  const assetItems: AssetItem[] = []

  for (const job of historyItems) {
    // Only process completed jobs with preview output
    if (job.status !== 'completed' || !job.preview_output) {
      continue
    }

    const task = new TaskItemImpl(job)

    if (!task.previewOutput) {
      continue
    }

    const assetItem = mapTaskOutputToAssetItem(task, task.previewOutput)

    assetItem.user_metadata = {
      ...assetItem.user_metadata,
      outputCount: job.outputs_count,
      allOutputs: task.previewableOutputs
    }

    assetItems.push(assetItem)
  }

  return assetItems.sort(
    (a, b) =>
      new Date(b.created_at ?? 0).getTime() -
      new Date(a.created_at ?? 0).getTime()
  )
}

const BATCH_SIZE = 200
const MAX_HISTORY_ITEMS = 1000 // Maximum items to keep in memory

export const useAssetsStore = defineStore('assets', () => {
  const assetDownloadStore = useAssetDownloadStore()
  const modelToNodeStore = useModelToNodeStore()

  // Track assets currently being deleted (for loading overlay)
  const deletingAssetIds = shallowReactive(new Set<string>())

  const setAssetDeleting = (assetId: string, isDeleting: boolean) => {
    if (isDeleting) {
      deletingAssetIds.add(assetId)
    } else {
      deletingAssetIds.delete(assetId)
    }
  }

  const isAssetDeleting = (assetId: string): boolean => {
    return deletingAssetIds.has(assetId)
  }

  // Pagination state
  const historyOffset = ref(0)
  const hasMoreHistory = ref(true)
  const isLoadingMore = ref(false)

  const allHistoryItems = ref<AssetItem[]>([])

  const loadedIds = shallowReactive(new Set<string>())

  const fetchInputFiles = isCloud
    ? fetchInputFilesFromCloud
    : fetchInputFilesFromAPI

  const {
    state: inputAssets,
    isLoading: inputLoading,
    error: inputError,
    execute: updateInputs
  } = useAsyncState(fetchInputFiles, [], {
    immediate: false,
    resetOnExecute: false,
    onError: (err) => {
      console.error('Error fetching input assets:', err)
    }
  })

  /**
   * Fetch history assets with pagination support
   * @param loadMore - true for pagination (append), false for initial load (replace)
   */
  const fetchHistoryAssets = async (loadMore = false): Promise<AssetItem[]> => {
    // Reset state for initial load
    if (!loadMore) {
      historyOffset.value = 0
      hasMoreHistory.value = true
      allHistoryItems.value = []
      loadedIds.clear()
    }

    // Fetch from server with offset
    const history = await api.getHistory(BATCH_SIZE, {
      offset: historyOffset.value
    })

    // Convert JobListItems to AssetItems
    const newAssets = mapHistoryToAssets(history)

    if (loadMore) {
      // Filter out duplicates and insert in sorted order
      for (const asset of newAssets) {
        if (loadedIds.has(asset.id)) {
          continue // Skip duplicates
        }
        loadedIds.add(asset.id)

        // Find insertion index to maintain sorted order (newest first)
        const assetTime = new Date(asset.created_at ?? 0).getTime()
        const insertIndex = allHistoryItems.value.findIndex(
          (item) => new Date(item.created_at ?? 0).getTime() < assetTime
        )

        if (insertIndex === -1) {
          // Asset is oldest, append to end
          allHistoryItems.value.push(asset)
        } else {
          // Insert at the correct position
          allHistoryItems.value.splice(insertIndex, 0, asset)
        }
      }
    } else {
      // Initial load: replace all
      allHistoryItems.value = newAssets
      newAssets.forEach((asset) => loadedIds.add(asset.id))
    }

    // Update pagination state
    historyOffset.value += BATCH_SIZE
    hasMoreHistory.value = history.length === BATCH_SIZE

    if (allHistoryItems.value.length > MAX_HISTORY_ITEMS) {
      const removed = allHistoryItems.value.slice(MAX_HISTORY_ITEMS)
      allHistoryItems.value = allHistoryItems.value.slice(0, MAX_HISTORY_ITEMS)

      // Clean up Set
      removed.forEach((item) => loadedIds.delete(item.id))
    }

    return allHistoryItems.value
  }

  const historyAssets = ref<AssetItem[]>([])
  const historyLoading = ref(false)
  const historyError = ref<unknown>(null)

  /**
   * Initial load of history assets
   */
  const updateHistory = async () => {
    historyLoading.value = true
    historyError.value = null
    try {
      await fetchHistoryAssets(false)
      historyAssets.value = allHistoryItems.value
    } catch (err) {
      console.error('Error fetching history assets:', err)
      historyError.value = err
      // Keep existing data when error occurs
      if (!historyAssets.value.length) {
        historyAssets.value = []
      }
    } finally {
      historyLoading.value = false
    }
  }

  /**
   * Load more history items (infinite scroll)
   */
  const loadMoreHistory = async () => {
    // Guard: prevent concurrent loads and check if more items available
    if (!hasMoreHistory.value || isLoadingMore.value) return

    isLoadingMore.value = true
    historyError.value = null

    try {
      await fetchHistoryAssets(true)
      historyAssets.value = allHistoryItems.value
    } catch (err) {
      console.error('Error loading more history:', err)
      historyError.value = err
      // Keep existing data when error occurs (consistent with updateHistory)
      if (!historyAssets.value.length) {
        historyAssets.value = []
      }
    } finally {
      isLoadingMore.value = false
    }
  }

  /**
   * Map of asset hash filename to asset item for O(1) lookup
   * Cloud assets use asset_hash for the hash-based filename
   */
  const inputAssetsByFilename = computed(() => {
    const map = new Map<string, AssetItem>()
    for (const asset of inputAssets.value) {
      if (asset.asset_hash) {
        map.set(asset.asset_hash, asset)
      }
    }
    return map
  })

  /**
   * @param filename Hash-based filename (e.g., "72e786ff...efb7.png")
   * @returns Human-readable asset name or original filename if not found
   */
  function getInputName(filename: string): string {
    return inputAssetsByFilename.value.get(filename)?.name ?? filename
  }

  const MODEL_BATCH_SIZE = 500

  interface ModelPaginationState {
    assets: Map<string, AssetItem>
    offset: number
    hasMore: boolean
    isLoading: boolean
    error?: Error
  }

  /**
   * Model assets cached by category (e.g., 'checkpoints', 'loras')
   * Multiple node types sharing the same category share the same cache entry.
   * Public API accepts nodeType for backwards compatibility but translates
   * to category internally using modelToNodeStore.getCategoryForNodeType().
   * Cloud-only feature - empty Maps in desktop builds
   */
  const getModelState = () => {
    if (isCloud) {
      const modelStateByCategory = ref(new Map<string, ModelPaginationState>())

      const assetsArrayCache = new Map<
        string,
        { source: Map<string, AssetItem>; array: AssetItem[] }
      >()

      const pendingRequestByCategory = new Map<string, ModelPaginationState>()

      function createState(
        existingAssets?: Map<string, AssetItem>
      ): ModelPaginationState {
        const assets = new Map(existingAssets)
        return reactive({
          assets,
          offset: 0,
          hasMore: true,
          isLoading: true
        })
      }

      function isStale(category: string, state: ModelPaginationState): boolean {
        const committed = modelStateByCategory.value.get(category)
        const pending = pendingRequestByCategory.get(category)
        return committed !== state && pending !== state
      }

      const EMPTY_ASSETS: AssetItem[] = []

      /**
       * Resolve a key to a category. Handles both nodeType and tag:xxx formats.
       * @param key Either a nodeType (e.g., 'CheckpointLoaderSimple') or tag key (e.g., 'tag:models')
       * @returns The category or undefined if not resolvable
       */
      function resolveCategory(key: string): string | undefined {
        if (key.startsWith('tag:')) {
          return key
        }
        return modelToNodeStore.getCategoryForNodeType(key)
      }

      /**
       * Get assets by nodeType or tag key.
       * Translates nodeType to category internally for cache lookup.
       * @param key Either a nodeType (e.g., 'CheckpointLoaderSimple') or tag key (e.g., 'tag:models')
       */
      function getAssets(key: string): AssetItem[] {
        const category = resolveCategory(key)
        if (!category) return EMPTY_ASSETS

        const state = modelStateByCategory.value.get(category)
        const assetsMap = state?.assets
        if (!assetsMap) return EMPTY_ASSETS

        const cached = assetsArrayCache.get(category)
        if (cached && cached.source === assetsMap) {
          return cached.array
        }

        const array = Array.from(assetsMap.values())
        assetsArrayCache.set(category, { source: assetsMap, array })
        return array
      }

      function isLoading(key: string): boolean {
        const category = resolveCategory(key)
        if (!category) return false
        return modelStateByCategory.value.get(category)?.isLoading ?? false
      }

      function getError(key: string): Error | undefined {
        const category = resolveCategory(key)
        if (!category) return undefined
        return modelStateByCategory.value.get(category)?.error
      }

      function hasMore(key: string): boolean {
        const category = resolveCategory(key)
        if (!category) return false
        return modelStateByCategory.value.get(category)?.hasMore ?? false
      }

      function hasAssetKey(key: string): boolean {
        const category = resolveCategory(key)
        if (!category) return false
        return modelStateByCategory.value.has(category)
      }

      /**
       * Check if a category exists in the cache.
       * Checks both direct category keys and tag-prefixed keys.
       * @param category The category to check (e.g., 'checkpoints', 'loras')
       */
      function hasCategory(category: string): boolean {
        return (
          modelStateByCategory.value.has(category) ||
          modelStateByCategory.value.has(`tag:${category}`)
        )
      }

      /**
       * Internal helper to fetch and cache assets for a category.
       * Loads first batch immediately, then progressively loads remaining batches.
       * Keeps existing data visible until new data is successfully fetched.
       *
       * Concurrent calls for the same category are short-circuited: if a request
       * is already in progress (tracked via pendingRequestByCategory), subsequent
       * calls return immediately to avoid redundant work.
       */
      async function updateModelsForCategory(
        category: string,
        fetcher: (options: PaginationOptions) => Promise<AssetItem[]>
      ): Promise<void> {
        // Short-circuit if a request for this category is already in progress
        if (pendingRequestByCategory.has(category)) {
          return
        }

        const existingState = modelStateByCategory.value.get(category)
        const state = createState(existingState?.assets)

        const seenIds = new Set<string>()

        const hasExistingData = modelStateByCategory.value.has(category)
        if (hasExistingData) {
          pendingRequestByCategory.set(category, state)
        } else {
          // Also track in pending map for initial loads to prevent concurrent calls
          pendingRequestByCategory.set(category, state)
          modelStateByCategory.value.set(category, state)
        }

        async function loadBatches(): Promise<void> {
          while (state.hasMore) {
            try {
              const newAssets = await fetcher({
                limit: MODEL_BATCH_SIZE,
                offset: state.offset
              })

              if (isStale(category, state)) return

              const isFirstBatch = state.offset === 0
              if (isFirstBatch) {
                assetsArrayCache.delete(category)
                if (hasExistingData) {
                  pendingRequestByCategory.delete(category)
                  modelStateByCategory.value.set(category, state)
                }
              }

              // Merge new assets into existing map and track seen IDs
              for (const asset of newAssets) {
                seenIds.add(asset.id)
                state.assets.set(asset.id, asset)
              }
              state.assets = new Map(state.assets)

              state.offset += newAssets.length
              state.hasMore = newAssets.length === MODEL_BATCH_SIZE

              if (isFirstBatch) {
                state.isLoading = false
              }

              if (state.hasMore) {
                await new Promise((resolve) => setTimeout(resolve, 50))
              }
            } catch (err) {
              if (isStale(category, state)) return
              console.error(`Error loading batch for ${category}:`, err)

              state.error = err instanceof Error ? err : new Error(String(err))
              state.hasMore = false
              state.isLoading = false
              pendingRequestByCategory.delete(category)

              return
            }
          }

          const staleIds = [...state.assets.keys()].filter(
            (id) => !seenIds.has(id)
          )
          for (const id of staleIds) {
            state.assets.delete(id)
          }
          assetsArrayCache.delete(category)
          pendingRequestByCategory.delete(category)
        }

        await loadBatches()
      }

      /**
       * Fetch and cache model assets for a specific node type.
       * Translates nodeType to category internally - multiple node types
       * sharing the same category will share the same cache entry.
       * @param nodeType The node type to fetch assets for (e.g., 'CheckpointLoaderSimple')
       */
      async function updateModelsForNodeType(nodeType: string): Promise<void> {
        const category = modelToNodeStore.getCategoryForNodeType(nodeType)
        if (!category) return

        // Use category as cache key but fetch using nodeType for API compatibility
        await updateModelsForCategory(category, (opts) =>
          assetService.getAssetsForNodeType(nodeType, opts)
        )
      }

      /**
       * Fetch and cache model assets for a specific tag
       * @param tag The tag to fetch assets for (e.g., 'models')
       */
      async function updateModelsForTag(tag: string): Promise<void> {
        const category = `tag:${tag}`
        await updateModelsForCategory(category, (opts) =>
          assetService.getAssetsByTag(tag, true, opts)
        )
      }

      /**
       * Invalidate the cache for a specific category.
       * Forces a refetch on next access.
       * @param category The category to invalidate (e.g., 'checkpoints', 'loras')
       */
      function invalidateCategory(category: string): void {
        modelStateByCategory.value.delete(category)
        assetsArrayCache.delete(category)
        pendingRequestByCategory.delete(category)
      }

      /**
       * Optimistically update an asset in the cache
       * @param assetId The asset ID to update
       * @param updates Partial asset data to merge
       * @param cacheKey Optional cache key to target (nodeType or 'tag:xxx')
       */
      function updateAssetInCache(
        assetId: string,
        updates: Partial<AssetItem>,
        cacheKey?: string
      ) {
        const category = cacheKey ? resolveCategory(cacheKey) : undefined
        if (cacheKey && !category) return

        const categoriesToCheck = category
          ? [category]
          : Array.from(modelStateByCategory.value.keys())

        for (const cat of categoriesToCheck) {
          const state = modelStateByCategory.value.get(cat)
          if (!state?.assets) continue

          const existingAsset = state.assets.get(assetId)
          if (existingAsset) {
            const updatedAsset = { ...existingAsset, ...updates }
            state.assets.set(assetId, updatedAsset)
            assetsArrayCache.delete(cat)
            if (cacheKey) return
          }
        }
      }

      /**
       * Update asset metadata with optimistic cache update
       * @param asset The asset to update
       * @param userMetadata The user_metadata to save
       * @param cacheKey Optional cache key to target for optimistic update
       */
      async function updateAssetMetadata(
        asset: AssetItem,
        userMetadata: Record<string, unknown>,
        cacheKey?: string
      ) {
        const originalMetadata = asset.user_metadata
        updateAssetInCache(asset.id, { user_metadata: userMetadata }, cacheKey)

        try {
          const updatedAsset = await assetService.updateAsset(asset.id, {
            user_metadata: userMetadata
          })
          updateAssetInCache(asset.id, updatedAsset, cacheKey)
        } catch (error) {
          console.error('Failed to update asset metadata:', error)
          updateAssetInCache(
            asset.id,
            { user_metadata: originalMetadata },
            cacheKey
          )
        }
      }

      /**
       * Update asset tags using add/remove endpoints
       * @param asset The asset to update (used to read current tags)
       * @param newTags The desired tags array
       * @param cacheKey Optional cache key to target for optimistic update
       */
      async function updateAssetTags(
        asset: AssetItem,
        newTags: string[],
        cacheKey?: string
      ) {
        const originalTags = asset.tags
        const tagsToAdd = difference(newTags, originalTags)
        const tagsToRemove = difference(originalTags, newTags)

        if (tagsToAdd.length === 0 && tagsToRemove.length === 0) return

        updateAssetInCache(asset.id, { tags: newTags }, cacheKey)

        try {
          const removeResult =
            tagsToRemove.length > 0
              ? await assetService.removeAssetTags(asset.id, tagsToRemove)
              : undefined

          const addResult =
            tagsToAdd.length > 0
              ? await assetService.addAssetTags(asset.id, tagsToAdd)
              : undefined

          const finalTags = (addResult ?? removeResult)?.total_tags
          if (finalTags) {
            updateAssetInCache(asset.id, { tags: finalTags }, cacheKey)
          }
        } catch (error) {
          console.error('Failed to update asset tags:', error)
          updateAssetInCache(asset.id, { tags: originalTags }, cacheKey)
        }
      }

      /**
       * Invalidate model caches for a given category (e.g., 'checkpoints', 'loras')
       * Clears the category cache and tag-based caches so next access triggers refetch
       * @param category The model category to invalidate (e.g., 'checkpoints')
       */
      function invalidateModelsForCategory(category: string): void {
        invalidateCategory(category)
        invalidateCategory(`tag:${category}`)
        invalidateCategory('tag:models')
      }

      return {
        getAssets,
        isLoading,
        getError,
        hasMore,
        hasAssetKey,
        hasCategory,
        updateModelsForNodeType,
        updateModelsForTag,
        invalidateCategory,
        updateAssetMetadata,
        updateAssetTags,
        invalidateModelsForCategory
      }
    }

    const emptyAssets: AssetItem[] = []
    return {
      getAssets: () => emptyAssets,
      isLoading: () => false,
      getError: () => undefined,
      hasMore: () => false,
      hasAssetKey: () => false,
      hasCategory: () => false,
      updateModelsForNodeType: async () => {},
      invalidateCategory: () => {},
      updateModelsForTag: async () => {},
      updateAssetMetadata: async () => {},
      updateAssetTags: async () => {},
      invalidateModelsForCategory: () => {}
    }
  }

  const {
    getAssets,
    isLoading: isModelLoading,
    getError,
    hasMore,
    hasAssetKey,
    hasCategory,
    updateModelsForNodeType,
    updateModelsForTag,
    invalidateCategory,
    updateAssetMetadata,
    updateAssetTags,
    invalidateModelsForCategory
  } = getModelState()

  // Watch for completed downloads and refresh model caches
  whenever(
    () => assetDownloadStore.lastCompletedDownload,
    async (latestDownload) => {
      const { modelType } = latestDownload

      const providers = modelToNodeStore
        .getAllNodeProviders(modelType)
        .filter((provider) => provider.nodeDef?.name)

      const nodeTypeUpdates = providers.map((provider) =>
        updateModelsForNodeType(provider.nodeDef.name).then(
          () => provider.nodeDef.name
        )
      )

      // Also update by tag in case modal was opened with assetType
      const tagUpdates = [
        updateModelsForTag(modelType),
        updateModelsForTag('models')
      ]

      const results = await Promise.allSettled([
        ...nodeTypeUpdates,
        ...tagUpdates
      ])

      for (const result of results) {
        if (result.status === 'rejected') {
          console.error(
            `Failed to refresh model cache for provider: ${result.reason}`
          )
        }
      }
    }
  )

  return {
    // States
    inputAssets,
    historyAssets,
    inputLoading,
    historyLoading,
    inputError,
    historyError,
    hasMoreHistory,
    isLoadingMore,

    // Deletion tracking
    deletingAssetIds,
    setAssetDeleting,
    isAssetDeleting,

    // Actions
    updateInputs,
    updateHistory,
    loadMoreHistory,

    // Input mapping helpers
    inputAssetsByFilename,
    getInputName,

    // Model assets - accessors
    getAssets,
    isModelLoading,
    getError,
    hasMore,
    hasAssetKey,
    hasCategory,

    // Model assets - actions
    updateModelsForNodeType,
    updateModelsForTag,
    invalidateCategory,
    updateAssetMetadata,
    updateAssetTags,
    invalidateModelsForCategory
  }
})
