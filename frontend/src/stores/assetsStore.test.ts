import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick, watch } from 'vue'

import { useAssetsStore } from '@/stores/assetsStore'
import { api } from '@/scripts/api'
import type { JobListItem } from '@/platform/remote/comfyui/jobs/jobTypes'
import { assetService } from '@/platform/assets/services/assetService'

// Mock the api module
vi.mock('@/scripts/api', () => ({
  api: {
    getHistory: vi.fn(),
    internalURL: vi.fn((path) => `http://localhost:3000${path}`),
    apiURL: vi.fn((path) => `http://localhost:3000/api${path}`),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    user: 'test-user'
  }
}))

// Mock the asset service
vi.mock('@/platform/assets/services/assetService', () => ({
  assetService: {
    getAssetsByTag: vi.fn(),
    getAssetsForNodeType: vi.fn()
  }
}))

// Mock distribution type - hoisted so it can be changed per test
const mockIsCloud = vi.hoisted(() => ({ value: false }))
vi.mock('@/platform/distribution/types', () => ({
  get isCloud() {
    return mockIsCloud.value
  }
}))

// Mock modelToNodeStore with proper node providers and category lookups
vi.mock('@/stores/modelToNodeStore', () => ({
  useModelToNodeStore: () => ({
    getAllNodeProviders: vi.fn((category: string) => {
      const providers: Record<
        string,
        Array<{ nodeDef: { name: string }; key: string }>
      > = {
        checkpoints: [
          { nodeDef: { name: 'CheckpointLoaderSimple' }, key: 'ckpt_name' },
          { nodeDef: { name: 'ImageOnlyCheckpointLoader' }, key: 'ckpt_name' }
        ],
        loras: [
          { nodeDef: { name: 'LoraLoader' }, key: 'lora_name' },
          { nodeDef: { name: 'LoraLoaderModelOnly' }, key: 'lora_name' }
        ],
        vae: [{ nodeDef: { name: 'VAELoader' }, key: 'vae_name' }]
      }
      return providers[category] ?? []
    }),
    getCategoryForNodeType: vi.fn((nodeType: string) => {
      const nodeToCategory: Record<string, string> = {
        CheckpointLoaderSimple: 'checkpoints',
        ImageOnlyCheckpointLoader: 'checkpoints',
        LoraLoader: 'loras',
        LoraLoaderModelOnly: 'loras',
        VAELoader: 'vae'
      }
      return nodeToCategory[nodeType]
    }),
    getNodeProvider: vi.fn(),
    registerDefaults: vi.fn()
  })
}))

// Mock TaskItemImpl
vi.mock('@/stores/queueStore', () => ({
  TaskItemImpl: class {
    public flatOutputs: Array<{
      supportsPreview: boolean
      filename: string
      subfolder: string
      type: string
      url: string
    }>
    public previewOutput:
      | {
          supportsPreview: boolean
          filename: string
          subfolder: string
          type: string
          url: string
        }
      | undefined
    public jobId: string

    constructor(public job: JobListItem) {
      this.jobId = job.id
      this.flatOutputs = [
        {
          supportsPreview: true,
          filename: 'test.png',
          subfolder: '',
          type: 'output',
          url: 'http://test.com/test.png'
        }
      ]
      this.previewOutput = this.flatOutputs[0]
    }

    get previewableOutputs() {
      return this.flatOutputs.filter((o) => o.supportsPreview)
    }
  }
}))

// Mock asset mappers - add unique timestamps
vi.mock('@/platform/assets/composables/media/assetMappers', () => ({
  mapInputFileToAssetItem: vi.fn((name, index, type) => ({
    id: `${type}-${index}`,
    name,
    size: 0,
    created_at: new Date(Date.now() - index * 1000).toISOString(),
    tags: [type],
    preview_url: `http://test.com/${name}`
  })),
  mapTaskOutputToAssetItem: vi.fn((task, output) => {
    const index = parseInt(task.jobId.split('_')[1]) || 0
    return {
      id: task.jobId,
      name: output.filename,
      size: 0,
      created_at: new Date(Date.now() - index * 1000).toISOString(),
      tags: ['output'],
      preview_url: output.url,
      user_metadata: {}
    }
  })
}))

describe('assetsStore - Refactored (Option A)', () => {
  let store: ReturnType<typeof useAssetsStore>

  // Helper function to create mock job items
  const createMockJobItem = (index: number): JobListItem => ({
    id: `prompt_${index}`,
    status: 'completed',
    create_time: 1000 + index,
    update_time: 1000 + index,
    last_state_update: 1000 + index,
    priority: 1000 + index,
    preview_output: {
      filename: `output_${index}.png`,
      subfolder: '',
      type: 'output',
      nodeId: 'node_1',
      mediaType: 'images'
    }
  })

  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    store = useAssetsStore()
    vi.clearAllMocks()
  })

  describe('Initial Load', () => {
    it('should load initial history items', async () => {
      const mockHistory = Array.from({ length: 10 }, (_, i) =>
        createMockJobItem(i)
      )
      vi.mocked(api.getHistory).mockResolvedValue(mockHistory)

      await store.updateHistory()

      expect(api.getHistory).toHaveBeenCalledWith(200, { offset: 0 })
      expect(store.historyAssets).toHaveLength(10)
      expect(store.hasMoreHistory).toBe(false) // Less than BATCH_SIZE
      expect(store.historyLoading).toBe(false)
      expect(store.historyError).toBe(null)
    })

    it('should set hasMoreHistory to true when batch is full', async () => {
      const mockHistory = Array.from({ length: 200 }, (_, i) =>
        createMockJobItem(i)
      )
      vi.mocked(api.getHistory).mockResolvedValue(mockHistory)

      await store.updateHistory()

      expect(store.historyAssets).toHaveLength(200)
      expect(store.hasMoreHistory).toBe(true) // Exactly BATCH_SIZE
    })

    it('should handle errors during initial load', async () => {
      const error = new Error('Failed to fetch')
      vi.mocked(api.getHistory).mockRejectedValue(error)

      await store.updateHistory()

      expect(store.historyAssets).toHaveLength(0)
      expect(store.historyError).toBe(error)
      expect(store.historyLoading).toBe(false)
    })
  })

  describe('Pagination', () => {
    it('should accumulate items when loading more', async () => {
      // First batch - full BATCH_SIZE
      const firstBatch = Array.from({ length: 200 }, (_, i) =>
        createMockJobItem(i)
      )
      vi.mocked(api.getHistory).mockResolvedValueOnce(firstBatch)

      await store.updateHistory()
      expect(store.historyAssets).toHaveLength(200)
      expect(store.hasMoreHistory).toBe(true)

      // Second batch - different items
      const secondBatch = Array.from({ length: 200 }, (_, i) =>
        createMockJobItem(200 + i)
      )
      vi.mocked(api.getHistory).mockResolvedValueOnce(secondBatch)

      await store.loadMoreHistory()

      expect(api.getHistory).toHaveBeenCalledWith(200, { offset: 200 })
      expect(store.historyAssets).toHaveLength(400) // Accumulated
      expect(store.hasMoreHistory).toBe(true)
    })

    it('should prevent duplicate items during pagination', async () => {
      // First batch - full BATCH_SIZE
      const firstBatch = Array.from({ length: 200 }, (_, i) =>
        createMockJobItem(i)
      )
      vi.mocked(api.getHistory).mockResolvedValueOnce(firstBatch)

      await store.updateHistory()
      expect(store.historyAssets).toHaveLength(200)

      // Second batch with some duplicates
      const secondBatch = [
        createMockJobItem(2), // Duplicate
        createMockJobItem(5), // Duplicate
        ...Array.from({ length: 198 }, (_, i) => createMockJobItem(200 + i)) // New
      ]
      vi.mocked(api.getHistory).mockResolvedValueOnce(secondBatch)

      await store.loadMoreHistory()

      // Should only add new items (198 new, 2 duplicates filtered)
      expect(store.historyAssets).toHaveLength(398)

      // Verify no duplicates
      const assetIds = store.historyAssets.map((a) => a.id)
      const uniqueAssetIds = new Set(assetIds)
      expect(uniqueAssetIds.size).toBe(store.historyAssets.length)
    })

    it('should stop loading when no more items', async () => {
      // First batch - less than BATCH_SIZE
      const firstBatch = Array.from({ length: 50 }, (_, i) =>
        createMockJobItem(i)
      )
      vi.mocked(api.getHistory).mockResolvedValueOnce(firstBatch)

      await store.updateHistory()
      expect(store.hasMoreHistory).toBe(false)

      // Try to load more - should return early
      await store.loadMoreHistory()

      // Should only have been called once (initial load)
      expect(api.getHistory).toHaveBeenCalledTimes(1)
    })

    it('should handle race conditions with concurrent loads', async () => {
      // Setup initial state with full batch
      const initialBatch = Array.from({ length: 200 }, (_, i) =>
        createMockJobItem(i)
      )
      vi.mocked(api.getHistory).mockResolvedValueOnce(initialBatch)
      await store.updateHistory()
      expect(store.hasMoreHistory).toBe(true)

      // Clear mock to count only loadMore calls
      vi.mocked(api.getHistory).mockClear()

      // Setup slow API response
      let resolveLoadMore: (value: JobListItem[]) => void
      const loadMorePromise = new Promise<JobListItem[]>((resolve) => {
        resolveLoadMore = resolve
      })
      vi.mocked(api.getHistory).mockReturnValueOnce(loadMorePromise)

      // Start first loadMore
      const firstLoad = store.loadMoreHistory()

      // Try concurrent load - should be ignored
      const secondLoad = store.loadMoreHistory()

      // Resolve
      const secondBatch = Array.from({ length: 200 }, (_, i) =>
        createMockJobItem(200 + i)
      )
      resolveLoadMore!(secondBatch)

      await Promise.all([firstLoad, secondLoad])

      // Only one API call
      expect(api.getHistory).toHaveBeenCalledTimes(1)
    })

    it('should respect MAX_HISTORY_ITEMS limit', async () => {
      const BATCH_COUNT = 6 // 6 × 200 = 1200 items

      // Initial load
      const firstBatch = Array.from({ length: 200 }, (_, i) =>
        createMockJobItem(i)
      )
      vi.mocked(api.getHistory).mockResolvedValueOnce(firstBatch)
      await store.updateHistory()

      // Load additional batches
      for (let batch = 1; batch < BATCH_COUNT; batch++) {
        const items = Array.from({ length: 200 }, (_, i) =>
          createMockJobItem(batch * 200 + i)
        )
        vi.mocked(api.getHistory).mockResolvedValueOnce(items)
        await store.loadMoreHistory()
      }

      // Should be capped at MAX_HISTORY_ITEMS (1000)
      expect(store.historyAssets).toHaveLength(1000)
    })
  })

  describe('Sorting', () => {
    it('should maintain date sorting after pagination', async () => {
      // First batch
      const firstBatch = Array.from({ length: 200 }, (_, i) =>
        createMockJobItem(i)
      )
      vi.mocked(api.getHistory).mockResolvedValueOnce(firstBatch)

      await store.updateHistory()

      // Second batch
      const secondBatch = Array.from({ length: 200 }, (_, i) =>
        createMockJobItem(200 + i)
      )
      vi.mocked(api.getHistory).mockResolvedValueOnce(secondBatch)

      await store.loadMoreHistory()

      // Verify sorting (newest first - lower index = newer)
      for (let i = 1; i < store.historyAssets.length; i++) {
        const prevDate = new Date(store.historyAssets[i - 1].created_at ?? 0)
        const currDate = new Date(store.historyAssets[i].created_at ?? 0)
        expect(prevDate.getTime()).toBeGreaterThanOrEqual(currDate.getTime())
      }
    })
  })

  describe('Error Handling', () => {
    it('should preserve existing data when loadMore fails', async () => {
      // First successful load - full batch
      const firstBatch = Array.from({ length: 200 }, (_, i) =>
        createMockJobItem(i)
      )
      vi.mocked(api.getHistory).mockResolvedValueOnce(firstBatch)

      await store.updateHistory()
      expect(store.historyAssets).toHaveLength(200)

      // Second load fails
      const error = new Error('Network error')
      vi.mocked(api.getHistory).mockRejectedValueOnce(error)

      await store.loadMoreHistory()

      // Should keep existing data
      expect(store.historyAssets).toHaveLength(200)
      expect(store.historyError).toBe(error)
      expect(store.isLoadingMore).toBe(false)
    })

    it('should clear error state on successful retry', async () => {
      // First load succeeds
      const firstBatch = Array.from({ length: 200 }, (_, i) =>
        createMockJobItem(i)
      )
      vi.mocked(api.getHistory).mockResolvedValueOnce(firstBatch)

      await store.updateHistory()

      // Second load fails
      const error = new Error('Network error')
      vi.mocked(api.getHistory).mockRejectedValueOnce(error)

      await store.loadMoreHistory()
      expect(store.historyError).toBe(error)

      // Third load succeeds
      const thirdBatch = Array.from({ length: 200 }, (_, i) =>
        createMockJobItem(200 + i)
      )
      vi.mocked(api.getHistory).mockResolvedValueOnce(thirdBatch)

      await store.loadMoreHistory()

      // Error should be cleared
      expect(store.historyError).toBe(null)
      expect(store.historyAssets).toHaveLength(400)
    })

    it('should handle errors with proper loading state', async () => {
      const error = new Error('API error')
      vi.mocked(api.getHistory).mockRejectedValue(error)

      await store.updateHistory()

      expect(store.historyLoading).toBe(false)
      expect(store.historyError).toBe(error)
    })
  })

  describe('Memory Management', () => {
    it('should cleanup when exceeding MAX_HISTORY_ITEMS', async () => {
      // Load 1200 items (exceeds 1000 limit)
      const batches = 6

      for (let batch = 0; batch < batches; batch++) {
        const items = Array.from({ length: 200 }, (_, i) =>
          createMockJobItem(batch * 200 + i)
        )
        vi.mocked(api.getHistory).mockResolvedValueOnce(items)

        if (batch === 0) {
          await store.updateHistory()
        } else {
          await store.loadMoreHistory()
        }
      }

      // Should be limited to 1000
      expect(store.historyAssets).toHaveLength(1000)

      // All items should be unique (Set cleanup works)
      const assetIds = store.historyAssets.map((a) => a.id)
      const uniqueAssetIds = new Set(assetIds)
      expect(uniqueAssetIds.size).toBe(1000)
    })

    it('should maintain correct state after cleanup', async () => {
      // Load items beyond limit
      for (let batch = 0; batch < 6; batch++) {
        const items = Array.from({ length: 200 }, (_, i) =>
          createMockJobItem(batch * 200 + i)
        )
        vi.mocked(api.getHistory).mockResolvedValueOnce(items)

        if (batch === 0) {
          await store.updateHistory()
        } else {
          await store.loadMoreHistory()
        }
      }

      expect(store.historyAssets).toHaveLength(1000)

      // Should still maintain sorting
      for (let i = 1; i < store.historyAssets.length; i++) {
        const prevDate = new Date(store.historyAssets[i - 1].created_at ?? 0)
        const currDate = new Date(store.historyAssets[i].created_at ?? 0)
        expect(prevDate.getTime()).toBeGreaterThanOrEqual(currDate.getTime())
      }
    })
  })

  describe('jobDetailView Support', () => {
    it('should include outputCount and allOutputs in user_metadata', async () => {
      const mockHistory = Array.from({ length: 5 }, (_, i) =>
        createMockJobItem(i)
      )
      vi.mocked(api.getHistory).mockResolvedValue(mockHistory)

      await store.updateHistory()

      // Check first asset
      const asset = store.historyAssets[0]
      expect(asset.user_metadata).toBeDefined()
      expect(asset.user_metadata).toHaveProperty('outputCount')
      expect(asset.user_metadata).toHaveProperty('allOutputs')
      expect(Array.isArray(asset.user_metadata!.allOutputs)).toBe(true)
    })
  })
})

describe('assetsStore - Model Assets Cache (Cloud)', () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    mockIsCloud.value = true
    vi.clearAllMocks()
  })

  afterEach(() => {
    mockIsCloud.value = false
  })

  const createMockAsset = (id: string, tags: string[] = ['models']) => ({
    id,
    name: `asset-${id}`,
    size: 100,
    created_at: new Date().toISOString(),
    tags,
    preview_url: `http://test.com/${id}`
  })

  describe('getAssets cache invalidation', () => {
    it('should invalidate cache before mutating assets during batch loading', async () => {
      const store = useAssetsStore()
      const nodeType = 'CheckpointLoaderSimple'

      const firstBatch = Array.from({ length: 500 }, (_, i) =>
        createMockAsset(`asset-${i}`)
      )
      const secondBatch = Array.from({ length: 100 }, (_, i) =>
        createMockAsset(`asset-${500 + i}`)
      )

      let callCount = 0
      vi.mocked(assetService.getAssetsForNodeType).mockImplementation(
        async () => {
          callCount++
          return callCount === 1 ? firstBatch : secondBatch
        }
      )

      await store.updateModelsForNodeType(nodeType)

      // Wait for background batch loading to complete
      await vi.waitFor(() => {
        expect(
          vi.mocked(assetService.getAssetsForNodeType)
        ).toHaveBeenCalledTimes(2)
      })

      const assets = store.getAssets(nodeType)
      expect(assets).toHaveLength(600)
    })

    it('should not return stale cached array after background batch completes', async () => {
      const store = useAssetsStore()
      const nodeType = 'LoraLoader'

      // First batch must be exactly MODEL_BATCH_SIZE (500) to trigger hasMore
      const firstBatch = Array.from({ length: 500 }, (_, i) =>
        createMockAsset(`first-${i}`)
      )
      const secondBatch = [createMockAsset('new-asset')]

      let callCount = 0
      vi.mocked(assetService.getAssetsForNodeType).mockImplementation(
        async () => {
          callCount++
          return callCount === 1 ? firstBatch : secondBatch
        }
      )

      await store.updateModelsForNodeType(nodeType)

      // Wait for background batch loading to complete
      await vi.waitFor(() => {
        expect(
          vi.mocked(assetService.getAssetsForNodeType)
        ).toHaveBeenCalledTimes(2)
      })

      const assets = store.getAssets(nodeType)
      expect(assets).toHaveLength(501)
      expect(assets.map((a) => a.id)).toContain('new-asset')
    })

    it('should return cached array on subsequent getAssets calls', async () => {
      const store = useAssetsStore()
      const nodeType = 'CheckpointLoaderSimple'
      const assets = [createMockAsset('cache-test-1')]

      vi.mocked(assetService.getAssetsForNodeType).mockResolvedValue(assets)
      await store.updateModelsForNodeType(nodeType)

      const firstCall = store.getAssets(nodeType)
      const secondCall = store.getAssets(nodeType)

      expect(secondCall).toBe(firstCall)
      expect(firstCall).toHaveLength(1)
    })
  })

  describe('concurrent request handling', () => {
    it('should short-circuit concurrent calls to prevent duplicate work', async () => {
      const store = useAssetsStore()
      const nodeType = 'CheckpointLoaderSimple'
      const firstBatch = Array.from({ length: 5 }, (_, i) =>
        createMockAsset(`first-${i}`)
      )

      vi.mocked(assetService.getAssetsForNodeType).mockResolvedValue(firstBatch)

      // Start two concurrent requests for the same category
      const firstRequest = store.updateModelsForNodeType(nodeType)
      const secondRequest = store.updateModelsForNodeType(nodeType)
      await Promise.all([firstRequest, secondRequest])

      // Second request should be short-circuited, only one API call made
      expect(
        vi.mocked(assetService.getAssetsForNodeType)
      ).toHaveBeenCalledTimes(1)
      expect(store.getAssets(nodeType)).toHaveLength(5)
    })

    it('should allow new request after previous completes', async () => {
      const store = useAssetsStore()
      const nodeType = 'CheckpointLoaderSimple'
      const firstBatch = [createMockAsset('first-1')]
      const secondBatch = [
        createMockAsset('second-1'),
        createMockAsset('second-2')
      ]

      vi.mocked(assetService.getAssetsForNodeType).mockResolvedValueOnce(
        firstBatch
      )
      await store.updateModelsForNodeType(nodeType)
      expect(store.getAssets(nodeType)).toHaveLength(1)

      // After first completes, a new request should work
      vi.mocked(assetService.getAssetsForNodeType).mockResolvedValueOnce(
        secondBatch
      )
      store.invalidateCategory('checkpoints')
      await store.updateModelsForNodeType(nodeType)

      expect(store.getAssets(nodeType)).toHaveLength(2)
      expect(
        vi.mocked(assetService.getAssetsForNodeType)
      ).toHaveBeenCalledTimes(2)
    })
  })

  describe('shallowReactive state reactivity', () => {
    it('should trigger reactivity on isModelLoading change', async () => {
      const store = useAssetsStore()
      const nodeType = 'CheckpointLoaderSimple'

      const loadingStates: boolean[] = []
      watch(
        () => store.isModelLoading(nodeType),
        (val) => loadingStates.push(val),
        { immediate: true }
      )

      vi.mocked(assetService.getAssetsForNodeType).mockResolvedValue([])
      await store.updateModelsForNodeType(nodeType)
      await nextTick()

      expect(loadingStates).toContain(true)
      expect(loadingStates).toContain(false)
    })
  })

  describe('category-keyed cache', () => {
    it('should share cache between node types of the same category', async () => {
      const store = useAssetsStore()
      const assets = [createMockAsset('shared-1'), createMockAsset('shared-2')]

      vi.mocked(assetService.getAssetsForNodeType).mockResolvedValue(assets)

      await store.updateModelsForNodeType('CheckpointLoaderSimple')

      expect(store.getAssets('CheckpointLoaderSimple')).toHaveLength(2)
      expect(store.getAssets('ImageOnlyCheckpointLoader')).toHaveLength(2)
      expect(
        vi.mocked(assetService.getAssetsForNodeType)
      ).toHaveBeenCalledTimes(1)
    })

    it('should return empty array for unknown node types', () => {
      const store = useAssetsStore()
      expect(store.getAssets('UnknownNodeType')).toEqual([])
    })

    it('should not fetch for unknown node types', async () => {
      const store = useAssetsStore()
      await store.updateModelsForNodeType('UnknownNodeType')
      expect(
        vi.mocked(assetService.getAssetsForNodeType)
      ).not.toHaveBeenCalled()
    })
  })

  describe('invalidateCategory', () => {
    it('should clear cache for a category', async () => {
      const store = useAssetsStore()
      const assets = [createMockAsset('asset-1'), createMockAsset('asset-2')]

      vi.mocked(assetService.getAssetsForNodeType).mockResolvedValue(assets)
      await store.updateModelsForNodeType('CheckpointLoaderSimple')
      expect(store.getAssets('CheckpointLoaderSimple')).toHaveLength(2)

      store.invalidateCategory('checkpoints')

      expect(store.getAssets('CheckpointLoaderSimple')).toEqual([])
      expect(store.hasAssetKey('CheckpointLoaderSimple')).toBe(false)
    })

    it('should allow refetch after invalidation', async () => {
      const store = useAssetsStore()
      const initialAssets = [createMockAsset('initial-1')]
      const refreshedAssets = [
        createMockAsset('refreshed-1'),
        createMockAsset('refreshed-2')
      ]

      vi.mocked(assetService.getAssetsForNodeType).mockResolvedValueOnce(
        initialAssets
      )
      await store.updateModelsForNodeType('LoraLoader')
      expect(store.getAssets('LoraLoader')).toHaveLength(1)

      store.invalidateCategory('loras')

      vi.mocked(assetService.getAssetsForNodeType).mockResolvedValueOnce(
        refreshedAssets
      )
      await store.updateModelsForNodeType('LoraLoader')

      expect(store.getAssets('LoraLoader')).toHaveLength(2)
    })

    it('should invalidate tag-based caches', async () => {
      const store = useAssetsStore()
      const assets = [createMockAsset('tag-asset-1')]

      vi.mocked(assetService.getAssetsByTag).mockResolvedValue(assets)
      await store.updateModelsForTag('models')
      expect(store.getAssets('tag:models')).toHaveLength(1)

      store.invalidateCategory('tag:models')

      expect(store.getAssets('tag:models')).toEqual([])
    })
  })

  describe('hasCategory', () => {
    it('should return true for loaded categories', async () => {
      const store = useAssetsStore()
      const assets = [createMockAsset('asset-1')]

      vi.mocked(assetService.getAssetsForNodeType).mockResolvedValue(assets)
      await store.updateModelsForNodeType('CheckpointLoaderSimple')

      expect(store.hasCategory('checkpoints')).toBe(true)
    })

    it('should return true for tag-based category when tag: prefix is not used', async () => {
      const store = useAssetsStore()
      const assets = [createMockAsset('asset-1')]

      vi.mocked(assetService.getAssetsByTag).mockResolvedValue(assets)
      await store.updateModelsForTag('models')

      // hasCategory('models') checks for both 'models' and 'tag:models'
      expect(store.hasCategory('models')).toBe(true)
    })

    it('should return false for unloaded categories', () => {
      const store = useAssetsStore()

      expect(store.hasCategory('checkpoints')).toBe(false)
      expect(store.hasCategory('unknown-category')).toBe(false)
    })

    it('should return false after category is invalidated', async () => {
      const store = useAssetsStore()
      const assets = [createMockAsset('asset-1')]

      vi.mocked(assetService.getAssetsForNodeType).mockResolvedValue(assets)
      await store.updateModelsForNodeType('CheckpointLoaderSimple')

      expect(store.hasCategory('checkpoints')).toBe(true)

      store.invalidateCategory('checkpoints')

      expect(store.hasCategory('checkpoints')).toBe(false)
    })
  })

  describe('invalidateModelsForCategory', () => {
    it('should clear cache for category and trigger refetch on next access', async () => {
      const store = useAssetsStore()
      const initialAssets = [createMockAsset('initial-1')]
      const refreshedAssets = [
        createMockAsset('refreshed-1'),
        createMockAsset('refreshed-2')
      ]

      vi.mocked(assetService.getAssetsForNodeType).mockResolvedValueOnce(
        initialAssets
      )
      await store.updateModelsForNodeType('CheckpointLoaderSimple')
      expect(store.getAssets('CheckpointLoaderSimple')).toHaveLength(1)

      store.invalidateModelsForCategory('checkpoints')

      // Cache should be cleared
      expect(store.hasCategory('checkpoints')).toBe(false)
      expect(store.getAssets('CheckpointLoaderSimple')).toEqual([])

      // Next fetch should get fresh data
      vi.mocked(assetService.getAssetsForNodeType).mockResolvedValueOnce(
        refreshedAssets
      )
      await store.updateModelsForNodeType('CheckpointLoaderSimple')
      expect(store.getAssets('CheckpointLoaderSimple')).toHaveLength(2)
    })

    it('should clear tag-based caches', async () => {
      const store = useAssetsStore()
      const tagAssets = [createMockAsset('tag-1'), createMockAsset('tag-2')]

      vi.mocked(assetService.getAssetsByTag).mockResolvedValue(tagAssets)
      await store.updateModelsForTag('checkpoints')
      await store.updateModelsForTag('models')

      expect(store.getAssets('tag:checkpoints')).toHaveLength(2)
      expect(store.getAssets('tag:models')).toHaveLength(2)

      store.invalidateModelsForCategory('checkpoints')

      expect(store.getAssets('tag:checkpoints')).toEqual([])
      expect(store.getAssets('tag:models')).toEqual([])
    })

    it('should handle unknown categories gracefully', () => {
      const store = useAssetsStore()

      expect(() =>
        store.invalidateModelsForCategory('unknown-category')
      ).not.toThrow()
    })
  })
})
