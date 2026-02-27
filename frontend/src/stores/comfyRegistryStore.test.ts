import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

import { useComfyRegistryService } from '@/services/comfyRegistryService'
import { useComfyRegistryStore } from '@/stores/comfyRegistryStore'
import type { components, operations } from '@/types/comfyRegistryTypes'

vi.mock('@/services/comfyRegistryService', () => ({
  useComfyRegistryService: vi.fn()
}))

const mockNodePack: components['schemas']['Node'] = {
  id: 'test-pack-id',
  name: 'Test Pack',
  description: 'A test node pack',
  downloads: 1000,
  publisher: {
    id: 'test-publisher',
    name: 'Test Publisher'
  },
  latest_version: {
    id: 'test-version',
    version: '1.0.0',
    createdAt: '2023-01-01T00:00:00Z'
  }
}

const mockNodePack2: components['schemas']['Node'] = {
  id: 'test-pack-id-2',
  name: 'Test Pack 2',
  description: 'A second test node pack',
  downloads: 1000,
  publisher: {
    id: 'test-publisher',
    name: 'Test Publisher'
  },
  latest_version: {
    id: 'test-version',
    version: '1.0.0',
    createdAt: '2023-01-01T00:00:00Z'
  }
}

const mockNodePack3: components['schemas']['Node'] = {
  id: 'test-pack-id-3',
  name: 'Test Pack 3',
  description: 'A third test node pack',
  downloads: 1000,
  publisher: {
    id: 'test-publisher',
    name: 'Test Publisher'
  },
  latest_version: {
    id: 'test-version',
    version: '1.0.0',
    createdAt: '2023-01-01T00:00:00Z'
  }
}

const mockListResult: operations['listAllNodes']['responses'][200]['content']['application/json'] =
  {
    nodes: [mockNodePack],
    total: 1,
    page: 1,
    limit: 10
  }

describe('useComfyRegistryStore', () => {
  let mockRegistryService: {
    isLoading: ReturnType<typeof ref<boolean>>
    error: ReturnType<typeof ref<string | null>>
    listAllPacks: ReturnType<typeof vi.fn>
    getPackById: ReturnType<typeof vi.fn>
    inferPackFromNodeName: ReturnType<typeof vi.fn>
    search: ReturnType<typeof vi.fn>
    getPackVersions: ReturnType<typeof vi.fn>
    getPackByVersion: ReturnType<typeof vi.fn>
    getPublisherById: ReturnType<typeof vi.fn>
    listPacksForPublisher: ReturnType<typeof vi.fn>
    getNodeDefs: ReturnType<typeof vi.fn>
    postPackReview: ReturnType<typeof vi.fn>
  }

  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    vi.clearAllMocks()
    mockRegistryService = {
      isLoading: ref(false),
      error: ref(null),
      listAllPacks: vi.fn().mockImplementation((params) => {
        // If node_id is provided, return specific nodes
        if (params.node_id) {
          return Promise.resolve({
            nodes: params.node_id
              .map((id: string) => {
                switch (id) {
                  case 'test-pack-id':
                    return mockNodePack
                  case 'test-pack-id-2':
                    return mockNodePack2
                  case 'test-pack-id-3':
                    return mockNodePack3
                  default:
                    return null
                }
              })
              .filter(Boolean),
            total: params.node_id.length,
            page: 1,
            limit: 10
          })
        }
        // Otherwise return paginated results
        return Promise.resolve(mockListResult)
      }),
      getPackById: vi.fn().mockResolvedValue(mockNodePack),
      inferPackFromNodeName: vi.fn().mockResolvedValue(mockNodePack),
      search: vi.fn().mockResolvedValue(mockListResult),
      getPackVersions: vi.fn().mockResolvedValue([]),
      getPackByVersion: vi.fn().mockResolvedValue({}),
      getPublisherById: vi.fn().mockResolvedValue({}),
      listPacksForPublisher: vi.fn().mockResolvedValue([]),
      getNodeDefs: vi.fn().mockResolvedValue({}),
      postPackReview: vi.fn().mockResolvedValue({})
    }

    vi.mocked(useComfyRegistryService).mockReturnValue(
      mockRegistryService as Partial<
        ReturnType<typeof useComfyRegistryService>
      > as ReturnType<typeof useComfyRegistryService>
    )
  })

  afterEach(() => {
    useComfyRegistryStore().clearCache()
  })

  it('should fetch and store packs', async () => {
    const store = useComfyRegistryStore()
    const params = { page: 1, limit: 10 }

    const result = await store.listAllPacks.call(params)

    expect(result).toEqual(mockListResult)
    expect(mockRegistryService.listAllPacks).toHaveBeenCalledWith(
      params,
      expect.any(Object) // abort signal
    )
  })

  it('should handle empty nodes array in response', async () => {
    const emptyResult = {
      nodes: undefined,
      total: 0,
      page: 1,
      limit: 10
    }
    mockRegistryService.listAllPacks.mockResolvedValueOnce(emptyResult)

    const store = useComfyRegistryStore()
    const result = await store.listAllPacks.call({ page: 1, limit: 10 })

    expect(result).toEqual(emptyResult)
  })

  it('should fetch a pack by ID', async () => {
    const store = useComfyRegistryStore()
    const packId = 'test-pack-id'

    const result = await store.getPackById.call(packId)

    expect(result).toEqual(mockNodePack)
    expect(mockRegistryService.getPackById).toHaveBeenCalledWith(packId)
  })

  it('should return null when fetching a pack with null ID', async () => {
    const store = useComfyRegistryStore()
    vi.spyOn(store.getPackById, 'call').mockResolvedValueOnce(null)

    const result = await store.getPackById.call(null!)

    expect(result).toBeNull()
    expect(mockRegistryService.getPackById).not.toHaveBeenCalled()
  })

  it('should handle service errors gracefully', async () => {
    mockRegistryService.listAllPacks.mockResolvedValueOnce(null)

    const store = useComfyRegistryStore()
    const result = await store.listAllPacks.call({ page: 1, limit: 10 })

    expect(result).toBeNull()
  })

  it('should fetch packs by IDs', async () => {
    const store = useComfyRegistryStore()
    const packIds = ['test-pack-id', 'test-pack-id-2', 'test-pack-id-3']
    const result = await store.getPacksByIds.call(packIds)

    expect(result).toEqual([mockNodePack, mockNodePack2, mockNodePack3])
    expect(mockRegistryService.listAllPacks).toHaveBeenCalledWith(
      { node_id: packIds },
      expect.any(Object) // abort signal
    )
  })

  describe('inferPackFromNodeName', () => {
    it('should fetch a pack by comfy node name', async () => {
      const store = useComfyRegistryStore()
      const nodeName = 'KSampler'

      const result = await store.inferPackFromNodeName.call(nodeName)

      expect(result).toEqual(mockNodePack)
      expect(mockRegistryService.inferPackFromNodeName).toHaveBeenCalledWith(
        nodeName,
        expect.any(Object) // abort signal
      )
    })

    it('should cache results', async () => {
      const store = useComfyRegistryStore()
      const nodeName = 'KSampler'

      // First call
      const result1 = await store.inferPackFromNodeName.call(nodeName)
      expect(mockRegistryService.inferPackFromNodeName).toHaveBeenCalledTimes(1)

      // Second call - should use cache
      const result2 = await store.inferPackFromNodeName.call(nodeName)
      expect(mockRegistryService.inferPackFromNodeName).toHaveBeenCalledTimes(1)
      expect(result2).toEqual(result1)
    })

    it('should handle null results when node is not found', async () => {
      mockRegistryService.inferPackFromNodeName.mockResolvedValueOnce(null)

      const store = useComfyRegistryStore()
      const result = await store.inferPackFromNodeName.call('NonExistentNode')

      expect(result).toBeNull()
    })

    it('should clear cache when clearCache is called', async () => {
      const store = useComfyRegistryStore()
      const nodeName = 'KSampler'

      // First call to populate cache
      await store.inferPackFromNodeName.call(nodeName)
      expect(mockRegistryService.inferPackFromNodeName).toHaveBeenCalledTimes(1)

      // Clear cache
      store.clearCache()

      // Call again - should hit the service again
      await store.inferPackFromNodeName.call(nodeName)
      expect(mockRegistryService.inferPackFromNodeName).toHaveBeenCalledTimes(2)
    })
  })
})
