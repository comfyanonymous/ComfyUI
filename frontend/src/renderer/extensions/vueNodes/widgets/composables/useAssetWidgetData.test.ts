import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick, ref } from 'vue'

import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import { useAssetWidgetData } from '@/renderer/extensions/vueNodes/widgets/composables/useAssetWidgetData'

vi.mock('@/platform/distribution/types', () => ({
  isCloud: true
}))

const mockAssetsByKey = new Map<string, AssetItem[]>()
const mockLoadingByKey = new Map<string, boolean>()
const mockErrorByKey = new Map<string, Error | undefined>()
const mockInitializedKeys = new Set<string>()
const mockUpdateModelsForNodeType = vi.fn()
const mockGetCategoryForNodeType = vi.fn()

vi.mock('@/stores/assetsStore', () => ({
  useAssetsStore: () => ({
    getAssets: (key: string) => mockAssetsByKey.get(key) ?? [],
    isModelLoading: (key: string) => mockLoadingByKey.get(key) ?? false,
    getError: (key: string) => mockErrorByKey.get(key),
    hasAssetKey: (key: string) => mockInitializedKeys.has(key),
    updateModelsForNodeType: mockUpdateModelsForNodeType
  })
}))

vi.mock('@/stores/modelToNodeStore', () => ({
  useModelToNodeStore: () => ({
    getCategoryForNodeType: mockGetCategoryForNodeType
  })
}))

describe('useAssetWidgetData (cloud mode, isCloud=true)', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockAssetsByKey.clear()
    mockLoadingByKey.clear()
    mockErrorByKey.clear()
    mockInitializedKeys.clear()
    mockGetCategoryForNodeType.mockReturnValue(undefined)

    mockUpdateModelsForNodeType.mockImplementation(
      async (): Promise<AssetItem[]> => {
        return []
      }
    )
  })

  const createMockAsset = (
    id: string,
    name: string,
    filename: string,
    previewUrl?: string
  ): AssetItem => ({
    id,
    name,
    size: 1024,
    tags: ['models', 'checkpoints'],
    created_at: '2025-01-01T00:00:00Z',
    preview_url: previewUrl,
    user_metadata: {
      filename
    }
  })

  it('fetches assets for a given node type', async () => {
    const mockAssets: AssetItem[] = [
      createMockAsset(
        'asset-1',
        'Beautiful Model',
        'models/beautiful_model.safetensors',
        '/api/preview/asset-1'
      ),
      createMockAsset('asset-2', 'Model B', 'model_b.safetensors', '/preview/2')
    ]

    mockGetCategoryForNodeType.mockReturnValue('checkpoints')

    mockUpdateModelsForNodeType.mockImplementation(
      async (_nodeType: string): Promise<AssetItem[]> => {
        mockInitializedKeys.add(_nodeType)
        mockAssetsByKey.set(_nodeType, mockAssets)
        mockLoadingByKey.set(_nodeType, false)
        return mockAssets
      }
    )

    const nodeType = ref('CheckpointLoaderSimple')
    const { category, assets, isLoading } = useAssetWidgetData(nodeType)

    await nextTick()
    await vi.waitFor(() => !isLoading.value)

    expect(mockUpdateModelsForNodeType).toHaveBeenCalledWith(
      'CheckpointLoaderSimple'
    )
    expect(category.value).toBe('checkpoints')
    expect(assets.value).toEqual(mockAssets)
    expect(assets.value).toHaveLength(2)
    expect(assets.value[0].id).toBe('asset-1')
    expect(assets.value[0].name).toBe('Beautiful Model')
    expect(assets.value[0].preview_url).toBe('/api/preview/asset-1')
  })

  it('handles API errors gracefully', async () => {
    const mockError = new Error('Network error')

    mockUpdateModelsForNodeType.mockImplementation(
      async (_nodeType: string): Promise<AssetItem[]> => {
        mockInitializedKeys.add(_nodeType)
        mockErrorByKey.set(_nodeType, mockError)
        mockAssetsByKey.set(_nodeType, [])
        mockLoadingByKey.set(_nodeType, false)
        return []
      }
    )

    const nodeType = ref('CheckpointLoaderSimple')
    const { assets, error, isLoading } = useAssetWidgetData(nodeType)

    await nextTick()
    await vi.waitFor(() => !isLoading.value)

    expect(error.value).toBe(mockError)
    expect(assets.value).toEqual([])
  })

  it('returns empty for unknown node type', async () => {
    mockGetCategoryForNodeType.mockReturnValue(undefined)

    mockUpdateModelsForNodeType.mockImplementation(
      async (_nodeType: string): Promise<AssetItem[]> => {
        mockInitializedKeys.add(_nodeType)
        mockAssetsByKey.set(_nodeType, [])
        mockLoadingByKey.set(_nodeType, false)
        return []
      }
    )

    const nodeType = ref('UnknownNodeType')
    const { category, assets } = useAssetWidgetData(nodeType)

    await nextTick()

    expect(category.value).toBeUndefined()
    expect(assets.value).toEqual([])
  })

  describe('MaybeRefOrGetter parameter support', () => {
    it('accepts plain string value', async () => {
      const mockAssets: AssetItem[] = [
        createMockAsset('asset-1', 'Model A', 'model_a.safetensors')
      ]

      mockGetCategoryForNodeType.mockReturnValue('checkpoints')
      mockUpdateModelsForNodeType.mockImplementation(
        async (_nodeType: string): Promise<AssetItem[]> => {
          mockInitializedKeys.add(_nodeType)
          mockAssetsByKey.set(_nodeType, mockAssets)
          mockLoadingByKey.set(_nodeType, false)
          return mockAssets
        }
      )

      const { category, assets, isLoading } = useAssetWidgetData(
        'CheckpointLoaderSimple'
      )

      await nextTick()
      await vi.waitFor(() => !isLoading.value)

      expect(mockUpdateModelsForNodeType).toHaveBeenCalledWith(
        'CheckpointLoaderSimple'
      )
      expect(category.value).toBe('checkpoints')
      expect(assets.value).toEqual(mockAssets)
    })

    it('accepts getter function', async () => {
      const mockAssets: AssetItem[] = [
        createMockAsset('asset-1', 'Model A', 'model_a.safetensors')
      ]

      mockGetCategoryForNodeType.mockReturnValue('loras')
      mockUpdateModelsForNodeType.mockImplementation(
        async (_nodeType: string): Promise<AssetItem[]> => {
          mockInitializedKeys.add(_nodeType)
          mockAssetsByKey.set(_nodeType, mockAssets)
          mockLoadingByKey.set(_nodeType, false)
          return mockAssets
        }
      )

      const nodeType = ref('LoraLoader')
      const { category, assets, isLoading } = useAssetWidgetData(
        () => nodeType.value
      )

      await nextTick()
      await vi.waitFor(() => !isLoading.value)

      expect(mockUpdateModelsForNodeType).toHaveBeenCalledWith('LoraLoader')
      expect(category.value).toBe('loras')
      expect(assets.value).toEqual(mockAssets)
    })

    it('accepts ref (backward compatibility)', async () => {
      const mockAssets: AssetItem[] = [
        createMockAsset('asset-1', 'Model A', 'model_a.safetensors')
      ]

      mockGetCategoryForNodeType.mockReturnValue('checkpoints')
      mockUpdateModelsForNodeType.mockImplementation(
        async (_nodeType: string): Promise<AssetItem[]> => {
          mockInitializedKeys.add(_nodeType)
          mockAssetsByKey.set(_nodeType, mockAssets)
          mockLoadingByKey.set(_nodeType, false)
          return mockAssets
        }
      )

      const nodeTypeRef = ref('CheckpointLoaderSimple')
      const { category, assets, isLoading } = useAssetWidgetData(nodeTypeRef)

      await nextTick()
      await vi.waitFor(() => !isLoading.value)

      expect(mockUpdateModelsForNodeType).toHaveBeenCalledWith(
        'CheckpointLoaderSimple'
      )
      expect(category.value).toBe('checkpoints')
      expect(assets.value).toEqual(mockAssets)
    })

    it('handles undefined node type gracefully', async () => {
      const { category, assets, isLoading, error } =
        useAssetWidgetData(undefined)

      await nextTick()

      expect(mockUpdateModelsForNodeType).not.toHaveBeenCalled()
      expect(category.value).toBeUndefined()
      expect(assets.value).toEqual([])
      expect(isLoading.value).toBe(false)
      expect(error.value).toBeNull()
    })
  })
})
