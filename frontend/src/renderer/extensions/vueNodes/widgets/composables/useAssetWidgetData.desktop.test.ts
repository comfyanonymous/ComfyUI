import { describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

import { useAssetWidgetData } from '@/renderer/extensions/vueNodes/widgets/composables/useAssetWidgetData'

vi.mock('@/platform/distribution/types', () => ({
  isCloud: false
}))

const mockUpdateModelsForNodeType = vi.fn()
const mockGetCategoryForNodeType = vi.fn()

vi.mock('@/stores/assetsStore', () => ({
  useAssetsStore: () => ({
    getAssets: () => [],
    isModelLoading: () => false,
    getError: () => undefined,
    hasAssetKey: () => false,
    updateModelsForNodeType: mockUpdateModelsForNodeType
  })
}))

vi.mock('@/stores/modelToNodeStore', () => ({
  useModelToNodeStore: () => ({
    getCategoryForNodeType: mockGetCategoryForNodeType
  })
}))

describe('useAssetWidgetData (desktop/isCloud=false)', () => {
  it('returns empty/default values without calling stores', () => {
    const nodeType = ref('CheckpointLoaderSimple')
    const { category, assets, isLoading, error } = useAssetWidgetData(nodeType)

    expect(category.value).toBeUndefined()
    expect(assets.value).toEqual([])
    expect(isLoading.value).toBe(false)
    expect(error.value).toBeNull()
    expect(mockUpdateModelsForNodeType).not.toHaveBeenCalled()
    expect(mockGetCategoryForNodeType).not.toHaveBeenCalled()
  })
})
