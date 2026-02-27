import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import type { AssetItem } from '@/platform/assets/schemas/assetSchema'

import { useMediaAssetActions } from './useMediaAssetActions'

// Use vi.hoisted to create a mutable reference for isCloud
const mockIsCloud = vi.hoisted(() => ({ value: false }))

// Track the filename passed to createAnnotatedPath
const capturedFilenames = vi.hoisted(() => ({ values: [] as string[] }))

vi.mock('@/platform/distribution/types', () => ({
  get isCloud() {
    return mockIsCloud.value
  }
}))

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({
    add: vi.fn()
  })
}))

vi.mock('vue-i18n', () => ({
  useI18n: () => ({
    t: (key: string) => key
  }),
  createI18n: () => ({
    global: {
      t: (key: string) => key
    }
  })
}))

const mockShowDialog = vi.hoisted(() => vi.fn())
vi.mock('@/stores/dialogStore', () => ({
  useDialogStore: () => ({
    showDialog: mockShowDialog
  })
}))

const mockInvalidateModelsForCategory = vi.hoisted(() => vi.fn())
const mockSetAssetDeleting = vi.hoisted(() => vi.fn())
const mockUpdateHistory = vi.hoisted(() => vi.fn())
const mockUpdateInputs = vi.hoisted(() => vi.fn())
const mockHasCategory = vi.hoisted(() => vi.fn())
vi.mock('@/stores/assetsStore', () => ({
  useAssetsStore: () => ({
    setAssetDeleting: mockSetAssetDeleting,
    updateHistory: mockUpdateHistory,
    updateInputs: mockUpdateInputs,
    invalidateModelsForCategory: mockInvalidateModelsForCategory,
    hasCategory: mockHasCategory
  })
}))

vi.mock('@/stores/modelToNodeStore', () => ({
  useModelToNodeStore: () => ({})
}))

vi.mock('@/composables/useCopyToClipboard', () => ({
  useCopyToClipboard: () => ({
    copyToClipboard: vi.fn()
  })
}))

vi.mock('@/platform/workflow/core/services/workflowActionsService', () => ({
  useWorkflowActionsService: () => ({
    openWorkflowAction: vi.fn(),
    exportWorkflowAction: vi.fn()
  })
}))

vi.mock('@/services/litegraphService', () => ({
  useLitegraphService: () => ({
    addNodeOnGraph: vi.fn().mockReturnValue({
      widgets: [{ name: 'image', value: '', callback: vi.fn() }],
      graph: { setDirtyCanvas: vi.fn() }
    } as unknown as LGraphNode),
    getCanvasCenter: vi.fn().mockReturnValue([100, 100])
  })
}))

vi.mock('@/stores/nodeDefStore', () => ({
  useNodeDefStore: () => ({
    nodeDefsByName: {
      LoadImage: {
        name: 'LoadImage',
        display_name: 'Load Image'
      }
    }
  })
}))

vi.mock('@/utils/createAnnotatedPath', () => ({
  createAnnotatedPath: vi.fn((item: { filename: string }) => {
    capturedFilenames.values.push(item.filename)
    return item.filename
  })
}))

vi.mock('@/utils/loaderNodeUtil', () => ({
  detectNodeTypeFromFilename: vi.fn().mockReturnValue({
    nodeType: 'LoadImage',
    widgetName: 'image'
  })
}))

vi.mock('@/utils/typeGuardUtil', () => ({
  isResultItemType: vi.fn().mockReturnValue(true)
}))

const mockGetAssetType = vi.hoisted(() => vi.fn())
vi.mock('@/platform/assets/utils/assetTypeUtil', () => ({
  getAssetType: mockGetAssetType
}))

vi.mock('../schemas/assetMetadataSchema', () => ({
  getOutputAssetMetadata: vi.fn().mockReturnValue(null)
}))

const mockDeleteAsset = vi.hoisted(() => vi.fn())
vi.mock('../services/assetService', () => ({
  assetService: {
    deleteAsset: mockDeleteAsset
  }
}))

vi.mock('@/scripts/api', () => ({
  api: {
    deleteItem: vi.fn(),
    apiURL: vi.fn((path: string) => `http://localhost:8188/api${path}`),
    internalURL: vi.fn((path: string) => `http://localhost:8188${path}`),
    addEventListener: vi.fn(),
    removeEventListener: vi.fn(),
    user: 'test-user'
  }
}))

function createMockAsset(overrides: Partial<AssetItem> = {}): AssetItem {
  return {
    id: 'test-asset-id',
    name: 'original-name.jpeg',
    size: 1024,
    created_at: '2025-01-01T00:00:00Z',
    tags: ['input'],
    ...overrides
  }
}

describe('useMediaAssetActions', () => {
  beforeEach(() => {
    vi.resetModules()
    setActivePinia(createTestingPinia({ stubActions: false }))
    vi.clearAllMocks()
    capturedFilenames.values = []
    mockIsCloud.value = false
  })

  describe('addWorkflow', () => {
    describe('OSS mode (isCloud = false)', () => {
      beforeEach(() => {
        mockIsCloud.value = false
      })

      it('should use asset.name as filename', async () => {
        const actions = useMediaAssetActions()

        const asset = createMockAsset({
          name: 'my-image.jpeg',
          asset_hash: 'hash123.jpeg'
        })

        await actions.addWorkflow(asset)

        expect(capturedFilenames.values).toContain('my-image.jpeg')
      })
    })

    describe('Cloud mode (isCloud = true)', () => {
      beforeEach(() => {
        mockIsCloud.value = true
      })

      it('should use asset_hash as filename when available', async () => {
        const actions = useMediaAssetActions()

        const asset = createMockAsset({
          name: 'original.jpeg',
          asset_hash: 'abc123hash.jpeg'
        })

        await actions.addWorkflow(asset)

        expect(capturedFilenames.values).toContain('abc123hash.jpeg')
      })

      it('should fall back to asset.name when asset_hash is not available', async () => {
        const actions = useMediaAssetActions()

        const asset = createMockAsset({
          name: 'fallback-name.jpeg',
          asset_hash: undefined
        })

        await actions.addWorkflow(asset)

        expect(capturedFilenames.values).toContain('fallback-name.jpeg')
      })

      it('should fall back to asset.name when asset_hash is null', async () => {
        const actions = useMediaAssetActions()

        const asset = createMockAsset({
          name: 'fallback-null.jpeg',
          asset_hash: null
        })

        await actions.addWorkflow(asset)

        expect(capturedFilenames.values).toContain('fallback-null.jpeg')
      })
    })
  })

  describe('addMultipleToWorkflow', () => {
    describe('Cloud mode (isCloud = true)', () => {
      beforeEach(() => {
        mockIsCloud.value = true
      })

      it('should use asset_hash for each asset', async () => {
        const actions = useMediaAssetActions()

        const assets = [
          createMockAsset({
            id: '1',
            name: 'file1.jpeg',
            asset_hash: 'hash1.jpeg'
          }),
          createMockAsset({
            id: '2',
            name: 'file2.jpeg',
            asset_hash: 'hash2.jpeg'
          })
        ]

        await actions.addMultipleToWorkflow(assets)

        expect(capturedFilenames.values).toContain('hash1.jpeg')
        expect(capturedFilenames.values).toContain('hash2.jpeg')
        expect(capturedFilenames.values).not.toContain('file1.jpeg')
        expect(capturedFilenames.values).not.toContain('file2.jpeg')
      })
    })
  })

  describe('deleteAssets - model cache invalidation', () => {
    beforeEach(() => {
      mockIsCloud.value = true
      mockGetAssetType.mockReturnValue('input')
      mockDeleteAsset.mockResolvedValue(undefined)
      mockInvalidateModelsForCategory.mockClear()
      mockSetAssetDeleting.mockClear()
      mockUpdateHistory.mockClear()
      mockUpdateInputs.mockClear()
      mockHasCategory.mockClear()
      // By default, hasCategory returns true for model categories
      mockHasCategory.mockImplementation(
        (tag: string) => tag === 'checkpoints' || tag === 'loras'
      )
    })

    it('should invalidate model cache when deleting a model asset', async () => {
      const actions = useMediaAssetActions()

      const modelAsset = createMockAsset({
        id: 'checkpoint-1',
        name: 'model.safetensors',
        tags: ['models', 'checkpoints']
      })

      mockShowDialog.mockImplementation(
        ({ props }: { props: { onConfirm: () => Promise<void> } }) => {
          void props.onConfirm()
        }
      )

      await actions.deleteAssets(modelAsset)

      // Only 'checkpoints' exists in cache; 'models' is excluded
      expect(mockInvalidateModelsForCategory).toHaveBeenCalledTimes(1)
      expect(mockInvalidateModelsForCategory).toHaveBeenCalledWith(
        'checkpoints'
      )
    })

    it('should invalidate multiple categories for multiple assets', async () => {
      const actions = useMediaAssetActions()

      const assets = [
        createMockAsset({ id: '1', tags: ['models', 'checkpoints'] }),
        createMockAsset({ id: '2', tags: ['models', 'loras'] })
      ]

      mockShowDialog.mockImplementation(
        ({ props }: { props: { onConfirm: () => Promise<void> } }) => {
          void props.onConfirm()
        }
      )

      await actions.deleteAssets(assets)

      expect(mockInvalidateModelsForCategory).toHaveBeenCalledWith(
        'checkpoints'
      )
      expect(mockInvalidateModelsForCategory).toHaveBeenCalledWith('loras')
    })

    it('should not invalidate model cache for non-model assets', async () => {
      const actions = useMediaAssetActions()

      const inputAsset = createMockAsset({
        id: 'input-1',
        name: 'image.png',
        tags: ['input']
      })

      mockShowDialog.mockImplementation(
        ({ props }: { props: { onConfirm: () => Promise<void> } }) => {
          void props.onConfirm()
        }
      )

      await actions.deleteAssets(inputAsset)

      // 'input' tag is excluded, so no cache invalidation
      expect(mockInvalidateModelsForCategory).not.toHaveBeenCalled()
    })

    it('should only invalidate categories that exist in cache', async () => {
      const actions = useMediaAssetActions()

      // hasCategory returns false for 'unknown-category'
      mockHasCategory.mockImplementation((tag: string) => tag === 'checkpoints')

      const assets = [
        createMockAsset({ id: '1', tags: ['models', 'checkpoints'] }),
        createMockAsset({ id: '2', tags: ['models', 'unknown-category'] })
      ]

      mockShowDialog.mockImplementation(
        ({ props }: { props: { onConfirm: () => Promise<void> } }) => {
          void props.onConfirm()
        }
      )

      await actions.deleteAssets(assets)

      // Only checkpoints should be invalidated (unknown-category not in cache)
      expect(mockInvalidateModelsForCategory).toHaveBeenCalledTimes(1)
      expect(mockInvalidateModelsForCategory).toHaveBeenCalledWith(
        'checkpoints'
      )
    })
  })
})
