import { describe, expect, it, vi } from 'vitest'

import { useAssetBrowserDialog } from '@/platform/assets/composables/useAssetBrowserDialog'
import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import { useDialogStore } from '@/stores/dialogStore'

vi.mock('@/stores/dialogStore')

vi.mock('@/i18n', () => ({
  t: (key: string, params?: Record<string, string>) => {
    if (params) {
      return `${key}:${JSON.stringify(params)}`
    }
    return key
  }
}))

function createMockAsset(overrides: Partial<AssetItem> = {}): AssetItem {
  return {
    id: 'asset-123',
    name: 'test-model.safetensors',
    size: 1024,
    created_at: '2025-10-01T00:00:00Z',
    tags: ['models', 'checkpoints'],
    user_metadata: {
      filename: 'models/checkpoints/test-model.safetensors'
    },
    ...overrides
  }
}

function setupDialogMocks() {
  const mockShowDialog = vi.fn()
  const mockCloseDialog = vi.fn()
  vi.mocked(useDialogStore, { partial: true }).mockReturnValue({
    showDialog: mockShowDialog,
    closeDialog: mockCloseDialog
  })

  return { mockShowDialog, mockCloseDialog }
}

describe('useAssetBrowserDialog', () => {
  describe('Asset Selection Flow', () => {
    it('auto-closes dialog when asset is selected', async () => {
      const { mockShowDialog, mockCloseDialog } = setupDialogMocks()
      const assetBrowserDialog = useAssetBrowserDialog()
      const onAssetSelected = vi.fn()

      await assetBrowserDialog.show({
        nodeType: 'CheckpointLoaderSimple',
        inputName: 'ckpt_name',
        onAssetSelected
      })

      const dialogCall = mockShowDialog.mock.calls[0][0]
      const onSelectHandler = dialogCall.props.onSelect

      const mockAsset = {
        id: 'test-asset-id',
        name: 'test.safetensors',
        size: 1024,
        created_at: '2025-10-01T00:00:00Z',
        tags: ['models', 'checkpoints'],
        user_metadata: { filename: 'selected-asset-path' }
      }
      onSelectHandler(mockAsset)

      expect(onAssetSelected).toHaveBeenCalledWith(mockAsset)
      expect(mockCloseDialog).toHaveBeenCalledWith({
        key: 'global-asset-browser'
      })
    })

    it('closes dialog when close handler is called', async () => {
      const { mockShowDialog, mockCloseDialog } = setupDialogMocks()
      const assetBrowserDialog = useAssetBrowserDialog()

      await assetBrowserDialog.show({
        nodeType: 'CheckpointLoaderSimple',
        inputName: 'ckpt_name'
      })

      const dialogCall = mockShowDialog.mock.calls[0][0]
      const onCloseHandler = dialogCall.props.onClose

      onCloseHandler()

      expect(mockCloseDialog).toHaveBeenCalledWith({
        key: 'global-asset-browser'
      })
    })
  })

  describe('.browse() method', () => {
    it('opens asset browser dialog with tag-based filtering', async () => {
      const { mockShowDialog } = setupDialogMocks()
      const assetBrowserDialog = useAssetBrowserDialog()
      await assetBrowserDialog.browse({
        assetType: 'models',
        title: 'Model Library'
      })

      expect(mockShowDialog).toHaveBeenCalledWith(
        expect.objectContaining({
          key: 'global-asset-browser',
          props: expect.objectContaining({
            showLeftPanel: true,
            assetType: 'models'
          })
        })
      )
    })

    it('calls onAssetSelected callback when asset is selected', async () => {
      const { mockShowDialog } = setupDialogMocks()
      const assetBrowserDialog = useAssetBrowserDialog()
      const mockAsset = createMockAsset()
      const onAssetSelected = vi.fn()
      await assetBrowserDialog.browse({
        assetType: 'models',
        onAssetSelected
      })

      const dialogCall = mockShowDialog.mock.calls[0][0]
      const onSelectHandler = dialogCall.props.onSelect

      onSelectHandler(mockAsset)

      expect(onAssetSelected).toHaveBeenCalledWith(mockAsset)
    })

    it('closes dialog after asset selection', async () => {
      const { mockShowDialog, mockCloseDialog } = setupDialogMocks()
      const assetBrowserDialog = useAssetBrowserDialog()
      const mockAsset = createMockAsset()
      await assetBrowserDialog.browse({
        assetType: 'models'
      })

      const dialogCall = mockShowDialog.mock.calls[0][0]
      const onSelectHandler = dialogCall.props.onSelect

      onSelectHandler(mockAsset)

      expect(mockCloseDialog).toHaveBeenCalledWith({
        key: 'global-asset-browser'
      })
    })

    it('uses custom title when provided', async () => {
      const { mockShowDialog } = setupDialogMocks()
      const assetBrowserDialog = useAssetBrowserDialog()
      await assetBrowserDialog.browse({
        assetType: 'models',
        title: 'Custom Model Browser'
      })

      const dialogCall = mockShowDialog.mock.calls[0][0]
      expect(dialogCall.props.title).toBe('Custom Model Browser')
    })
  })

  describe('.show() behavior', () => {
    it('opens dialog without pre-fetched assets', async () => {
      const { mockShowDialog } = setupDialogMocks()
      const assetBrowserDialog = useAssetBrowserDialog()

      await assetBrowserDialog.show({
        nodeType: 'CheckpointLoaderSimple',
        inputName: 'ckpt_name'
      })

      const dialogCall = mockShowDialog.mock.calls[0][0]
      expect(dialogCall.props.nodeType).toBe('CheckpointLoaderSimple')
      expect(dialogCall.props.assets).toBeUndefined()
    })
  })
})
