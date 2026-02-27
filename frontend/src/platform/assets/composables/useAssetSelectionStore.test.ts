import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { useAssetSelectionStore } from './useAssetSelectionStore'

describe('useAssetSelectionStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  describe('addToSelection', () => {
    it('adds an asset ID to the selection', () => {
      const store = useAssetSelectionStore()
      store.addToSelection('asset-1')
      expect(store.isSelected('asset-1')).toBe(true)
      expect(store.selectedCount).toBe(1)
    })

    it('can add multiple IDs', () => {
      const store = useAssetSelectionStore()
      store.addToSelection('asset-1')
      store.addToSelection('asset-2')
      expect(store.selectedCount).toBe(2)
    })
  })

  describe('removeFromSelection', () => {
    it('removes an asset ID from the selection', () => {
      const store = useAssetSelectionStore()
      store.addToSelection('asset-1')
      store.removeFromSelection('asset-1')
      expect(store.isSelected('asset-1')).toBe(false)
      expect(store.selectedCount).toBe(0)
    })
  })

  describe('setSelection', () => {
    it('replaces entire selection with new IDs', () => {
      const store = useAssetSelectionStore()
      store.addToSelection('asset-1')
      store.addToSelection('asset-2')

      store.setSelection(['asset-3', 'asset-4'])

      expect(store.isSelected('asset-1')).toBe(false)
      expect(store.isSelected('asset-2')).toBe(false)
      expect(store.isSelected('asset-3')).toBe(true)
      expect(store.isSelected('asset-4')).toBe(true)
      expect(store.selectedCount).toBe(2)
    })

    it('can set to empty selection', () => {
      const store = useAssetSelectionStore()
      store.addToSelection('asset-1')
      store.setSelection([])
      expect(store.selectedCount).toBe(0)
    })
  })

  describe('clearSelection', () => {
    it('clears all selections and resets lastSelectedIndex', () => {
      const store = useAssetSelectionStore()
      store.addToSelection('asset-1')
      store.setLastSelectedIndex(5)

      store.clearSelection()

      expect(store.selectedCount).toBe(0)
      expect(store.lastSelectedIndex).toBe(-1)
    })

    it('resets lastSelectedAssetId', () => {
      const store = useAssetSelectionStore()
      store.addToSelection('asset-1')
      store.setLastSelectedAssetId('asset-1')

      store.clearSelection()

      expect(store.lastSelectedAssetId).toBe(null)
    })
  })

  describe('toggleSelection', () => {
    it('adds unselected item', () => {
      const store = useAssetSelectionStore()
      store.toggleSelection('asset-1')
      expect(store.isSelected('asset-1')).toBe(true)
    })

    it('removes selected item', () => {
      const store = useAssetSelectionStore()
      store.addToSelection('asset-1')
      store.toggleSelection('asset-1')
      expect(store.isSelected('asset-1')).toBe(false)
    })
  })

  describe('isSelected', () => {
    it('returns true for selected items', () => {
      const store = useAssetSelectionStore()
      store.addToSelection('asset-1')
      expect(store.isSelected('asset-1')).toBe(true)
    })

    it('returns false for unselected items', () => {
      const store = useAssetSelectionStore()
      expect(store.isSelected('asset-1')).toBe(false)
    })
  })

  describe('setLastSelectedIndex', () => {
    it('updates lastSelectedIndex', () => {
      const store = useAssetSelectionStore()
      store.setLastSelectedIndex(10)
      expect(store.lastSelectedIndex).toBe(10)
    })
  })

  describe('computed properties', () => {
    it('hasSelection returns true when items are selected', () => {
      const store = useAssetSelectionStore()
      expect(store.hasSelection).toBe(false)
      store.addToSelection('asset-1')
      expect(store.hasSelection).toBe(true)
    })

    it('selectedIdsArray returns array of selected IDs', () => {
      const store = useAssetSelectionStore()
      store.addToSelection('asset-1')
      store.addToSelection('asset-2')
      expect(store.selectedIdsArray).toContain('asset-1')
      expect(store.selectedIdsArray).toContain('asset-2')
    })
  })
})
