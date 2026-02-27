import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { legacyMenuCompat } from '@/lib/litegraph/src/contextMenuCompat'
import type { IContextMenuValue } from '@/lib/litegraph/src/litegraph'
import { LGraphCanvas } from '@/lib/litegraph/src/litegraph'
import { createMockCanvas } from '@/utils/__tests__/litegraphTestUtils'

describe('contextMenuCompat', () => {
  let originalGetCanvasMenuOptions: typeof LGraphCanvas.prototype.getCanvasMenuOptions
  let mockCanvas: LGraphCanvas

  beforeEach(() => {
    // Save original method
    originalGetCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions

    // Create mock canvas
    mockCanvas = createMockCanvas({
      constructor: {
        prototype: LGraphCanvas.prototype
      } as typeof LGraphCanvas
    } as Partial<LGraphCanvas>)

    // Clear console warnings
    vi.spyOn(console, 'warn').mockImplementation(() => {})
  })

  afterEach(() => {
    // Restore original method
    LGraphCanvas.prototype.getCanvasMenuOptions = originalGetCanvasMenuOptions
    vi.restoreAllMocks()
  })

  describe('install', () => {
    it('should install compatibility layer on prototype', () => {
      const methodName = 'getCanvasMenuOptions'

      // Install compatibility layer
      legacyMenuCompat.install(LGraphCanvas.prototype, methodName)

      // The method should still be callable
      expect(typeof LGraphCanvas.prototype.getCanvasMenuOptions).toBe(
        'function'
      )
    })

    it('should detect monkey patches and warn', () => {
      const methodName = 'getCanvasMenuOptions'
      const warnSpy = vi.spyOn(console, 'warn')

      // Install compatibility layer
      legacyMenuCompat.install(LGraphCanvas.prototype, methodName)

      // Set current extension before monkey-patching
      legacyMenuCompat.setCurrentExtension('Test Extension')

      // Simulate extension monkey-patching
      const original = LGraphCanvas.prototype.getCanvasMenuOptions
      LGraphCanvas.prototype.getCanvasMenuOptions =
        function (): (IContextMenuValue | null)[] {
          const items = original.call(this)
          items.push({ content: 'Custom Item', callback: () => {} })
          return items
        }

      // Should have logged a warning with extension name
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('[DEPRECATED]'),
        expect.any(String),
        expect.any(String)
      )
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining('"Test Extension"'),
        expect.any(String),
        expect.any(String)
      )

      // Clear extension
      legacyMenuCompat.setCurrentExtension(null)
    })

    it('should only warn once per unique function', () => {
      const methodName = 'getCanvasMenuOptions'
      const warnSpy = vi.spyOn(console, 'warn')

      legacyMenuCompat.install(LGraphCanvas.prototype, methodName)
      legacyMenuCompat.setCurrentExtension('test.extension')

      const patchFunction = function (
        this: LGraphCanvas
      ): (IContextMenuValue | null)[] {
        const items = originalGetCanvasMenuOptions.call(this)
        items.push({ content: 'Custom', callback: () => {} })
        return items
      }

      // Patch twice with same function
      LGraphCanvas.prototype.getCanvasMenuOptions = patchFunction
      LGraphCanvas.prototype.getCanvasMenuOptions = patchFunction

      // Should only warn once
      expect(warnSpy).toHaveBeenCalledTimes(1)
    })
  })

  describe('extractLegacyItems', () => {
    // Cache base items to ensure reference equality for set-based diffing
    const baseItem1 = { content: 'Item 1', callback: () => {} }
    const baseItem2 = { content: 'Item 2', callback: () => {} }

    beforeEach(() => {
      // Setup a mock original method that returns cached items
      // This ensures reference equality when set-based diffing compares items
      LGraphCanvas.prototype.getCanvasMenuOptions = function () {
        return [baseItem1, baseItem2]
      }

      // Install compatibility layer
      legacyMenuCompat.install(LGraphCanvas.prototype, 'getCanvasMenuOptions')
    })

    it('should extract items added by monkey patches', () => {
      // Monkey-patch to add items
      const original = LGraphCanvas.prototype.getCanvasMenuOptions
      LGraphCanvas.prototype.getCanvasMenuOptions =
        function (): (IContextMenuValue | null)[] {
          const items = original.apply(this)
          items.push({ content: 'Custom Item 1', callback: () => {} })
          items.push({ content: 'Custom Item 2', callback: () => {} })
          return items
        }

      // Extract legacy items
      const legacyItems = legacyMenuCompat.extractLegacyItems(
        'getCanvasMenuOptions',
        mockCanvas
      )

      expect(legacyItems).toHaveLength(2)
      expect(legacyItems[0]).toMatchObject({ content: 'Custom Item 1' })
      expect(legacyItems[1]).toMatchObject({ content: 'Custom Item 2' })
    })

    it('should return empty array when no items added', () => {
      // No monkey-patching, so no extra items
      const legacyItems = legacyMenuCompat.extractLegacyItems(
        'getCanvasMenuOptions',
        mockCanvas
      )

      expect(legacyItems).toHaveLength(0)
    })

    it('should detect replaced items as additions and warn about removed items', () => {
      const warnSpy = vi.spyOn(console, 'warn')

      // Monkey-patch that replaces items with different ones (same count)
      // With set-based diffing, these are detected as new items since they're different references
      LGraphCanvas.prototype.getCanvasMenuOptions = function () {
        return [
          { content: 'Replaced 1', callback: () => {} },
          { content: 'Replaced 2', callback: () => {} }
        ]
      }

      const legacyItems = legacyMenuCompat.extractLegacyItems(
        'getCanvasMenuOptions',
        mockCanvas
      )

      // Set-based diffing detects the replaced items as additions
      expect(legacyItems).toHaveLength(2)
      expect(legacyItems[0]).toMatchObject({ content: 'Replaced 1' })
      expect(legacyItems[1]).toMatchObject({ content: 'Replaced 2' })

      // Should warn about removed original items
      expect(warnSpy).toHaveBeenCalledWith(expect.stringContaining('removed'))
    })

    it('should handle errors gracefully', () => {
      const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => {})

      // Monkey-patch that throws error
      LGraphCanvas.prototype.getCanvasMenuOptions = function () {
        throw new Error('Test error')
      }

      const legacyItems = legacyMenuCompat.extractLegacyItems(
        'getCanvasMenuOptions',
        mockCanvas
      )

      expect(legacyItems).toHaveLength(0)
      expect(errorSpy).toHaveBeenCalledWith(
        expect.stringContaining('Failed to extract legacy items'),
        expect.any(Error)
      )
    })

    it('should handle multiple items with undefined content correctly', () => {
      // Setup base method with items that have undefined content
      LGraphCanvas.prototype.getCanvasMenuOptions = function () {
        return [
          { content: undefined, title: 'Separator 1' },
          { content: undefined, title: 'Separator 2' },
          { content: 'Item 1', callback: () => {} }
        ]
      }

      legacyMenuCompat.install(LGraphCanvas.prototype, 'getCanvasMenuOptions')

      // Monkey-patch to add an item with undefined content
      const original = LGraphCanvas.prototype.getCanvasMenuOptions
      LGraphCanvas.prototype.getCanvasMenuOptions =
        function (): (IContextMenuValue | null)[] {
          const items = original.apply(this)
          items.push({ content: undefined, title: 'Separator 3' })
          return items
        }

      // Extract legacy items
      const legacyItems = legacyMenuCompat.extractLegacyItems(
        'getCanvasMenuOptions',
        mockCanvas
      )

      // Should extract only the newly added item with undefined content
      // (not collapse with existing undefined content items)
      expect(legacyItems).toHaveLength(1)
      expect(legacyItems[0]).toMatchObject({
        content: undefined,
        title: 'Separator 3'
      })
    })
  })

  describe('integration', () => {
    // Cache base items to ensure reference equality for set-based diffing
    const integrationBaseItem = { content: 'Base Item', callback: () => {} }
    const integrationBaseItem1 = { content: 'Base Item 1', callback: () => {} }
    const integrationBaseItem2 = { content: 'Base Item 2', callback: () => {} }

    it('should work with multiple extensions patching', () => {
      // Setup base method with cached item
      LGraphCanvas.prototype.getCanvasMenuOptions = function () {
        return [integrationBaseItem]
      }

      legacyMenuCompat.install(LGraphCanvas.prototype, 'getCanvasMenuOptions')

      // First extension patches
      const original1 = LGraphCanvas.prototype.getCanvasMenuOptions
      LGraphCanvas.prototype.getCanvasMenuOptions =
        function (): (IContextMenuValue | null)[] {
          const items = original1.apply(this)
          items.push({ content: 'Extension 1 Item', callback: () => {} })
          return items
        }

      // Second extension patches
      const original2 = LGraphCanvas.prototype.getCanvasMenuOptions
      LGraphCanvas.prototype.getCanvasMenuOptions =
        function (): (IContextMenuValue | null)[] {
          const items = original2.apply(this)
          items.push({ content: 'Extension 2 Item', callback: () => {} })
          return items
        }

      // Extract legacy items
      const legacyItems = legacyMenuCompat.extractLegacyItems(
        'getCanvasMenuOptions',
        mockCanvas
      )

      // Should extract both items added by extensions
      expect(legacyItems).toHaveLength(2)
      expect(legacyItems[0]).toMatchObject({ content: 'Extension 1 Item' })
      expect(legacyItems[1]).toMatchObject({ content: 'Extension 2 Item' })
    })

    it('should extract legacy items only once even when called multiple times', () => {
      // Setup base method with cached items
      LGraphCanvas.prototype.getCanvasMenuOptions = function () {
        return [integrationBaseItem1, integrationBaseItem2]
      }

      legacyMenuCompat.install(LGraphCanvas.prototype, 'getCanvasMenuOptions')

      // Simulate legacy extension monkey-patching the prototype
      const original = LGraphCanvas.prototype.getCanvasMenuOptions
      LGraphCanvas.prototype.getCanvasMenuOptions =
        function (): (IContextMenuValue | null)[] {
          const items = original.apply(this)
          items.push({ content: 'Legacy Item 1', callback: () => {} })
          items.push({ content: 'Legacy Item 2', callback: () => {} })
          return items
        }

      // Extract legacy items multiple times (simulating repeated menu opens)
      const legacyItems1 = legacyMenuCompat.extractLegacyItems(
        'getCanvasMenuOptions',
        mockCanvas
      )
      const legacyItems2 = legacyMenuCompat.extractLegacyItems(
        'getCanvasMenuOptions',
        mockCanvas
      )
      const legacyItems3 = legacyMenuCompat.extractLegacyItems(
        'getCanvasMenuOptions',
        mockCanvas
      )

      // Each extraction should return the same items (no accumulation)
      expect(legacyItems1).toHaveLength(2)
      expect(legacyItems2).toHaveLength(2)
      expect(legacyItems3).toHaveLength(2)

      // Verify items are the expected ones
      expect(legacyItems1[0]).toMatchObject({ content: 'Legacy Item 1' })
      expect(legacyItems1[1]).toMatchObject({ content: 'Legacy Item 2' })

      expect(legacyItems2[0]).toMatchObject({ content: 'Legacy Item 1' })
      expect(legacyItems2[1]).toMatchObject({ content: 'Legacy Item 2' })

      expect(legacyItems3[0]).toMatchObject({ content: 'Legacy Item 1' })
      expect(legacyItems3[1]).toMatchObject({ content: 'Legacy Item 2' })
    })

    it('should not extract items from registered wrapper methods', () => {
      // Setup base method with cached item
      LGraphCanvas.prototype.getCanvasMenuOptions = function () {
        return [integrationBaseItem]
      }

      legacyMenuCompat.install(LGraphCanvas.prototype, 'getCanvasMenuOptions')

      // Create a wrapper that adds new API items (simulating useContextMenuTranslation)
      const originalMethod = LGraphCanvas.prototype.getCanvasMenuOptions
      const wrapperMethod = function (
        this: LGraphCanvas
      ): (IContextMenuValue | null)[] {
        const items = originalMethod.apply(this)
        // Add new API items
        items.push({ content: 'New API Item 1', callback: () => {} })
        items.push({ content: 'New API Item 2', callback: () => {} })
        return items
      }

      // Set the wrapper as the current method
      LGraphCanvas.prototype.getCanvasMenuOptions = wrapperMethod

      // Register the wrapper so it's not treated as a legacy patch
      legacyMenuCompat.registerWrapper(
        'getCanvasMenuOptions',
        wrapperMethod,
        originalMethod,
        LGraphCanvas.prototype // Wrapper is installed
      )

      // Extract legacy items - should return empty because current method is a registered wrapper
      const legacyItems = legacyMenuCompat.extractLegacyItems(
        'getCanvasMenuOptions',
        mockCanvas
      )

      expect(legacyItems).toHaveLength(0)
    })

    it('should extract legacy items even when a wrapper is registered but not active', () => {
      // Setup base method with cached item
      LGraphCanvas.prototype.getCanvasMenuOptions = function () {
        return [integrationBaseItem]
      }

      legacyMenuCompat.install(LGraphCanvas.prototype, 'getCanvasMenuOptions')

      // Register a wrapper (but don't set it as the current method)
      const originalMethod = LGraphCanvas.prototype.getCanvasMenuOptions
      const wrapperMethod = function (): (IContextMenuValue | null)[] {
        return [{ content: 'Wrapper Item', callback: () => {} }]
      }
      legacyMenuCompat.registerWrapper(
        'getCanvasMenuOptions',
        wrapperMethod,
        originalMethod
        // NOT passing prototype, so it won't be marked as installed
      )

      // Monkey-patch with a different function (legacy extension)
      const original = LGraphCanvas.prototype.getCanvasMenuOptions
      LGraphCanvas.prototype.getCanvasMenuOptions =
        function (): (IContextMenuValue | null)[] {
          const items = original.apply(this)
          items.push({ content: 'Legacy Item', callback: () => {} })
          return items
        }

      // Extract legacy items - should return the legacy item because current method is NOT the wrapper
      const legacyItems = legacyMenuCompat.extractLegacyItems(
        'getCanvasMenuOptions',
        mockCanvas
      )

      expect(legacyItems).toHaveLength(1)
      expect(legacyItems[0]).toMatchObject({ content: 'Legacy Item' })
    })
  })
})
