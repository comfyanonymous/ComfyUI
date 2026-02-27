import { describe, expect, it, vi } from 'vitest'

import { legacyMenuCompat } from '@/lib/litegraph/src/contextMenuCompat'
import type {
  IContextMenuValue,
  LGraphNode
} from '@/lib/litegraph/src/litegraph'
import { LGraphCanvas } from '@/lib/litegraph/src/litegraph'

/**
 * Test that demonstrates the extension name appearing in deprecation warnings
 */
describe('Context Menu Extension Name in Warnings', () => {
  it('should include extension name in deprecation warning', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})

    // Install compatibility layer
    legacyMenuCompat.install(LGraphCanvas.prototype, 'getCanvasMenuOptions')

    // Simulate what happens during extension setup
    legacyMenuCompat.setCurrentExtension('MyCustomExtension')

    // Extension monkey-patches the method
    const original = LGraphCanvas.prototype.getCanvasMenuOptions
    LGraphCanvas.prototype.getCanvasMenuOptions =
      function (): (IContextMenuValue | null)[] {
        const items = original.call(this)
        items.push({ content: 'My Custom Menu Item', callback: () => {} })
        return items
      }

    // Clear extension (happens after setup completes)
    legacyMenuCompat.setCurrentExtension(null)

    // Verify the warning includes the extension name
    expect(warnSpy).toHaveBeenCalled()
    const warningMessage = warnSpy.mock.calls[0][0]

    expect(warningMessage).toContain('[DEPRECATED]')
    expect(warningMessage).toContain('getCanvasMenuOptions')
    expect(warningMessage).toContain('"MyCustomExtension"')

    vi.restoreAllMocks()
  })

  it('should include extension name for node menu patches', () => {
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})

    // Install compatibility layer
    legacyMenuCompat.install(LGraphCanvas.prototype, 'getNodeMenuOptions')

    // Simulate what happens during extension setup
    legacyMenuCompat.setCurrentExtension('AnotherExtension')

    // Extension monkey-patches the method
    const original = LGraphCanvas.prototype.getNodeMenuOptions
    LGraphCanvas.prototype.getNodeMenuOptions = function (node: LGraphNode) {
      const items = original.call(this, node)
      items.push({ content: 'My Node Menu Item', callback: () => {} })
      return items
    }

    // Clear extension (happens after setup completes)
    legacyMenuCompat.setCurrentExtension(null)

    // Verify the warning includes extension info
    expect(warnSpy).toHaveBeenCalled()
    const warningMessage = warnSpy.mock.calls[0][0]

    expect(warningMessage).toContain('[DEPRECATED]')
    expect(warningMessage).toContain('getNodeMenuOptions')
    expect(warningMessage).toContain('"AnotherExtension"')

    vi.restoreAllMocks()
  })
})
