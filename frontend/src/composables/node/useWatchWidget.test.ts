import { describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

import { useComputedWithWidgetWatch } from '@/composables/node/useWatchWidget'
import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import { createMockLGraphNode } from '@/utils/__tests__/litegraphTestUtils'

// Mock useChainCallback
vi.mock('@/composables/functional/useChainCallback', () => ({
  useChainCallback: vi.fn((original, newCallback) => {
    return function (this: unknown, ...args: unknown[]) {
      original?.call(this, ...args)
      newCallback.call(this, ...args)
    }
  })
}))

describe('useComputedWithWidgetWatch', () => {
  const createMockNode = (
    widgets: Array<{
      name: string
      value: unknown
      callback?: (...args: unknown[]) => void
    }> = []
  ): LGraphNode => {
    const baseNode = createMockLGraphNode()
    return Object.assign(baseNode, {
      widgets: widgets.map((widget) => ({
        name: widget.name,
        value: widget.value,
        callback: widget.callback
      })),
      graph: {
        setDirtyCanvas: vi.fn()
      }
    })
  }

  it('should create a reactive computed that responds to widget changes', async () => {
    const mockNode = createMockNode([
      { name: 'width', value: 100 },
      { name: 'height', value: 200 }
    ])

    const computedWithWidgetWatch = useComputedWithWidgetWatch(mockNode)

    const computedValue = computedWithWidgetWatch(() => {
      const width =
        (mockNode.widgets?.find((w) => w.name === 'width')?.value as number) ||
        0
      const height =
        (mockNode.widgets?.find((w) => w.name === 'height')?.value as number) ||
        0
      return width * height
    })

    // Initial value should be computed correctly
    expect(computedValue.value).toBe(20000)

    // Change widget value and trigger callback
    const widthWidget = mockNode.widgets?.find((w) => w.name === 'width')
    if (widthWidget && widthWidget.callback) {
      widthWidget.value = 150
      widthWidget.callback(widthWidget.value)
    }

    await nextTick()

    // Computed should update
    expect(computedValue.value).toBe(30000)
  })

  it('should only observe specific widgets when widgetNames is provided', async () => {
    const mockNode = createMockNode([
      { name: 'width', value: 100 },
      { name: 'height', value: 200 },
      { name: 'depth', value: 50 }
    ])

    const computedWithWidgetWatch = useComputedWithWidgetWatch(mockNode, {
      widgetNames: ['width', 'height']
    })

    const computedValue = computedWithWidgetWatch(() => {
      return mockNode.widgets?.map((w) => w.value).join('-') || ''
    })

    expect(computedValue.value).toBe('100-200-50')

    // Change observed widget
    const widthWidget = mockNode.widgets?.find((w) => w.name === 'width')
    if (widthWidget && widthWidget.callback) {
      widthWidget.value = 150
      widthWidget.callback(widthWidget.value)
    }

    await nextTick()
    expect(computedValue.value).toBe('150-200-50')

    // Change non-observed widget - should not trigger update
    const depthWidget = mockNode.widgets?.find((w) => w.name === 'depth')
    if (depthWidget) {
      depthWidget.value = 75
      // Depth widget callback should not have been modified
      expect(depthWidget.callback).toBeUndefined()
    }
  })

  it('should trigger canvas redraw when triggerCanvasRedraw is true', async () => {
    const mockNode = createMockNode([{ name: 'value', value: 10 }])

    const computedWithWidgetWatch = useComputedWithWidgetWatch(mockNode, {
      triggerCanvasRedraw: true
    })

    computedWithWidgetWatch(() => mockNode.widgets?.[0]?.value || 0)

    // Change widget value
    const widget = mockNode.widgets?.[0]
    if (widget && widget.callback) {
      widget.value = 20
      widget.callback(widget.value)
    }

    await nextTick()

    // Canvas redraw should have been triggered
    expect(mockNode.graph?.setDirtyCanvas).toHaveBeenCalledWith(true, true)
  })

  it('should not trigger canvas redraw when triggerCanvasRedraw is false', async () => {
    const mockNode = createMockNode([{ name: 'value', value: 10 }])

    const computedWithWidgetWatch = useComputedWithWidgetWatch(mockNode, {
      triggerCanvasRedraw: false
    })

    computedWithWidgetWatch(() => mockNode.widgets?.[0]?.value || 0)

    // Change widget value
    const widget = mockNode.widgets?.[0]
    if (widget && widget.callback) {
      widget.value = 20
      widget.callback(widget.value)
    }

    await nextTick()

    // Canvas redraw should not have been triggered
    expect(mockNode.graph?.setDirtyCanvas).not.toHaveBeenCalled()
  })

  it('should handle nodes without widgets gracefully', () => {
    const mockNode = createMockNode([])

    const computedWithWidgetWatch = useComputedWithWidgetWatch(mockNode)

    const computedValue = computedWithWidgetWatch(() => 'no widgets')

    expect(computedValue.value).toBe('no widgets')
  })

  it('should chain with existing widget callbacks', async () => {
    const existingCallback = vi.fn()
    const mockNode = createMockNode([
      { name: 'value', value: 10, callback: existingCallback }
    ])

    const computedWithWidgetWatch = useComputedWithWidgetWatch(mockNode)
    computedWithWidgetWatch(() => mockNode.widgets?.[0]?.value || 0)

    // Trigger widget callback
    const widget = mockNode.widgets?.[0]
    if (widget && widget.callback) {
      widget.callback(widget.value)
    }

    await nextTick()

    // Both existing callback and our callback should have been called
    expect(existingCallback).toHaveBeenCalled()
  })
})
