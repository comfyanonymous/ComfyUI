import { beforeEach, describe, expect, test, vi } from 'vitest'

import { LGraphNode } from '@/lib/litegraph/src/litegraph'
import { ComponentWidgetImpl, DOMWidgetImpl } from '@/scripts/domWidget'

const isPromotedByAnyMock = vi.hoisted(() => vi.fn())

// Mock dependencies
vi.mock('@/stores/domWidgetStore', () => ({
  useDomWidgetStore: () => ({
    unregisterWidget: vi.fn()
  })
}))

vi.mock('@/stores/promotionStore', () => ({
  usePromotionStore: () => ({
    isPromotedByAny: isPromotedByAnyMock
  })
}))

vi.mock('@/utils/formatUtil', () => ({
  generateUUID: () => 'test-uuid'
}))

type MockCanvasContext = Pick<
  CanvasRenderingContext2D,
  | 'beginPath'
  | 'rect'
  | 'fill'
  | 'save'
  | 'restore'
  | 'strokeRect'
  | 'fillStyle'
  | 'strokeStyle'
>

function createMockContext(): MockCanvasContext {
  return {
    beginPath: vi.fn(),
    rect: vi.fn(),
    fill: vi.fn(),
    save: vi.fn(),
    restore: vi.fn(),
    strokeRect: vi.fn(),
    fillStyle: '#000',
    strokeStyle: '#000'
  }
}

describe('DOMWidget Y Position Preservation', () => {
  test('BaseDOMWidgetImpl createCopyForNode preserves Y position', () => {
    const mockNode = new LGraphNode('test-node')
    const originalWidget = new ComponentWidgetImpl({
      node: mockNode,
      name: 'test-widget',
      component: { template: '<div></div>' },
      inputSpec: { name: 'test', type: 'string' },
      options: {}
    })

    // Set a specific Y position
    originalWidget.y = 66

    const newNode = new LGraphNode('new-node')
    const clonedWidget = originalWidget.createCopyForNode(newNode)

    // Verify Y position is preserved
    expect(clonedWidget.y).toBe(66)
    expect(clonedWidget.node).toBe(newNode)
    expect(clonedWidget.name).toBe('test-widget')
  })

  test('DOMWidgetImpl createCopyForNode preserves Y position', () => {
    const mockNode = new LGraphNode('test-node')
    const mockElement = document.createElement('div')

    const originalWidget = new DOMWidgetImpl({
      node: mockNode,
      name: 'test-dom-widget',
      type: 'test',
      element: mockElement,
      options: {}
    })

    // Set a specific Y position
    originalWidget.y = 42

    const newNode = new LGraphNode('new-node')
    const clonedWidget = originalWidget.createCopyForNode(newNode)

    // Verify Y position is preserved
    expect(clonedWidget.y).toBe(42)
    expect(clonedWidget.node).toBe(newNode)
    expect(clonedWidget.element).toBe(mockElement)
    expect(clonedWidget.name).toBe('test-dom-widget')
  })

  test('Y position defaults to 0 when not set', () => {
    const mockNode = new LGraphNode('test-node')
    const originalWidget = new ComponentWidgetImpl({
      node: mockNode,
      name: 'test-widget',
      component: { template: '<div></div>' },
      inputSpec: { name: 'test', type: 'string' },
      options: {}
    })

    // Don't explicitly set Y (should be 0 by default)
    const newNode = new LGraphNode('new-node')
    const clonedWidget = originalWidget.createCopyForNode(newNode)

    // Verify Y position is preserved (should be 0)
    expect(clonedWidget.y).toBe(0)
  })
})

describe('DOMWidget draw promotion behavior', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  test('draws promoted outline for visible promoted widgets', () => {
    isPromotedByAnyMock.mockReturnValue(true)

    const node = new LGraphNode('test-node')
    const rootGraph = { id: 'root-graph-id' }
    node.graph = { rootGraph } as never
    const onDraw = vi.fn()

    const widget = new DOMWidgetImpl({
      node,
      name: 'seed',
      type: 'text',
      element: document.createElement('div'),
      options: { onDraw }
    })
    const ctx = createMockContext()

    widget.draw(ctx as CanvasRenderingContext2D, node, 200, 30, 40)

    expect(isPromotedByAnyMock).toHaveBeenCalledWith(
      'root-graph-id',
      '-1',
      'seed'
    )
    expect(ctx.strokeRect).toHaveBeenCalledOnce()
    expect(onDraw).toHaveBeenCalledWith(widget)
  })

  test('does not draw promoted outline when widget is not promoted', () => {
    isPromotedByAnyMock.mockReturnValue(false)

    const node = new LGraphNode('test-node')
    const rootGraph = { id: 'root-graph-id' }
    node.graph = { rootGraph } as never
    const onDraw = vi.fn()

    const widget = new DOMWidgetImpl({
      node,
      name: 'seed',
      type: 'text',
      element: document.createElement('div'),
      options: { onDraw }
    })
    const ctx = createMockContext()

    widget.draw(ctx as CanvasRenderingContext2D, node, 200, 30, 40)

    expect(ctx.strokeRect).not.toHaveBeenCalled()
    expect(onDraw).toHaveBeenCalledWith(widget)
  })

  test('skips promotion lookup when widget is hidden', () => {
    const node = new LGraphNode('test-node')
    const rootGraph = { id: 'root-graph-id' }
    node.graph = { rootGraph } as never
    const onDraw = vi.fn()

    const widget = new DOMWidgetImpl({
      node,
      name: 'seed',
      type: 'text',
      element: document.createElement('div'),
      options: { onDraw }
    })
    widget.hidden = true
    const ctx = createMockContext()

    widget.draw(ctx as CanvasRenderingContext2D, node, 200, 30, 40)

    expect(isPromotedByAnyMock).not.toHaveBeenCalled()
    expect(ctx.strokeRect).not.toHaveBeenCalled()
    expect(onDraw).toHaveBeenCalledWith(widget)
  })
})
