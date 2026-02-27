import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import {
  LGraph,
  LGraphCanvas,
  LGraphNode,
  LiteGraph
} from '@/lib/litegraph/src/litegraph'
import { layoutStore } from '@/renderer/core/layout/store/layoutStore'

vi.mock('@/renderer/core/layout/store/layoutStore', () => ({
  layoutStore: {
    querySlotAtPoint: vi.fn(),
    queryRerouteAtPoint: vi.fn(),
    getNodeLayoutRef: vi.fn(() => ({ value: null })),
    getSlotLayout: vi.fn(),
    setSource: vi.fn(),
    batchUpdateNodeBounds: vi.fn()
  }
}))

describe('LGraphCanvas slot hit detection', () => {
  let graph: LGraph
  let canvas: LGraphCanvas
  let node: LGraphNode
  let canvasElement: HTMLCanvasElement

  beforeEach(() => {
    vi.clearAllMocks()

    canvasElement = document.createElement('canvas')
    canvasElement.width = 800
    canvasElement.height = 600

    const ctx = {
      save: vi.fn(),
      restore: vi.fn(),
      translate: vi.fn(),
      scale: vi.fn(),
      fillRect: vi.fn(),
      strokeRect: vi.fn(),
      fillText: vi.fn(),
      measureText: vi.fn().mockReturnValue({ width: 50 }),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      fill: vi.fn(),
      closePath: vi.fn(),
      arc: vi.fn(),
      rect: vi.fn(),
      clip: vi.fn(),
      clearRect: vi.fn(),
      setTransform: vi.fn(),
      roundRect: vi.fn(),
      getTransform: vi
        .fn()
        .mockReturnValue({ a: 1, b: 0, c: 0, d: 1, e: 0, f: 0 }),
      font: '',
      fillStyle: '',
      strokeStyle: '',
      lineWidth: 1,
      globalAlpha: 1,
      textAlign: 'left' as CanvasTextAlign,
      textBaseline: 'alphabetic' as CanvasTextBaseline
    } as Partial<CanvasRenderingContext2D> as CanvasRenderingContext2D

    canvasElement.getContext = vi.fn().mockReturnValue(ctx)
    canvasElement.getBoundingClientRect = vi.fn().mockReturnValue({
      left: 0,
      top: 0,
      width: 800,
      height: 600
    })

    graph = new LGraph()
    canvas = new LGraphCanvas(canvasElement, graph, {
      skip_render: true
    })

    // Create a test node with an output slot
    node = new LGraphNode('Test Node')
    node.pos = [100, 100]
    node.size = [150, 80]
    node.addOutput('output', 'number')
    graph.add(node)

    // Enable Vue nodes mode for the test
    LiteGraph.vueNodesMode = true
  })

  afterEach(() => {
    LiteGraph.vueNodesMode = false
  })

  describe('processMouseDown slot fallback in Vue nodes mode', () => {
    it('should query layoutStore.querySlotAtPoint when clicking outside node bounds', () => {
      // Click position outside node bounds (node is at 100,100 with size 150x80)
      // So node covers x: 100-250, y: 100-180
      // Click at x=255 is outside the right edge
      const clickX = 255
      const clickY = 120

      // Verify the click is outside the node bounds
      expect(node.isPointInside(clickX, clickY)).toBe(false)
      expect(graph.getNodeOnPos(clickX, clickY)).toBeNull()

      // Mock the slot query to return our node's slot
      vi.mocked(layoutStore.querySlotAtPoint).mockReturnValue({
        nodeId: String(node.id),
        index: 0,
        type: 'output',
        position: { x: 252, y: 120 },
        bounds: { x: 246, y: 110, width: 20, height: 20 }
      })

      // Call processMouseDown - this should trigger the slot fallback
      canvas.processMouseDown(
        new MouseEvent('pointerdown', {
          button: 1, // Middle button
          clientX: clickX,
          clientY: clickY
        })
      )

      // The fix should query the layout store when no node is found at click position
      expect(layoutStore.querySlotAtPoint).toHaveBeenCalledWith({
        x: clickX,
        y: clickY
      })
    })

    it('should NOT query layoutStore when node is found directly at click position', () => {
      // Initialize node's bounding rect
      node.updateArea()

      // Populate visible_nodes (normally done during render)
      canvas.visible_nodes = [node]

      // Click inside the node bounds
      const clickX = 150
      const clickY = 140

      // Verify the click is inside the node bounds
      expect(node.isPointInside(clickX, clickY)).toBe(true)
      expect(graph.getNodeOnPos(clickX, clickY)).toBe(node)

      // Call processMouseDown
      canvas.processMouseDown(
        new MouseEvent('pointerdown', {
          button: 1,
          clientX: clickX,
          clientY: clickY
        })
      )

      // Should NOT query the layout store since node was found directly
      expect(layoutStore.querySlotAtPoint).not.toHaveBeenCalled()
    })

    it('should NOT query layoutStore when not in Vue nodes mode', () => {
      LiteGraph.vueNodesMode = false

      const clickX = 255
      const clickY = 120

      // Call processMouseDown
      canvas.processMouseDown(
        new MouseEvent('pointerdown', {
          button: 1,
          clientX: clickX,
          clientY: clickY
        })
      )

      // Should NOT query the layout store in non-Vue mode
      expect(layoutStore.querySlotAtPoint).not.toHaveBeenCalled()
    })

    it('should find node via slot query for input slots extending beyond left edge', () => {
      node.addInput('input', 'number')

      // Click position left of node (node starts at x=100)
      const clickX = 95
      const clickY = 140

      // Verify outside bounds
      expect(node.isPointInside(clickX, clickY)).toBe(false)

      vi.mocked(layoutStore.querySlotAtPoint).mockReturnValue({
        nodeId: String(node.id),
        index: 0,
        type: 'input',
        position: { x: 98, y: 140 },
        bounds: { x: 88, y: 130, width: 20, height: 20 }
      })

      canvas.processMouseDown(
        new MouseEvent('pointerdown', {
          button: 1,
          clientX: clickX,
          clientY: clickY
        })
      )

      expect(layoutStore.querySlotAtPoint).toHaveBeenCalledWith({
        x: clickX,
        y: clickY
      })
    })
  })
})
