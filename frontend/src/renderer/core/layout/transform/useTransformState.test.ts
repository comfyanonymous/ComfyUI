import { beforeEach, describe, expect, it } from 'vitest'

import { useTransformState } from '@/renderer/core/layout/transform/useTransformState'
import type { LGraphCanvas } from '@/lib/litegraph/src/LGraphCanvas'

// Create a mock canvas context for transform testing
function createMockCanvasContext() {
  return {
    canvas: {
      width: 1280,
      height: 720,
      getBoundingClientRect: () => ({
        left: 0,
        top: 0,
        width: 1280,
        height: 720,
        right: 1280,
        bottom: 720,
        x: 0,
        y: 0
      })
    },
    ds: {
      offset: [0, 0],
      scale: 1
    }
  }
}

describe('useTransformState', () => {
  const transformState = useTransformState()

  beforeEach(() => {
    transformState.syncWithCanvas({
      ds: { offset: [0, 0] }
    } as LGraphCanvas)
  })

  describe('initial state', () => {
    it('should initialize with default camera values', () => {
      const { camera } = transformState
      expect(camera.x).toBe(0)
      expect(camera.y).toBe(0)
      expect(camera.z).toBe(1)
    })

    it('should generate correct initial transform style', () => {
      const { transformStyle } = transformState
      expect(transformStyle.value).toEqual({
        transform: 'scale(1) translate(0px, 0px)',
        transformOrigin: '0 0'
      })
    })
  })

  describe('syncWithCanvas', () => {
    it('should sync camera state with canvas transform', () => {
      const { syncWithCanvas, camera } = transformState
      const mockCanvas = createMockCanvasContext()

      // Set mock canvas transform
      mockCanvas.ds.offset = [100, 50]
      mockCanvas.ds.scale = 2

      syncWithCanvas(mockCanvas as LGraphCanvas)

      expect(camera.x).toBe(100)
      expect(camera.y).toBe(50)
      expect(camera.z).toBe(2)
    })

    it('should handle null canvas gracefully', () => {
      const { syncWithCanvas, camera } = transformState

      syncWithCanvas(null! as LGraphCanvas)

      // Should remain at initial values
      expect(camera.x).toBe(0)
      expect(camera.y).toBe(0)
      expect(camera.z).toBe(1)
    })

    it('should handle canvas without ds property', () => {
      const { syncWithCanvas, camera } = transformState
      const canvasWithoutDs = { canvas: {} }

      syncWithCanvas(canvasWithoutDs as LGraphCanvas)

      // Should remain at initial values
      expect(camera.x).toBe(0)
      expect(camera.y).toBe(0)
      expect(camera.z).toBe(1)
    })

    it('should update transform style after sync', () => {
      const { syncWithCanvas, transformStyle } = transformState
      const mockCanvas = createMockCanvasContext()

      mockCanvas.ds.offset = [150, 75]
      mockCanvas.ds.scale = 0.5

      syncWithCanvas(mockCanvas as LGraphCanvas)

      expect(transformStyle.value).toEqual({
        transform: 'scale(0.5) translate(150px, 75px)',
        transformOrigin: '0 0'
      })
    })
  })

  describe('coordinate conversions', () => {
    beforeEach(() => {
      // Set up a known transform state
      const mockCanvas = createMockCanvasContext()
      mockCanvas.ds.offset = [100, 50]
      mockCanvas.ds.scale = 2
      transformState.syncWithCanvas(mockCanvas as LGraphCanvas)
    })

    describe('canvasToScreen', () => {
      it('should convert canvas coordinates to screen coordinates', () => {
        const { canvasToScreen } = transformState

        const canvasPoint = { x: 10, y: 20 }
        const screenPoint = canvasToScreen(canvasPoint)

        // screen = (canvas + offset) * scale
        // x: (10 + 100) * 2 = 220
        // y: (20 + 50) * 2 = 140
        expect(screenPoint).toEqual({ x: 220, y: 140 })
      })

      it('should handle zero coordinates', () => {
        const { canvasToScreen } = transformState

        const screenPoint = canvasToScreen({ x: 0, y: 0 })
        expect(screenPoint).toEqual({ x: 200, y: 100 })
      })

      it('should handle negative coordinates', () => {
        const { canvasToScreen } = transformState

        const screenPoint = canvasToScreen({ x: -10, y: -20 })
        expect(screenPoint).toEqual({ x: 180, y: 60 })
      })
    })

    describe('screenToCanvas', () => {
      it('should convert screen coordinates to canvas coordinates', () => {
        const { screenToCanvas } = transformState

        const screenPoint = { x: 220, y: 140 }
        const canvasPoint = screenToCanvas(screenPoint)

        // canvas = screen / scale - offset
        // x: 220 / 2 - 100 = 10
        // y: 140 / 2 - 50 = 20
        expect(canvasPoint).toEqual({ x: 10, y: 20 })
      })

      it('should be inverse of canvasToScreen', () => {
        const { canvasToScreen, screenToCanvas } = transformState

        const originalPoint = { x: 25, y: 35 }
        const screenPoint = canvasToScreen(originalPoint)
        const backToCanvas = screenToCanvas(screenPoint)

        expect(backToCanvas.x).toBeCloseTo(originalPoint.x)
        expect(backToCanvas.y).toBeCloseTo(originalPoint.y)
      })
    })
  })

  describe('getNodeScreenBounds', () => {
    beforeEach(() => {
      const mockCanvas = createMockCanvasContext()
      mockCanvas.ds.offset = [100, 50]
      mockCanvas.ds.scale = 2
      transformState.syncWithCanvas(mockCanvas as LGraphCanvas)
    })

    it('should calculate correct screen bounds for a node', () => {
      const { getNodeScreenBounds } = transformState

      const nodePos: [number, number] = [10, 20]
      const nodeSize: [number, number] = [200, 100]
      const bounds = getNodeScreenBounds(nodePos, nodeSize)

      // Top-left: canvasToScreen(10, 20) = (220, 140)
      // Width: 200 * 2 = 400
      // Height: 100 * 2 = 200
      expect(bounds.x).toBe(220)
      expect(bounds.y).toBe(140)
      expect(bounds.width).toBe(400)
      expect(bounds.height).toBe(200)
    })
  })

  describe('isNodeInViewport', () => {
    beforeEach(() => {
      const mockCanvas = createMockCanvasContext()
      mockCanvas.ds.offset = [0, 0]
      mockCanvas.ds.scale = 1
      transformState.syncWithCanvas(mockCanvas as LGraphCanvas)
    })

    const viewport = { width: 1000, height: 600 }

    it('should return true for nodes inside viewport', () => {
      const { isNodeInViewport } = transformState

      const nodePos: [number, number] = [100, 100]
      const nodeSize: [number, number] = [200, 100]

      expect(isNodeInViewport(nodePos, nodeSize, viewport)).toBe(true)
    })

    it('should return false for nodes completely outside viewport', () => {
      const { isNodeInViewport } = transformState

      // Node far to the right
      expect(isNodeInViewport([2000, 100], [200, 100], viewport)).toBe(false)

      // Node far to the left
      expect(isNodeInViewport([-500, 100], [200, 100], viewport)).toBe(false)

      // Node far below
      expect(isNodeInViewport([100, 1000], [200, 100], viewport)).toBe(false)

      // Node far above
      expect(isNodeInViewport([100, -500], [200, 100], viewport)).toBe(false)
    })

    it('should return true for nodes partially in viewport with margin', () => {
      const { isNodeInViewport } = transformState

      // Node slightly outside but within margin
      const nodePos: [number, number] = [-50, -50]
      const nodeSize: [number, number] = [100, 100]

      expect(isNodeInViewport(nodePos, nodeSize, viewport, 0.2)).toBe(true)
    })

    it('should return false for tiny nodes (size culling)', () => {
      const { isNodeInViewport } = transformState

      // Node is in viewport but too small
      const nodePos: [number, number] = [100, 100]
      const nodeSize: [number, number] = [3, 3] // Less than 4 pixels

      expect(isNodeInViewport(nodePos, nodeSize, viewport)).toBe(false)
    })

    it('should adjust margin based on zoom level', () => {
      const { isNodeInViewport, syncWithCanvas } = transformState
      const mockCanvas = createMockCanvasContext()

      // Test with very low zoom
      mockCanvas.ds.scale = 0.05
      syncWithCanvas(mockCanvas as LGraphCanvas)

      // Node at edge should still be visible due to increased margin
      expect(isNodeInViewport([1100, 100], [200, 100], viewport)).toBe(true)

      // Test with high zoom
      mockCanvas.ds.scale = 4
      syncWithCanvas(mockCanvas as LGraphCanvas)

      // Margin should be tighter
      expect(isNodeInViewport([1100, 100], [200, 100], viewport)).toBe(false)
    })
  })

  describe('getViewportBounds', () => {
    beforeEach(() => {
      const mockCanvas = createMockCanvasContext()
      mockCanvas.ds.offset = [100, 50]
      mockCanvas.ds.scale = 2
      transformState.syncWithCanvas(mockCanvas as LGraphCanvas)
    })

    it('should calculate viewport bounds in canvas coordinates', () => {
      const { getViewportBounds } = transformState
      const viewport = { width: 1000, height: 600 }

      const bounds = getViewportBounds(viewport, 0.2)

      // With 20% margin:
      // marginX = 1000 * 0.2 = 200
      // marginY = 600 * 0.2 = 120
      // topLeft in screen: (-200, -120)
      // bottomRight in screen: (1200, 720)

      // Convert to canvas coordinates (canvas = screen / scale - offset):
      // topLeft: (-200 / 2 - 100, -120 / 2 - 50) = (-200, -110)
      // bottomRight: (1200 / 2 - 100, 720 / 2 - 50) = (500, 310)

      expect(bounds.x).toBe(-200)
      expect(bounds.y).toBe(-110)
      expect(bounds.width).toBe(700) // 500 - (-200)
      expect(bounds.height).toBe(420) // 310 - (-110)
    })

    it('should handle zero margin', () => {
      const { getViewportBounds } = transformState
      const viewport = { width: 1000, height: 600 }

      const bounds = getViewportBounds(viewport, 0)

      // No margin, so viewport bounds are exact
      expect(bounds.x).toBe(-100) // 0 / 2 - 100
      expect(bounds.y).toBe(-50) // 0 / 2 - 50
      expect(bounds.width).toBe(500) // 1000 / 2
      expect(bounds.height).toBe(300) // 600 / 2
    })
  })

  describe('edge cases', () => {
    it('should handle extreme zoom levels', () => {
      const { syncWithCanvas, canvasToScreen } = transformState
      const mockCanvas = createMockCanvasContext()

      // Very small zoom
      mockCanvas.ds.scale = 0.001
      syncWithCanvas(mockCanvas as LGraphCanvas)

      const point1 = canvasToScreen({ x: 1000, y: 1000 })
      expect(point1.x).toBeCloseTo(1)
      expect(point1.y).toBeCloseTo(1)

      // Very large zoom
      mockCanvas.ds.scale = 100
      syncWithCanvas(mockCanvas as LGraphCanvas)

      const point2 = canvasToScreen({ x: 1, y: 1 })
      expect(point2.x).toBe(100)
      expect(point2.y).toBe(100)
    })

    it('should handle zero scale in screenToCanvas', () => {
      const { syncWithCanvas, screenToCanvas } = transformState
      const mockCanvas = createMockCanvasContext()

      // Scale of 0 gets converted to 1 by || operator
      mockCanvas.ds.scale = 0
      syncWithCanvas(mockCanvas as LGraphCanvas)

      // Should use scale of 1 due to camera.z || 1 in implementation
      const result = screenToCanvas({ x: 100, y: 100 })
      expect(result.x).toBe(100) // (100 - 0) / 1
      expect(result.y).toBe(100) // (100 - 0) / 1
    })
  })
})
