import { useRafFn } from '@vueuse/core'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'
import type { Ref } from 'vue'

import type { LGraph } from '@/lib/litegraph/src/litegraph'
import {
  calculateMinimapScale,
  calculateNodeBounds,
  enforceMinimumBounds
} from '@/renderer/core/spatial/boundsCalculator'
import { useMinimapViewport } from '@/renderer/extensions/minimap/composables/useMinimapViewport'
import type { MinimapCanvas } from '@/renderer/extensions/minimap/types'

vi.mock('@vueuse/core')
vi.mock('@/renderer/core/spatial/boundsCalculator', () => ({
  calculateNodeBounds: vi.fn(),
  calculateMinimapScale: vi.fn(),
  enforceMinimumBounds: vi.fn()
}))

describe('useMinimapViewport', () => {
  let mockCanvas: MinimapCanvas
  let mockGraph: LGraph

  beforeEach(() => {
    vi.clearAllMocks()

    mockCanvas = {
      canvas: {
        clientWidth: 800,
        clientHeight: 600,
        width: 1600,
        height: 1200
      } as HTMLCanvasElement,
      ds: {
        scale: 1,
        offset: [0, 0]
      },
      setDirty: vi.fn()
    }

    mockGraph = {
      _nodes: [
        { pos: [100, 100], size: [150, 80] },
        { pos: [300, 200], size: [120, 60] }
      ]
    } as Partial<LGraph> as LGraph

    vi.mocked(useRafFn, { partial: true }).mockReturnValue({
      resume: vi.fn(),
      pause: vi.fn()
    })
  })

  it('should initialize with default bounds', () => {
    const canvasRef = ref(mockCanvas) as Ref<MinimapCanvas | null>
    const graphRef = ref(mockGraph) as Ref<LGraph | null>

    const viewport = useMinimapViewport(canvasRef, graphRef, 250, 200)

    expect(viewport.bounds.value).toEqual({
      minX: 0,
      minY: 0,
      maxX: 0,
      maxY: 0,
      width: 0,
      height: 0
    })

    expect(viewport.scale.value).toBe(1)
  })

  it('should calculate graph bounds from nodes', () => {
    vi.mocked(calculateNodeBounds).mockReturnValue({
      minX: 100,
      minY: 100,
      maxX: 420,
      maxY: 260,
      width: 320,
      height: 160
    })

    vi.mocked(enforceMinimumBounds).mockImplementation((bounds) => bounds)

    const canvasRef = ref(mockCanvas) as Ref<MinimapCanvas | null>
    const graphRef = ref(mockGraph) as Ref<LGraph | null>

    const viewport = useMinimapViewport(canvasRef, graphRef, 250, 200)

    viewport.updateBounds()

    expect(calculateNodeBounds).toHaveBeenCalledWith(mockGraph._nodes)
    expect(enforceMinimumBounds).toHaveBeenCalled()
  })

  it('should handle empty graph', () => {
    vi.mocked(calculateNodeBounds).mockReturnValue(null)

    const canvasRef = ref(mockCanvas) as Ref<MinimapCanvas | null>
    const graphRef = ref({
      _nodes: []
    } as Partial<LGraph> as LGraph) as Ref<LGraph | null>

    const viewport = useMinimapViewport(canvasRef, graphRef, 250, 200)

    viewport.updateBounds()

    expect(viewport.bounds.value).toEqual({
      minX: 0,
      minY: 0,
      maxX: 100,
      maxY: 100,
      width: 100,
      height: 100
    })
  })

  it('should update canvas dimensions', () => {
    const canvasRef = ref(mockCanvas) as Ref<MinimapCanvas | null>
    const graphRef = ref(mockGraph) as Ref<LGraph | null>

    const viewport = useMinimapViewport(canvasRef, graphRef, 250, 200)

    viewport.updateCanvasDimensions()

    expect(viewport.canvasDimensions.value).toEqual({
      width: 800,
      height: 600
    })
  })

  it('should calculate viewport transform', () => {
    vi.mocked(calculateNodeBounds).mockReturnValue({
      minX: 0,
      minY: 0,
      maxX: 500,
      maxY: 400,
      width: 500,
      height: 400
    })

    vi.mocked(enforceMinimumBounds).mockImplementation((bounds) => bounds)
    vi.mocked(calculateMinimapScale).mockReturnValue(0.5)

    const canvasRef = ref(mockCanvas) as Ref<MinimapCanvas | null>
    const graphRef = ref(mockGraph) as Ref<LGraph | null>

    const viewport = useMinimapViewport(canvasRef, graphRef, 250, 200)

    // Set canvas transform
    mockCanvas.ds.scale = 2
    mockCanvas.ds.offset = [-100, -50]

    // Update bounds and viewport
    viewport.updateBounds()
    viewport.updateCanvasDimensions()
    viewport.updateViewport()

    const transform = viewport.viewportTransform.value

    // World coordinates
    const worldX = -(-100) // -offset[0] = 100
    const worldY = -(-50) // -offset[1] = 50

    // Viewport size in world coordinates
    const viewportWidth = 800 / 2 // canvasWidth / scale = 400
    const viewportHeight = 600 / 2 // canvasHeight / scale = 300

    // Center offsets
    const centerOffsetX = (250 - 500 * 0.5) / 2 // (250 - 250) / 2 = 0
    const centerOffsetY = (200 - 400 * 0.5) / 2 // (200 - 200) / 2 = 0

    // Expected values based on implementation: (worldX - bounds.minX) * scale + centerOffsetX
    expect(transform.x).toBeCloseTo((worldX - 0) * 0.5 + centerOffsetX) // (100 - 0) * 0.5 + 0 = 50
    expect(transform.y).toBeCloseTo((worldY - 0) * 0.5 + centerOffsetY) // (50 - 0) * 0.5 + 0 = 25
    expect(transform.width).toBeCloseTo(viewportWidth * 0.5) // 400 * 0.5 = 200
    expect(transform.height).toBeCloseTo(viewportHeight * 0.5) // 300 * 0.5 = 150
  })

  it('should center view on world coordinates', () => {
    const canvasRef = ref(mockCanvas) as Ref<MinimapCanvas | null>
    const graphRef = ref(mockGraph) as Ref<LGraph | null>

    const viewport = useMinimapViewport(canvasRef, graphRef, 250, 200)

    viewport.updateCanvasDimensions()
    mockCanvas.ds.scale = 2

    viewport.centerViewOn(300, 200)

    // Should update canvas offset to center on the given world coordinates
    const expectedOffsetX = -(300 - 800 / 2 / 2) // -(worldX - viewportWidth/2)
    const expectedOffsetY = -(200 - 600 / 2 / 2) // -(worldY - viewportHeight/2)

    expect(mockCanvas.ds.offset[0]).toBe(expectedOffsetX)
    expect(mockCanvas.ds.offset[1]).toBe(expectedOffsetY)
    expect(mockCanvas.setDirty).toHaveBeenCalledWith(true, true)
  })

  it('should start and stop viewport sync', () => {
    const startSyncMock = vi.fn()
    const stopSyncMock = vi.fn()

    vi.mocked(useRafFn, { partial: true }).mockReturnValue({
      resume: startSyncMock,
      pause: stopSyncMock
    })

    const canvasRef = ref(mockCanvas) as Ref<MinimapCanvas | null>
    const graphRef = ref(mockGraph) as Ref<LGraph | null>

    const viewport = useMinimapViewport(canvasRef, graphRef, 250, 200)

    viewport.startViewportSync()
    expect(startSyncMock).toHaveBeenCalled()

    viewport.stopViewportSync()
    expect(stopSyncMock).toHaveBeenCalled()
  })

  it('should handle null canvas gracefully', () => {
    const canvasRef = ref(null) as Ref<MinimapCanvas | null>
    const graphRef = ref(mockGraph) as Ref<LGraph | null>

    const viewport = useMinimapViewport(canvasRef, graphRef, 250, 200)

    // Should not throw
    expect(() => viewport.updateCanvasDimensions()).not.toThrow()
    expect(() => viewport.updateViewport()).not.toThrow()
    expect(() => viewport.centerViewOn(100, 100)).not.toThrow()
  })

  it('should calculate scale correctly', () => {
    const testBounds = {
      minX: 0,
      minY: 0,
      maxX: 500,
      maxY: 400,
      width: 500,
      height: 400
    }

    vi.mocked(calculateNodeBounds).mockReturnValue(testBounds)
    vi.mocked(enforceMinimumBounds).mockImplementation((bounds) => bounds)
    vi.mocked(calculateMinimapScale).mockReturnValue(0.4)

    const canvasRef = ref(mockCanvas) as Ref<MinimapCanvas | null>
    const graphRef = ref(mockGraph) as Ref<LGraph | null>

    const viewport = useMinimapViewport(canvasRef, graphRef, 250, 200)

    viewport.updateBounds()

    expect(calculateMinimapScale).toHaveBeenCalledWith(testBounds, 250, 200)
    expect(viewport.scale.value).toBe(0.4)
  })

  it('should handle device pixel ratio', () => {
    const originalDPR = window.devicePixelRatio
    Object.defineProperty(window, 'devicePixelRatio', {
      value: 2,
      configurable: true
    })

    const canvasRef = ref(mockCanvas) as Ref<MinimapCanvas | null>
    const graphRef = ref(mockGraph) as Ref<LGraph | null>

    const viewport = useMinimapViewport(canvasRef, graphRef, 250, 200)

    viewport.updateCanvasDimensions()

    // Should use client dimensions or calculate from canvas dimensions / dpr
    expect(viewport.canvasDimensions.value.width).toBe(800)
    expect(viewport.canvasDimensions.value.height).toBe(600)

    Object.defineProperty(window, 'devicePixelRatio', {
      value: originalDPR,
      configurable: true
    })
  })
})
