import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'
import type { Ref } from 'vue'

import { useMinimapInteraction } from '@/renderer/extensions/minimap/composables/useMinimapInteraction'
import type { MinimapCanvas } from '@/renderer/extensions/minimap/types'

describe('useMinimapInteraction', () => {
  let mockContainer: HTMLDivElement
  let mockCanvas: MinimapCanvas
  let centerViewOnMock: (worldX: number, worldY: number) => void

  beforeEach(() => {
    vi.clearAllMocks()

    mockContainer = {
      getBoundingClientRect: vi.fn().mockReturnValue({
        left: 100,
        top: 50,
        width: 250,
        height: 200
      })
    } as Partial<HTMLDivElement> as HTMLDivElement

    mockCanvas = {
      ds: {
        scale: 1,
        offset: [0, 0]
      },
      setDirty: vi.fn()
    } as Partial<MinimapCanvas> as MinimapCanvas

    centerViewOnMock = vi.fn<(worldX: number, worldY: number) => void>()
  })

  it('should initialize with default values', () => {
    const containerRef = ref(mockContainer)
    const boundsRef = ref({ minX: 0, minY: 0, width: 500, height: 400 })
    const scaleRef = ref(0.5)
    const canvasRef = ref(mockCanvas) as Ref<MinimapCanvas | null>

    const interaction = useMinimapInteraction(
      containerRef,
      boundsRef,
      scaleRef,
      250,
      200,
      centerViewOnMock,
      canvasRef
    )

    expect(interaction.isDragging.value).toBe(false)
    expect(interaction.containerRect.value).toEqual({
      left: 0,
      top: 0,
      width: 250,
      height: 200
    })
  })

  it('should update container rect', () => {
    const containerRef = ref(mockContainer)
    const boundsRef = ref({ minX: 0, minY: 0, width: 500, height: 400 })
    const scaleRef = ref(0.5)
    const canvasRef = ref(mockCanvas) as Ref<MinimapCanvas | null>

    const interaction = useMinimapInteraction(
      containerRef,
      boundsRef,
      scaleRef,
      250,
      200,
      centerViewOnMock,
      canvasRef
    )

    interaction.updateContainerRect()

    expect(mockContainer.getBoundingClientRect).toHaveBeenCalled()

    expect(interaction.containerRect.value).toEqual({
      left: 100,
      top: 50,
      width: 250,
      height: 200
    })
  })

  it('should handle pointer down and start dragging', () => {
    const containerRef = ref(mockContainer)
    const boundsRef = ref({ minX: 0, minY: 0, width: 500, height: 400 })
    const scaleRef = ref(0.5)
    const canvasRef = ref(mockCanvas) as Ref<MinimapCanvas | null>

    const interaction = useMinimapInteraction(
      containerRef,
      boundsRef,
      scaleRef,
      250,
      200,
      centerViewOnMock,
      canvasRef
    )

    const event = new PointerEvent('pointerdown', {
      clientX: 150,
      clientY: 100
    })

    interaction.handlePointerDown(event)

    expect(interaction.isDragging.value).toBe(true)
    expect(mockContainer.getBoundingClientRect).toHaveBeenCalled()
    expect(centerViewOnMock).toHaveBeenCalled()
  })

  it('should handle pointer move when dragging', () => {
    const containerRef = ref(mockContainer)
    const boundsRef = ref({ minX: 0, minY: 0, width: 500, height: 400 })
    const scaleRef = ref(0.5)
    const canvasRef = ref(mockCanvas) as Ref<MinimapCanvas | null>

    const interaction = useMinimapInteraction(
      containerRef,
      boundsRef,
      scaleRef,
      250,
      200,
      centerViewOnMock,
      canvasRef
    )

    // Start dragging
    interaction.handlePointerDown(
      new PointerEvent('pointerdown', {
        clientX: 150,
        clientY: 100
      })
    )

    // Move pointer
    const moveEvent = new PointerEvent('pointermove', {
      clientX: 200,
      clientY: 150
    })

    interaction.handlePointerMove(moveEvent)

    // Should calculate world coordinates and center view
    expect(centerViewOnMock).toHaveBeenCalledTimes(2) // Once on down, once on move

    // Calculate expected world coordinates
    const x = 200 - 100 // clientX - containerLeft
    const y = 150 - 50 // clientY - containerTop
    const offsetX = (250 - 500 * 0.5) / 2 // (width - bounds.width * scale) / 2
    const offsetY = (200 - 400 * 0.5) / 2 // (height - bounds.height * scale) / 2
    const worldX = (x - offsetX) / 0.5 + 0 // (x - offsetX) / scale + bounds.minX
    const worldY = (y - offsetY) / 0.5 + 0 // (y - offsetY) / scale + bounds.minY

    expect(centerViewOnMock).toHaveBeenLastCalledWith(worldX, worldY)
  })

  it('should not move when not dragging', () => {
    const containerRef = ref(mockContainer)
    const boundsRef = ref({ minX: 0, minY: 0, width: 500, height: 400 })
    const scaleRef = ref(0.5)
    const canvasRef = ref(mockCanvas) as Ref<MinimapCanvas | null>

    const interaction = useMinimapInteraction(
      containerRef,
      boundsRef,
      scaleRef,
      250,
      200,
      centerViewOnMock,
      canvasRef
    )

    const moveEvent = new PointerEvent('pointermove', {
      clientX: 200,
      clientY: 150
    })

    interaction.handlePointerMove(moveEvent)

    expect(centerViewOnMock).not.toHaveBeenCalled()
  })

  it('should handle pointer up to stop dragging', () => {
    const containerRef = ref(mockContainer)
    const boundsRef = ref({ minX: 0, minY: 0, width: 500, height: 400 })
    const scaleRef = ref(0.5)
    const canvasRef = ref(mockCanvas) as Ref<MinimapCanvas | null>

    const interaction = useMinimapInteraction(
      containerRef,
      boundsRef,
      scaleRef,
      250,
      200,
      centerViewOnMock,
      canvasRef
    )

    // Start dragging
    interaction.handlePointerDown(
      new PointerEvent('pointerdown', {
        clientX: 150,
        clientY: 100
      })
    )

    expect(interaction.isDragging.value).toBe(true)

    interaction.handlePointerUp()

    expect(interaction.isDragging.value).toBe(false)
  })

  it('should handle wheel events for zooming', () => {
    const containerRef = ref(mockContainer)
    const boundsRef = ref({ minX: 0, minY: 0, width: 500, height: 400 })
    const scaleRef = ref(0.5)
    const canvasRef = ref(mockCanvas) as Ref<MinimapCanvas | null>

    const interaction = useMinimapInteraction(
      containerRef,
      boundsRef,
      scaleRef,
      250,
      200,
      centerViewOnMock,
      canvasRef
    )

    const wheelEvent = new WheelEvent('wheel', {
      deltaY: -100,
      clientX: 200,
      clientY: 150
    })
    wheelEvent.preventDefault = vi.fn()

    interaction.handleWheel(wheelEvent)

    // Should update canvas scale (zoom in)
    expect(mockCanvas.ds.scale).toBeCloseTo(1.1)
    expect(centerViewOnMock).toHaveBeenCalled()
  })

  it('should respect zoom limits', () => {
    const containerRef = ref(mockContainer)
    const boundsRef = ref({ minX: 0, minY: 0, width: 500, height: 400 })
    const scaleRef = ref(0.5)
    const canvasRef = ref(mockCanvas) as Ref<MinimapCanvas | null>

    const interaction = useMinimapInteraction(
      containerRef,
      boundsRef,
      scaleRef,
      250,
      200,
      centerViewOnMock,
      canvasRef
    )

    // Set scale close to minimum
    mockCanvas.ds.scale = 0.11

    const wheelEvent = new WheelEvent('wheel', {
      deltaY: 100, // Zoom out
      clientX: 200,
      clientY: 150
    })
    wheelEvent.preventDefault = vi.fn()

    interaction.handleWheel(wheelEvent)

    // Should not go below minimum scale
    expect(mockCanvas.ds.scale).toBe(0.11)
    expect(centerViewOnMock).not.toHaveBeenCalled()
  })

  it('should handle null container gracefully', () => {
    const containerRef = ref<HTMLDivElement | null>(null)
    const boundsRef = ref({ minX: 0, minY: 0, width: 500, height: 400 })
    const scaleRef = ref(0.5)
    const canvasRef = ref(mockCanvas) as Ref<MinimapCanvas | null>

    const interaction = useMinimapInteraction(
      containerRef,
      boundsRef,
      scaleRef,
      250,
      200,
      centerViewOnMock,
      canvasRef
    )

    // Should not throw
    expect(() => interaction.updateContainerRect()).not.toThrow()
    expect(() =>
      interaction.handlePointerDown(new PointerEvent('pointerdown'))
    ).not.toThrow()
  })

  it('should handle null canvas gracefully', () => {
    const containerRef = ref(mockContainer)
    const boundsRef = ref({ minX: 0, minY: 0, width: 500, height: 400 })
    const scaleRef = ref(0.5)
    const canvasRef = ref(null) as Ref<MinimapCanvas | null>

    const interaction = useMinimapInteraction(
      containerRef,
      boundsRef,
      scaleRef,
      250,
      200,
      centerViewOnMock,
      canvasRef
    )

    // Should not throw
    expect(() =>
      interaction.handlePointerMove(new PointerEvent('pointermove'))
    ).not.toThrow()
    expect(() => interaction.handleWheel(new WheelEvent('wheel'))).not.toThrow()
    expect(centerViewOnMock).not.toHaveBeenCalled()
  })
})
