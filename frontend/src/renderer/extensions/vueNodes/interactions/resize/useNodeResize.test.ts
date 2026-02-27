import type { MockInstance } from 'vitest'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { CompassCorners } from '@/lib/litegraph/src/interfaces'

import type { ResizeCallbackPayload } from './useNodeResize'

type ResizeCallback = (
  payload: ResizeCallbackPayload,
  element: HTMLElement
) => void

// Capture pointermove/pointerup handlers registered via useEventListener
const eventHandlers = vi.hoisted(() => ({
  pointermove: null as ((e: PointerEvent) => void) | null,
  pointerup: null as ((e: PointerEvent) => void) | null
}))

vi.mock('@vueuse/core', () => ({
  useEventListener: vi.fn(
    (eventName: string, handler: (...args: unknown[]) => void) => {
      if (eventName === 'pointermove' || eventName === 'pointerup') {
        eventHandlers[eventName] = handler as (e: PointerEvent) => void
      }
      return vi.fn()
    }
  )
}))

vi.mock('@/renderer/core/layout/transform/useTransformState', () => ({
  useTransformState: () => ({
    camera: { x: 0, y: 0, z: 1 }
  })
}))

vi.mock('@/renderer/extensions/vueNodes/composables/useNodeSnap', () => ({
  useNodeSnap: () => ({
    shouldSnap: vi.fn(() => false),
    applySnapToPosition: vi.fn((pos: { x: number; y: number }) => pos),
    applySnapToSize: vi.fn((size: { width: number; height: number }) => size)
  })
}))

vi.mock('@/renderer/extensions/vueNodes/composables/useShiftKeySync', () => ({
  useShiftKeySync: () => ({
    trackShiftKey: vi.fn(() => vi.fn())
  })
}))

vi.mock('@/renderer/core/layout/store/layoutStore', () => ({
  layoutStore: {
    isResizingVueNodes: { value: false },
    getNodeLayoutRef: vi.fn(() => ({
      value: {
        position: { x: 100, y: 200 },
        size: { width: 300, height: 400 }
      }
    }))
  }
}))

function createMockNodeElement(
  width = 300,
  height = 400,
  minContentHeight = 150
): HTMLElement {
  const element = document.createElement('div')
  element.setAttribute('data-node-id', 'test-node')
  element.style.setProperty('min-width', '225px')
  element.getBoundingClientRect = () => {
    // When --node-height is '0px', return the content-driven minimum height
    const nodeHeight = element.style.getPropertyValue('--node-height')
    const h = nodeHeight === '0px' ? minContentHeight : height
    return {
      width,
      height: h,
      x: 0,
      y: 0,
      top: 0,
      left: 0,
      right: width,
      bottom: h,
      toJSON: () => {}
    } as DOMRect
  }
  return element
}

function createMockHandle(nodeElement: HTMLElement): HTMLElement {
  const handle = document.createElement('div')
  nodeElement.appendChild(handle)
  handle.setPointerCapture = vi.fn()
  handle.releasePointerCapture = vi.fn()
  return handle
}

function createPointerEvent(
  type: string,
  overrides: Partial<PointerEvent> = {}
): PointerEvent {
  return {
    type,
    clientX: 0,
    clientY: 0,
    pointerId: 1,
    shiftKey: false,
    currentTarget: null,
    preventDefault: vi.fn(),
    stopPropagation: vi.fn(),
    ...overrides
  } as Partial<PointerEvent> as PointerEvent
}

function startResizeAt(
  startResize: (event: PointerEvent, corner: CompassCorners) => void,
  handle: HTMLElement,
  corner: CompassCorners,
  clientX = 500,
  clientY = 500
) {
  const downEvent = createPointerEvent('pointerdown', {
    currentTarget: handle,
    clientX,
    clientY
  } as Partial<PointerEvent>)
  startResize(downEvent, corner)
}

function simulateMove(
  deltaX: number,
  deltaY: number,
  startX = 500,
  startY = 500
) {
  const moveEvent = createPointerEvent('pointermove', {
    clientX: startX + deltaX,
    clientY: startY + deltaY
  })
  eventHandlers.pointermove?.(moveEvent)
}

describe('useNodeResize', () => {
  let callback: ResizeCallback & MockInstance<ResizeCallback>
  let nodeElement: HTMLElement
  let handle: HTMLElement

  beforeEach(async () => {
    vi.clearAllMocks()
    eventHandlers.pointermove = null
    eventHandlers.pointerup = null

    callback = vi.fn<ResizeCallback>()
    nodeElement = createMockNodeElement()
    handle = createMockHandle(nodeElement)

    // Need fresh import after mocks are set up
    const { useNodeResize } = await import('./useNodeResize')
    const { startResize } = useNodeResize(callback)

    // Store startResize for access in tests
    ;(globalThis as Record<string, unknown>).__testStartResize = startResize
  })

  function getStartResize() {
    return (globalThis as Record<string, unknown>).__testStartResize as (
      event: PointerEvent,
      corner: CompassCorners
    ) => void
  }

  describe('SE corner (default)', () => {
    it('increases size when dragging right and down', () => {
      startResizeAt(getStartResize(), handle, 'SE')
      simulateMove(50, 30)

      expect(callback).toHaveBeenCalledWith(
        expect.objectContaining({
          size: { width: 350, height: 430 }
        }),
        nodeElement
      )
    })

    it('does not include position in payload', () => {
      startResizeAt(getStartResize(), handle, 'SE')
      simulateMove(50, 30)

      const payload: ResizeCallbackPayload = callback.mock.calls[0][0]
      expect(payload.position).toBeUndefined()
    })

    it('clamps width to minimum', () => {
      startResizeAt(getStartResize(), handle, 'SE')
      simulateMove(-200, 0)

      const payload: ResizeCallbackPayload = callback.mock.calls[0][0]
      expect(payload.size.width).toBe(225)
    })
  })

  describe('NE corner', () => {
    it('increases width right, decreases height upward, shifts y position', () => {
      startResizeAt(getStartResize(), handle, 'NE')
      simulateMove(50, -30)

      const payload: ResizeCallbackPayload = callback.mock.calls[0][0]
      expect(payload.size).toEqual({ width: 350, height: 430 })
      expect(payload.position).toEqual({ x: 100, y: 170 })
    })

    it('clamps height to content minimum and fixes bottom edge', () => {
      startResizeAt(getStartResize(), handle, 'NE')
      simulateMove(0, 500)

      const payload: ResizeCallbackPayload = callback.mock.calls[0][0]
      // minContentHeight = 150, so height clamps to 150
      expect(payload.size.height).toBe(150)
      // y = startY + startHeight - minContentHeight = 200 + 400 - 150 = 450
      expect(payload.position!.y).toBe(450)
    })
  })

  describe('SW corner', () => {
    it('decreases width leftward, increases height downward, shifts x position', () => {
      startResizeAt(getStartResize(), handle, 'SW')
      simulateMove(-50, 30)

      const payload: ResizeCallbackPayload = callback.mock.calls[0][0]
      expect(payload.size).toEqual({ width: 350, height: 430 })
      expect(payload.position).toEqual({ x: 50, y: 200 })
    })

    it('clamps width to minimum and fixes right edge', () => {
      startResizeAt(getStartResize(), handle, 'SW')
      simulateMove(200, 0)

      const payload: ResizeCallbackPayload = callback.mock.calls[0][0]
      expect(payload.size.width).toBe(225)
      expect(payload.position!.x).toBe(175)
    })
  })

  describe('NW corner', () => {
    it('decreases width leftward, decreases height upward, shifts both x and y', () => {
      startResizeAt(getStartResize(), handle, 'NW')
      simulateMove(-50, -30)

      const payload: ResizeCallbackPayload = callback.mock.calls[0][0]
      expect(payload.size).toEqual({ width: 350, height: 430 })
      expect(payload.position).toEqual({ x: 50, y: 170 })
    })

    it('clamps width to minimum and fixes right edge', () => {
      startResizeAt(getStartResize(), handle, 'NW')
      simulateMove(200, 0)

      const payload: ResizeCallbackPayload = callback.mock.calls[0][0]
      expect(payload.size.width).toBe(225)
      expect(payload.position!.x).toBe(175)
    })

    it('clamps height to content minimum and fixes bottom edge', () => {
      startResizeAt(getStartResize(), handle, 'NW')
      simulateMove(0, 500)

      const payload: ResizeCallbackPayload = callback.mock.calls[0][0]
      // minContentHeight = 150, so height clamps to 150
      expect(payload.size.height).toBe(150)
      // y = startY + startHeight - minContentHeight = 200 + 400 - 150 = 450
      expect(payload.position!.y).toBe(450)
    })
  })
})
