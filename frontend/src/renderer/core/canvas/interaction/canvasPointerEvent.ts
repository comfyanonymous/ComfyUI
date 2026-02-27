import { useSharedCanvasPositionConversion } from '@/composables/element/useCanvasPositionConversion'
import type {
  CanvasPointerEvent,
  CanvasPointerExtensions
} from '@/lib/litegraph/src/types/events'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'

type PointerOffsets = {
  x: number
  y: number
}

const pointerHistory = new Map<number, PointerOffsets>()

const defineEnhancements = (
  event: PointerEvent,
  enhancement: CanvasPointerExtensions
) => {
  Object.defineProperties(event, {
    canvasX: { value: enhancement.canvasX, configurable: true, writable: true },
    canvasY: { value: enhancement.canvasY, configurable: true, writable: true },
    deltaX: { value: enhancement.deltaX, configurable: true, writable: true },
    deltaY: { value: enhancement.deltaY, configurable: true, writable: true },
    safeOffsetX: {
      value: enhancement.safeOffsetX,
      configurable: true,
      writable: true
    },
    safeOffsetY: {
      value: enhancement.safeOffsetY,
      configurable: true,
      writable: true
    }
  })
}

const createEnhancement = (event: PointerEvent): CanvasPointerExtensions => {
  const conversion = useSharedCanvasPositionConversion()
  conversion.update()

  const [canvasX, canvasY] = conversion.clientPosToCanvasPos([
    event.clientX,
    event.clientY
  ])

  const canvas = useCanvasStore().getCanvas()
  const { offset, scale } = canvas.ds

  const [originClientX, originClientY] = conversion.canvasPosToClientPos([0, 0])
  const left = originClientX - offset[0] * scale
  const top = originClientY - offset[1] * scale

  const safeOffsetX = event.clientX - left
  const safeOffsetY = event.clientY - top

  const previous = pointerHistory.get(event.pointerId)
  const deltaX = previous ? safeOffsetX - previous.x : 0
  const deltaY = previous ? safeOffsetY - previous.y : 0
  pointerHistory.set(event.pointerId, { x: safeOffsetX, y: safeOffsetY })

  return { canvasX, canvasY, deltaX, deltaY, safeOffsetX, safeOffsetY }
}

export const toCanvasPointerEvent = <T extends PointerEvent>(
  event: T
): T & CanvasPointerEvent => {
  const enhancement = createEnhancement(event)
  defineEnhancements(event, enhancement)
  return event as T & CanvasPointerEvent
}

export const clearCanvasPointerHistory = (pointerId: number) => {
  pointerHistory.delete(pointerId)
}
