import { createSharedComposable } from '@vueuse/core'
import { unref } from 'vue'
import { useMaskEditorStore } from '@/stores/maskEditorStore'

interface Point {
  x: number
  y: number
}

function useCoordinateTransformInternal() {
  const store = useMaskEditorStore()

  const screenToCanvas = (clientPoint: Point): Point => {
    const pointerZoneEl = unref(store.pointerZone)
    const canvasContainerEl = unref(store.canvasContainer)
    const canvasEl = unref(store.maskCanvas)

    if (!pointerZoneEl || !canvasContainerEl || !canvasEl) {
      console.warn('screenToCanvas called before elements are available')
      return { x: 0, y: 0 }
    }

    const pointerZoneRect = pointerZoneEl.getBoundingClientRect()
    const canvasContainerRect = canvasContainerEl.getBoundingClientRect()
    const canvasRect = canvasEl.getBoundingClientRect()

    const absoluteX = pointerZoneRect.left + clientPoint.x
    const absoluteY = pointerZoneRect.top + clientPoint.y

    const canvasX = absoluteX - canvasContainerRect.left
    const canvasY = absoluteY - canvasContainerRect.top

    const scaleX = canvasEl.width / canvasRect.width
    const scaleY = canvasEl.height / canvasRect.height

    const x = canvasX * scaleX
    const y = canvasY * scaleY

    return { x, y }
  }

  const canvasToScreen = (canvasPoint: Point): Point => {
    const pointerZoneEl = unref(store.pointerZone)
    const canvasContainerEl = unref(store.canvasContainer)
    const canvasEl = unref(store.maskCanvas)

    if (!pointerZoneEl || !canvasContainerEl || !canvasEl) {
      console.warn('canvasToScreen called before elements are available')
      return { x: 0, y: 0 }
    }

    const pointerZoneRect = pointerZoneEl.getBoundingClientRect()
    const canvasContainerRect = canvasContainerEl.getBoundingClientRect()
    const canvasRect = canvasEl.getBoundingClientRect()

    const scaleX = canvasRect.width / canvasEl.width
    const scaleY = canvasRect.height / canvasEl.height

    const displayX = canvasPoint.x * scaleX
    const displayY = canvasPoint.y * scaleY

    const absoluteX = canvasContainerRect.left + displayX
    const absoluteY = canvasContainerRect.top + displayY

    const x = absoluteX - pointerZoneRect.left
    const y = absoluteY - pointerZoneRect.top

    return { x, y }
  }

  return {
    screenToCanvas,
    canvasToScreen
  }
}

export const useCoordinateTransform = createSharedComposable(
  useCoordinateTransformInternal
)
