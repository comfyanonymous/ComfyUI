import { ref } from 'vue'
import type { Ref, ShallowRef } from 'vue'

import type { MinimapCanvas } from '../types'

export function useMinimapInteraction(
  containerRef: Readonly<ShallowRef<HTMLDivElement | null>>,
  bounds: Ref<{ minX: number; minY: number; width: number; height: number }>,
  scale: Ref<number>,
  width: number,
  height: number,
  centerViewOn: (worldX: number, worldY: number) => void,
  canvas: Ref<MinimapCanvas | null>
) {
  const isDragging = ref(false)
  const containerRect = ref({
    left: 0,
    top: 0,
    width: width,
    height: height
  })

  const updateContainerRect = () => {
    if (!containerRef.value) return

    const rect = containerRef.value.getBoundingClientRect()
    containerRect.value = {
      left: rect.left,
      top: rect.top,
      width: rect.width,
      height: rect.height
    }
  }

  const handlePointerDown = (e: PointerEvent) => {
    isDragging.value = true
    updateContainerRect()
    const target = e.currentTarget
    if (target instanceof HTMLElement) {
      target.setPointerCapture(e.pointerId)
    }
    handlePointerMove(e)
  }

  const handlePointerMove = (e: PointerEvent) => {
    if (!isDragging.value || !canvas.value) return

    const x = e.clientX - containerRect.value.left
    const y = e.clientY - containerRect.value.top

    const offsetX = (width - bounds.value.width * scale.value) / 2
    const offsetY = (height - bounds.value.height * scale.value) / 2

    const worldX = (x - offsetX) / scale.value + bounds.value.minX
    const worldY = (y - offsetY) / scale.value + bounds.value.minY

    centerViewOn(worldX, worldY)
  }

  const releasePointer = (e?: PointerEvent) => {
    isDragging.value = false
    if (!e) return

    const target = e.currentTarget
    if (
      target instanceof HTMLElement &&
      target.hasPointerCapture(e.pointerId)
    ) {
      target.releasePointerCapture(e.pointerId)
    }
  }

  const handlePointerUp = releasePointer

  const handlePointerCancel = releasePointer

  const handleWheel = (e: WheelEvent) => {
    e.preventDefault()

    const c = canvas.value
    if (!c) return

    if (
      containerRect.value.left === 0 &&
      containerRect.value.top === 0 &&
      containerRef.value
    ) {
      updateContainerRect()
    }

    const ds = c.ds
    const delta = e.deltaY > 0 ? 0.9 : 1.1

    const newScale = ds.scale * delta

    const MIN_SCALE = 0.1
    const MAX_SCALE = 10

    if (newScale < MIN_SCALE || newScale > MAX_SCALE) return

    const x = e.clientX - containerRect.value.left
    const y = e.clientY - containerRect.value.top

    const offsetX = (width - bounds.value.width * scale.value) / 2
    const offsetY = (height - bounds.value.height * scale.value) / 2

    const worldX = (x - offsetX) / scale.value + bounds.value.minX
    const worldY = (y - offsetY) / scale.value + bounds.value.minY

    ds.scale = newScale

    centerViewOn(worldX, worldY)
  }

  return {
    isDragging,
    containerRect,
    updateContainerRect,
    handlePointerDown,
    handlePointerMove,
    handlePointerUp,
    handlePointerCancel,
    handleWheel
  }
}
