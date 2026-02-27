import { useElementBounding } from '@vueuse/core'

import type { LGraphCanvas, Point } from '@/lib/litegraph/src/litegraph'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'

let sharedConverter: ReturnType<typeof useCanvasPositionConversion> | null =
  null

/**
 * Convert between canvas and client positions
 * @param canvasElement - The canvas element
 * @param lgCanvas - The litegraph canvas
 * @returns The canvas position conversion functions
 */
export const useCanvasPositionConversion = (
  canvasElement: Parameters<typeof useElementBounding>[0],
  lgCanvas: LGraphCanvas
) => {
  const { left, top, update } = useElementBounding(canvasElement)

  const clientPosToCanvasPos = (pos: Point): Point => {
    const { offset, scale } = lgCanvas.ds
    return [
      (pos[0] - left.value) / scale - offset[0],
      (pos[1] - top.value) / scale - offset[1]
    ]
  }

  const canvasPosToClientPos = (pos: Point): Point => {
    const { offset, scale } = lgCanvas.ds
    return [
      (pos[0] + offset[0]) * scale + left.value,
      (pos[1] + offset[1]) * scale + top.value
    ]
  }

  return {
    clientPosToCanvasPos,
    canvasPosToClientPos,
    update
  }
}

export function useSharedCanvasPositionConversion() {
  if (sharedConverter) return sharedConverter
  const lgCanvas = useCanvasStore().getCanvas()
  sharedConverter = useCanvasPositionConversion(lgCanvas.canvas, lgCanvas)
  return sharedConverter
}
