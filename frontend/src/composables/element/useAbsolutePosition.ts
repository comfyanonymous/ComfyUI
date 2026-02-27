import type { CSSProperties } from 'vue'
import { ref, watch } from 'vue'

import { useCanvasPositionConversion } from '@/composables/element/useCanvasPositionConversion'
import type { Size, Vector2 } from '@/lib/litegraph/src/litegraph'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'

export interface PositionConfig {
  /* The position of the element on litegraph canvas */
  pos: Vector2
  /* The size of the element on litegraph canvas */
  size: Size
  /* The scale factor of the canvas */
  scale?: number
}

export function useAbsolutePosition(options: { useTransform?: boolean } = {}) {
  const { useTransform = false } = options

  const canvasStore = useCanvasStore()
  const lgCanvas = canvasStore.getCanvas()
  const { canvasPosToClientPos, update: updateCanvasPosition } =
    useCanvasPositionConversion(lgCanvas.canvas, lgCanvas)

  const settingStore = useSettingStore()
  watch(
    [
      () => settingStore.get('Comfy.Sidebar.Location'),
      () => settingStore.get('Comfy.Sidebar.Size'),
      () => settingStore.get('Comfy.UseNewMenu')
    ],
    () => updateCanvasPosition(),
    { flush: 'post' }
  )

  /**
   * @note Do NOT convert style to a computed value, as it will cause lag when
   * updating the style on different animation frames. Vue's computed value is
   * evaluated asynchronously.
   */
  const style = ref<CSSProperties>({})

  /**
   * Compute the style of the element based on the position and size.
   *
   * @param position
   */
  const computeStyle = (position: PositionConfig): CSSProperties => {
    const { pos, size, scale = lgCanvas.ds.scale } = position
    const [left, top] = canvasPosToClientPos(pos)
    const [width, height] = size

    return useTransform
      ? {
          position: 'fixed',
          transformOrigin: '0 0',
          transform: `scale(${scale})`,
          left: `${left}px`,
          top: `${top}px`,
          width: `${width}px`,
          height: `${height}px`
        }
      : {
          position: 'fixed',
          left: `${left}px`,
          top: `${top}px`,
          width: `${width * scale}px`,
          height: `${height * scale}px`
        }
  }

  /**
   * Update the position of the element on the litegraph canvas.
   *
   * @param config
   */
  const updatePosition = (config: PositionConfig) => {
    style.value = computeStyle(config)
  }

  return {
    style,
    updatePosition
  }
}
