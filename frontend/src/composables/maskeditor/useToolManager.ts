import { ref, watch } from 'vue'
import type {
  Point,
  ImageLayer,
  ToolInternalSettings
} from '@/extensions/core/maskeditor/types'
import { Tools } from '@/extensions/core/maskeditor/types'
import { useMaskEditorStore } from '@/stores/maskEditorStore'
import { useBrushDrawing } from './useBrushDrawing'
import { useCanvasTools } from './useCanvasTools'
import { useCoordinateTransform } from './useCoordinateTransform'
import type { useKeyboard } from './useKeyboard'
import type { usePanAndZoom } from './usePanAndZoom'
import { app } from '@/scripts/app'

export function useToolManager(
  keyboard: ReturnType<typeof useKeyboard>,
  panZoom: ReturnType<typeof usePanAndZoom>
) {
  const store = useMaskEditorStore()

  const coordinateTransform = useCoordinateTransform()

  const brushDrawing = useBrushDrawing({
    useDominantAxis: app.extensionManager.setting.get<boolean>(
      'Comfy.MaskEditor.UseDominantAxis'
    ),
    brushAdjustmentSpeed: app.extensionManager.setting.get<number>(
      'Comfy.MaskEditor.BrushAdjustmentSpeed'
    )
  })
  const canvasTools = useCanvasTools()

  const mouseDownPoint = ref<Point | null>(null)

  const toolSettings: Record<Tools, Partial<ToolInternalSettings>> = {
    [Tools.MaskPen]: {
      newActiveLayerOnSet: 'mask'
    },
    [Tools.Eraser]: {},
    [Tools.PaintPen]: {
      newActiveLayerOnSet: 'rgb'
    },
    [Tools.MaskBucket]: {
      cursor: "url('/cursor/paintBucket.png') 30 25, auto",
      newActiveLayerOnSet: 'mask'
    },
    [Tools.MaskColorFill]: {
      cursor: "url('/cursor/colorSelect.png') 15 25, auto",
      newActiveLayerOnSet: 'mask'
    }
  }

  const setActiveLayer = (layer: ImageLayer) => {
    store.activeLayer = layer
    const currentTool = store.currentTool

    const maskOnlyTools = [Tools.MaskPen, Tools.MaskBucket, Tools.MaskColorFill]
    if (maskOnlyTools.includes(currentTool) && layer === 'rgb') {
      switchTool(Tools.PaintPen)
    }

    if (currentTool === Tools.PaintPen && layer === 'mask') {
      switchTool(Tools.MaskPen)
    }
  }

  const switchTool = (tool: Tools) => {
    store.currentTool = tool

    const newActiveLayer = toolSettings[tool].newActiveLayerOnSet
    if (newActiveLayer) {
      store.activeLayer = newActiveLayer
    }

    const cursor = toolSettings[tool].cursor
    const pointerZone = store.pointerZone

    if (cursor && pointerZone) {
      store.brushVisible = false
      pointerZone.style.cursor = cursor
    } else if (pointerZone) {
      store.brushVisible = true
      pointerZone.style.cursor = 'none'
    }
  }

  const updateCursor = () => {
    const currentTool = store.currentTool
    const cursor = toolSettings[currentTool].cursor
    const pointerZone = store.pointerZone

    if (cursor && pointerZone) {
      store.brushVisible = false
      pointerZone.style.cursor = cursor
    } else if (pointerZone) {
      store.brushVisible = true
      pointerZone.style.cursor = 'none'
    }

    store.brushPreviewGradientVisible = false
  }

  watch(
    () => store.currentTool,
    (newTool) => {
      if (newTool !== Tools.MaskColorFill) {
        canvasTools.clearLastColorSelectPoint()
      }
    }
  )

  const handlePointerDown = async (event: PointerEvent): Promise<void> => {
    event.preventDefault()
    if (event.pointerType === 'touch') return

    if (event.pointerType === 'pen') {
      panZoom.addPenPointerId(event.pointerId)
    }

    const isSpacePressed = keyboard.isKeyDown(' ')

    if (event.buttons === 4 || (event.buttons === 1 && isSpacePressed)) {
      panZoom.handlePanStart(event)

      store.brushVisible = false
      return
    }

    if (store.currentTool === Tools.PaintPen && event.button === 0) {
      await brushDrawing.startDrawing(event)

      return
    }

    if (store.currentTool === Tools.PaintPen && event.buttons === 1) {
      await brushDrawing.handleDrawing(event)
      return
    }

    if (store.currentTool === Tools.MaskBucket && event.button === 0) {
      const offset = { x: event.offsetX, y: event.offsetY }
      const coords_canvas = coordinateTransform.screenToCanvas(offset)
      canvasTools.paintBucketFill(coords_canvas)
      return
    }

    if (store.currentTool === Tools.MaskColorFill && event.button === 0) {
      const offset = { x: event.offsetX, y: event.offsetY }
      const coords_canvas = coordinateTransform.screenToCanvas(offset)
      await canvasTools.colorSelectFill(coords_canvas)
      return
    }

    if (event.altKey && event.button === 2) {
      store.isAdjustingBrush = true
      await brushDrawing.startBrushAdjustment(event)
      return
    }

    const isDrawingTool = [
      Tools.MaskPen,
      Tools.Eraser,
      Tools.PaintPen
    ].includes(store.currentTool)

    if ([0, 2].includes(event.button) && isDrawingTool) {
      await brushDrawing.startDrawing(event)
      return
    }
  }

  const handlePointerMove = async (event: PointerEvent): Promise<void> => {
    event.preventDefault()
    if (event.pointerType === 'touch') return

    const newCursorPoint = { x: event.clientX, y: event.clientY }
    panZoom.updateCursorPosition(newCursorPoint)

    const isSpacePressed = keyboard.isKeyDown(' ')

    if (event.buttons === 4 || (event.buttons === 1 && isSpacePressed)) {
      await panZoom.handlePanMove(event)
      return
    }

    const isDrawingTool = [
      Tools.MaskPen,
      Tools.Eraser,
      Tools.PaintPen
    ].includes(store.currentTool)
    if (!isDrawingTool) return

    if (
      store.isAdjustingBrush &&
      (store.currentTool === Tools.MaskPen ||
        store.currentTool === Tools.Eraser) &&
      event.altKey &&
      event.buttons === 2
    ) {
      await brushDrawing.handleBrushAdjustment(event)
      return
    }

    if (event.buttons === 1 || event.buttons === 2) {
      await brushDrawing.handleDrawing(event)
      return
    }
  }

  const handlePointerUp = async (event: PointerEvent): Promise<void> => {
    store.isPanning = false
    store.brushVisible = true

    if (event.pointerType === 'pen') {
      panZoom.removePenPointerId(event.pointerId)
    }

    if (event.pointerType === 'touch') return
    updateCursor()
    store.isAdjustingBrush = false
    await brushDrawing.drawEnd(event)
    mouseDownPoint.value = null
  }

  return {
    switchTool,
    setActiveLayer,
    updateCursor,

    handlePointerDown,
    handlePointerMove,
    handlePointerUp,

    brushDrawing
  }
}
