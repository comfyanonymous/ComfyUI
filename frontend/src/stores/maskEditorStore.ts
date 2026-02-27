import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import _ from 'es-toolkit/compat'
import type { TgpuRoot } from 'typegpu'

import {
  BrushShape,
  ColorComparisonMethod,
  MaskBlendMode,
  Tools
} from '@/extensions/core/maskeditor/types'
import type {
  Brush,
  ImageLayer,
  Offset,
  Point
} from '@/extensions/core/maskeditor/types'
import { useCanvasHistory } from '@/composables/maskeditor/useCanvasHistory'

export const useMaskEditorStore = defineStore('maskEditor', () => {
  const brushSettings = ref<Brush>({
    type: BrushShape.Arc,
    size: 10,
    opacity: 0.7,
    hardness: 1,
    stepSize: 10
  })

  const maskBlendMode = ref<MaskBlendMode>(MaskBlendMode.Black)
  const activeLayer = ref<ImageLayer>('mask')
  const rgbColor = ref<string>('#FF0000')

  const currentTool = ref<Tools>(Tools.MaskPen)
  const isAdjustingBrush = ref<boolean>(false)

  const paintBucketTolerance = ref<number>(5)
  const fillOpacity = ref<number>(100)

  const colorSelectTolerance = ref<number>(20)
  const colorSelectLivePreview = ref<boolean>(false)
  const colorComparisonMethod = ref<ColorComparisonMethod>(
    ColorComparisonMethod.Simple
  )
  const applyWholeImage = ref<boolean>(false)
  const maskBoundary = ref<boolean>(false)
  const maskTolerance = ref<number>(0)
  const selectionOpacity = ref<number>(100)

  const zoomRatio = ref<number>(1)
  const displayZoomRatio = ref<number>(1)
  const panOffset = ref<Offset>({ x: 0, y: 0 })
  const cursorPoint = ref<Point>({ x: 0, y: 0 })
  const resetZoomTrigger = ref<number>(0)
  const clearTrigger = ref<number>(0)

  const maskCanvas = ref<HTMLCanvasElement | null>(null)
  const maskCtx = ref<CanvasRenderingContext2D | null>(null)
  const rgbCanvas = ref<HTMLCanvasElement | null>(null)
  const rgbCtx = ref<CanvasRenderingContext2D | null>(null)
  const imgCanvas = ref<HTMLCanvasElement | null>(null)
  const imgCtx = ref<CanvasRenderingContext2D | null>(null)
  const canvasContainer = ref<HTMLElement | null>(null)
  const canvasBackground = ref<HTMLElement | null>(null)
  const pointerZone = ref<HTMLElement | null>(null)
  const image = ref<HTMLImageElement | null>(null)

  const maskOpacity = ref<number>(0.8)

  const brushVisible = ref<boolean>(true)
  const isPanning = ref<boolean>(false)
  const brushPreviewGradientVisible = ref<boolean>(false)

  const canvasHistory = useCanvasHistory(20)

  const tgpuRoot = ref<TgpuRoot | null>(null)

  const colorInput = ref<HTMLInputElement | null>(null)

  // GPU texture recreation signals
  /**
   * GPU texture data must use ArrayBuffer (not SharedArrayBuffer) for compatibility
   * with WebGPU's device.queue.writeTexture API. SharedArrayBuffer is not accepted
   * by the WebGPU specification and will cause runtime errors.
   *
   * @see https://gpuweb.github.io/gpuweb/#dom-gpuqueue-writetexture
   */
  type GPUCompatibleArray = Uint8ClampedArray & { buffer: ArrayBuffer }
  const gpuTexturesNeedRecreation = ref<boolean>(false)
  const gpuTextureWidth = ref<number>(0)
  const gpuTextureHeight = ref<number>(0)
  const pendingGPUMaskData = ref<GPUCompatibleArray | null>(null)
  const pendingGPURgbData = ref<GPUCompatibleArray | null>(null)

  watch(maskCanvas, (canvas) => {
    if (canvas) {
      maskCtx.value = canvas.getContext('2d', { willReadFrequently: true })
    }
  })

  watch(rgbCanvas, (canvas) => {
    if (canvas) {
      rgbCtx.value = canvas.getContext('2d', { willReadFrequently: true })
    }
  })

  watch(imgCanvas, (canvas) => {
    if (canvas) {
      imgCtx.value = canvas.getContext('2d', { willReadFrequently: true })
    }
  })

  const canUndo = computed(() => {
    return canvasHistory.canUndo.value
  })

  const canRedo = computed(() => {
    return canvasHistory.canRedo.value
  })

  const maskColor = computed(() => {
    switch (maskBlendMode.value) {
      case MaskBlendMode.Black:
        return { r: 0, g: 0, b: 0 }
      case MaskBlendMode.White:
        return { r: 255, g: 255, b: 255 }
      case MaskBlendMode.Negative:
        return { r: 255, g: 255, b: 255 }
      default:
        return { r: 0, g: 0, b: 0 }
    }
  })

  function setBrushSize(size: number): void {
    brushSettings.value.size = _.clamp(size, 1, 250)
  }

  function setBrushOpacity(opacity: number): void {
    brushSettings.value.opacity = _.clamp(opacity, 0, 1)
  }

  function setBrushHardness(hardness: number): void {
    brushSettings.value.hardness = _.clamp(hardness, 0, 1)
  }

  function setBrushStepSize(step: number): void {
    brushSettings.value.stepSize = _.clamp(step, 1, 100)
  }

  function resetBrushToDefault(): void {
    brushSettings.value.type = BrushShape.Arc
    brushSettings.value.size = 20
    brushSettings.value.opacity = 1
    brushSettings.value.hardness = 1
    brushSettings.value.stepSize = 5
  }

  function setPaintBucketTolerance(tolerance: number): void {
    paintBucketTolerance.value = _.clamp(tolerance, 0, 255)
  }

  function setFillOpacity(opacity: number): void {
    fillOpacity.value = _.clamp(opacity, 0, 100)
  }

  function setColorSelectTolerance(tolerance: number): void {
    colorSelectTolerance.value = _.clamp(tolerance, 0, 255)
  }

  function setMaskTolerance(tolerance: number): void {
    maskTolerance.value = _.clamp(tolerance, 0, 255)
  }

  function setSelectionOpacity(opacity: number): void {
    selectionOpacity.value = _.clamp(opacity, 0, 100)
  }

  function setZoomRatio(ratio: number): void {
    zoomRatio.value = Math.max(0.1, Math.min(10, ratio))
  }

  function setPanOffset(offset: Offset): void {
    panOffset.value = { ...offset }
  }

  function setCursorPoint(point: Point): void {
    cursorPoint.value = { ...point }
  }

  function resetZoom(): void {
    resetZoomTrigger.value++
  }

  function triggerClear(): void {
    clearTrigger.value++
  }

  function setMaskOpacity(opacity: number): void {
    maskOpacity.value = _.clamp(opacity, 0, 1)
  }

  function resetState(): void {
    brushSettings.value = {
      type: BrushShape.Arc,
      size: 10,
      opacity: 0.7,
      hardness: 1,
      stepSize: 5
    }
    maskBlendMode.value = MaskBlendMode.Black
    activeLayer.value = 'mask'
    rgbColor.value = '#FF0000'
    currentTool.value = Tools.MaskPen
    isAdjustingBrush.value = false
    paintBucketTolerance.value = 5
    fillOpacity.value = 100
    colorSelectTolerance.value = 20
    colorSelectLivePreview.value = false
    colorComparisonMethod.value = ColorComparisonMethod.Simple
    applyWholeImage.value = false
    maskBoundary.value = false
    maskTolerance.value = 0
    selectionOpacity.value = 100
    zoomRatio.value = 1
    panOffset.value = { x: 0, y: 0 }
    cursorPoint.value = { x: 0, y: 0 }
    maskOpacity.value = 0.8

    // Reset GPU recreation flags
    gpuTexturesNeedRecreation.value = false
    gpuTextureWidth.value = 0
    gpuTextureHeight.value = 0
    pendingGPUMaskData.value = null
    pendingGPURgbData.value = null
  }

  return {
    brushSettings,
    maskBlendMode,
    activeLayer,
    rgbColor,
    currentTool,
    isAdjustingBrush,
    paintBucketTolerance,
    fillOpacity,
    colorSelectTolerance,
    colorSelectLivePreview,
    colorComparisonMethod,
    applyWholeImage,
    maskBoundary,
    maskTolerance,
    selectionOpacity,
    zoomRatio,
    displayZoomRatio,
    panOffset,
    cursorPoint,
    resetZoomTrigger,
    maskCanvas,
    maskCtx,
    rgbCanvas,
    rgbCtx,
    imgCanvas,
    imgCtx,
    canvasContainer,
    canvasBackground,
    pointerZone,
    image,
    maskOpacity,
    canUndo,
    canRedo,
    maskColor,

    brushVisible,
    isPanning,
    brushPreviewGradientVisible,

    canvasHistory,

    tgpuRoot,

    // GPU texture recreation signals
    gpuTexturesNeedRecreation,
    gpuTextureWidth,
    gpuTextureHeight,
    pendingGPUMaskData,
    pendingGPURgbData,

    colorInput,

    setBrushSize,
    setBrushOpacity,
    setBrushHardness,
    setBrushStepSize,
    resetBrushToDefault,
    setPaintBucketTolerance,
    setFillOpacity,
    setColorSelectTolerance,
    setMaskTolerance,
    setSelectionOpacity,
    setZoomRatio,
    setPanOffset,
    setCursorPoint,
    resetZoom,
    triggerClear,
    clearTrigger,
    setMaskOpacity,
    resetState
  }
})
