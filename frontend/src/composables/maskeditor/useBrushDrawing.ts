/// <reference types="@webgpu/types" />
import { ref, watch, nextTick, onUnmounted } from 'vue'
import QuickLRU from '@alloc/quick-lru'
import { debounce } from 'es-toolkit/compat'
import { hexToRgb, parseToRgb } from '@/utils/colorUtil'
import { getStorageValue, setStorageValue } from '@/scripts/utils'
import {
  Tools,
  BrushShape,
  CompositionOperation
} from '@/extensions/core/maskeditor/types'
import type { Brush, Point } from '@/extensions/core/maskeditor/types'
import { useMaskEditorStore } from '@/stores/maskEditorStore'
import { useCoordinateTransform } from './useCoordinateTransform'
import { resampleSegment } from './splineUtils'
import { tgpu } from 'typegpu'
import { GPUBrushRenderer } from './gpu/GPUBrushRenderer'
import { StrokeProcessor } from './StrokeProcessor'
import { getEffectiveBrushSize, getEffectiveHardness } from './brushUtils'

/**
 * Saves the brush settings to local storage with a debounce.
 * @param key - The storage key.
 * @param brush - The brush settings object.
 */
const saveBrushToCache = debounce(function (key: string, brush: Brush): void {
  try {
    const brushString = JSON.stringify(brush)
    setStorageValue(key, brushString)
  } catch (error) {
    console.error('Failed to save brush to cache:', error)
  }
}, 300)

/**
 * Loads brush settings from local storage.
 * @param key - The storage key.
 * @returns The brush settings object or null if not found.
 */
function loadBrushFromCache(key: string): Brush | null {
  try {
    const brushString = getStorageValue(key)
    if (brushString) {
      return JSON.parse(brushString) as Brush
    } else {
      return null
    }
  } catch (error) {
    console.error('Failed to load brush from cache:', error)
    return null
  }
}

export function useBrushDrawing(initialSettings?: {
  useDominantAxis?: boolean
  brushAdjustmentSpeed?: number
}) {
  const store = useMaskEditorStore()

  const coordinateTransform = useCoordinateTransform()

  // GPU Resources (Scoped to this composable instance)
  let maskTexture: GPUTexture | null = null
  let rgbTexture: GPUTexture | null = null
  let device: GPUDevice | null = null
  let renderer: GPUBrushRenderer | null = null
  let previewContext: GPUCanvasContext | null = null
  let previewCanvas: HTMLCanvasElement | null = null

  // Readback buffers
  let readbackStorageMask: GPUBuffer | null = null
  let readbackStorageRgb: GPUBuffer | null = null
  let readbackStagingMask: GPUBuffer | null = null
  let readbackStagingRgb: GPUBuffer | null = null
  let currentBufferSize = 0

  // Flag to prevent redundant GPU updates
  const isSavingHistory = ref(false)

  // Brush texture cache
  const brushTextureCache = new QuickLRU<string, HTMLCanvasElement>({
    maxSize: 20
  })

  /**
   * Retrieves a cached brush texture or creates a new one if not found.
   * @param radius - The radius of the brush.
   * @param hardness - The hardness of the brush (0 to 1).
   * @param color - The color of the brush.
   * @param opacity - The opacity of the brush (0 to 1).
   * @returns The canvas element containing the brush texture.
   */
  function getCachedBrushTexture(
    radius: number,
    hardness: number,
    color: string,
    opacity: number
  ): HTMLCanvasElement {
    const cacheKey = `${radius}_${hardness}_${color}_${opacity}`

    if (brushTextureCache.has(cacheKey)) {
      return brushTextureCache.get(cacheKey)!
    }

    // Use integer dimensions
    const size = Math.ceil(radius * 2)
    const tempCanvas = document.createElement('canvas')
    const tempCtx = tempCanvas.getContext('2d')!
    tempCanvas.width = size
    tempCanvas.height = size

    const centerX = size / 2
    const centerY = size / 2
    const hardRadius = radius * hardness

    const imageData = tempCtx.createImageData(size, size)
    const data = imageData.data
    const { r, g, b } = parseToRgb(color)

    const fadeRange = radius - hardRadius

    for (let y = 0; y < size; y++) {
      // Calculate distance from pixel center
      const dy = y + 0.5 - centerY
      for (let x = 0; x < size; x++) {
        const dx = x + 0.5 - centerX
        const index = (y * size + x) * 4

        // Calculate Chebyshev distance
        const distFromEdge = Math.max(Math.abs(dx), Math.abs(dy))

        let pixelOpacity = 0
        if (distFromEdge <= hardRadius) {
          pixelOpacity = opacity
        } else if (distFromEdge <= radius) {
          const fadeProgress = (distFromEdge - hardRadius) / fadeRange
          // Apply quadratic falloff
          pixelOpacity = opacity * Math.pow(1 - fadeProgress, 2)
        }

        data[index] = r
        data[index + 1] = g
        data[index + 2] = b
        data[index + 3] = pixelOpacity * 255
      }
    }

    tempCtx.putImageData(imageData, 0, 0)
    brushTextureCache.set(cacheKey, tempCanvas)

    return tempCanvas
  }

  const isDrawing = ref(false)
  const isDrawingLine = ref(false)
  const lineStartPoint = ref<Point | null>(null)
  const lineRemainder = ref(0)
  const smoothingLastDrawTime = ref(new Date())
  const initialDraw = ref(true)

  // Dirty rectangle tracking
  const dirtyRect = ref({
    minX: Infinity,
    minY: Infinity,
    maxX: -Infinity,
    maxY: -Infinity
  })

  /**
   * Resets the dirty rectangle to its initial infinite state.
   */
  function resetDirtyRect() {
    dirtyRect.value = {
      minX: Infinity,
      minY: Infinity,
      maxX: -Infinity,
      maxY: -Infinity
    }
  }

  /**
   * Updates the dirty rectangle to include the specified area.
   * @param x - The x-coordinate of the center.
   * @param y - The y-coordinate of the center.
   * @param radius - The radius of the area.
   */
  function updateDirtyRect(x: number, y: number, radius: number) {
    // Add padding for anti-aliasing
    const padding = 2
    dirtyRect.value.minX = Math.min(dirtyRect.value.minX, x - radius - padding)
    dirtyRect.value.minY = Math.min(dirtyRect.value.minY, y - radius - padding)
    dirtyRect.value.maxX = Math.max(dirtyRect.value.maxX, x + radius + padding)
    dirtyRect.value.maxY = Math.max(dirtyRect.value.maxY, y + radius + padding)
  }

  // Stroke processor instance
  let strokeProcessor: StrokeProcessor | null = null

  const initialPoint = ref<Point | null>(null)
  const useDominantAxis = ref(initialSettings?.useDominantAxis ?? false)
  const brushAdjustmentSpeed = ref(initialSettings?.brushAdjustmentSpeed ?? 1.0)

  const cachedBrushSettings = loadBrushFromCache('maskeditor_brush_settings')
  if (cachedBrushSettings) {
    store.setBrushSize(cachedBrushSettings.size)
    store.setBrushOpacity(cachedBrushSettings.opacity)
    store.setBrushHardness(cachedBrushSettings.hardness)
    store.brushSettings.type = cachedBrushSettings.type
    store.setBrushStepSize(cachedBrushSettings.stepSize ?? 5)
  }

  // Handle external clear events
  watch(
    () => store.clearTrigger,
    () => {
      clearGPU()
    }
  )

  // Sync GPU on Undo/Redo
  watch(
    () => store.canvasHistory.currentStateIndex,
    async () => {
      // Skip update if state was just saved
      if (isSavingHistory.value) return

      // Update GPU textures to match restored canvas state
      await updateGPUFromCanvas()

      // Clear preview to remove artifacts
      if (renderer && previewContext) {
        renderer.clearPreview(previewContext)
      }
    }
  )

  const isRecreatingTextures = ref(false)

  watch(
    () => store.gpuTexturesNeedRecreation,
    async (needsRecreation) => {
      if (
        !needsRecreation ||
        !device ||
        !store.maskCanvas ||
        isRecreatingTextures.value
      )
        return

      isRecreatingTextures.value = true

      const width = store.gpuTextureWidth
      const height = store.gpuTextureHeight

      try {
        // Destroy old textures
        if (maskTexture) {
          maskTexture.destroy()
          maskTexture = null
        }
        if (rgbTexture) {
          rgbTexture.destroy()
          rgbTexture = null
        }

        // Create new textures with updated dimensions
        maskTexture = device.createTexture({
          size: [width, height],
          format: 'rgba8unorm',
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.RENDER_ATTACHMENT |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.COPY_SRC
        })

        rgbTexture = device.createTexture({
          size: [width, height],
          format: 'rgba8unorm',
          usage:
            GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.STORAGE_BINDING |
            GPUTextureUsage.RENDER_ATTACHMENT |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.COPY_SRC
        })

        // Upload pending data if available
        if (store.pendingGPUMaskData && store.pendingGPURgbData) {
          device.queue.writeTexture(
            { texture: maskTexture },
            store.pendingGPUMaskData,
            { bytesPerRow: width * 4 },
            { width, height }
          )

          device.queue.writeTexture(
            { texture: rgbTexture },
            store.pendingGPURgbData,
            { bytesPerRow: width * 4 },
            { width, height }
          )
        } else {
          // Fallback: read from canvas
          await updateGPUFromCanvas()
        }

        // Update preview canvas if it exists
        if (previewCanvas && renderer) {
          previewCanvas.width = width
          previewCanvas.height = height
        }

        // Recreate readback buffers with new size
        const bufferSize = width * height * 4
        if (currentBufferSize !== bufferSize) {
          readbackStorageMask?.destroy()
          readbackStorageRgb?.destroy()
          readbackStagingMask?.destroy()
          readbackStagingRgb?.destroy()

          readbackStorageMask = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
          })
          readbackStorageRgb = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
          })
          readbackStagingMask = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
          })
          readbackStagingRgb = device.createBuffer({
            size: bufferSize,
            usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
          })

          currentBufferSize = bufferSize
        }
      } catch (error) {
        console.error(
          '[useBrushDrawing] Failed to recreate GPU textures:',
          error
        )
      } finally {
        // Clear the recreation flag and pending data
        store.gpuTexturesNeedRecreation = false
        store.gpuTextureWidth = 0
        store.gpuTextureHeight = 0
        store.pendingGPUMaskData = null
        store.pendingGPURgbData = null
        isRecreatingTextures.value = false
      }
    }
  )

  // Cleanup GPU resources on unmount
  onUnmounted(() => {
    if (renderer) {
      renderer.destroy()
      renderer = null
    }
    if (maskTexture) {
      maskTexture.destroy()
      maskTexture = null
    }
    if (rgbTexture) {
      rgbTexture.destroy()
      rgbTexture = null
    }
    if (readbackStorageMask) {
      readbackStorageMask.destroy()
      readbackStorageMask = null
    }
    if (readbackStorageRgb) {
      readbackStorageRgb.destroy()
      readbackStorageRgb = null
    }
    if (readbackStagingMask) {
      readbackStagingMask.destroy()
      readbackStagingMask = null
    }
    if (readbackStagingRgb) {
      readbackStagingRgb.destroy()
      readbackStagingRgb = null
    }
    // We do not destroy the device as it might be shared or managed by TGPU
  })

  /**
   * Initializes the TypeGPU root and device if not already initialized.
   */
  async function initTypeGPU(): Promise<void> {
    if (store.tgpuRoot) {
      device = store.tgpuRoot.device
      return
    }

    try {
      const root = await tgpu.init()
      store.tgpuRoot = root
      device = root.device
      console.warn('✅ TypeGPU initialized! Root:', root)
      console.warn('Device info:', root.device.limits)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      console.warn('Failed to initialize TypeGPU:', message)
    }
  }

  /**
   * Premultiplies the alpha of an ImageData array in place.
   * @param data - The Uint8ClampedArray to modify.
   */
  function premultiplyData(data: Uint8ClampedArray) {
    for (let i = 0; i < data.length; i += 4) {
      const a = data[i + 3] / 255
      data[i] = Math.round(data[i] * a)
      data[i + 1] = Math.round(data[i + 1] * a)
      data[i + 2] = Math.round(data[i + 2] * a)
    }
  }

  /**
   * Updates the GPU textures from the current canvas state.
   */
  async function updateGPUFromCanvas(): Promise<void> {
    if (
      !device ||
      !maskTexture ||
      !rgbTexture ||
      !store.maskCanvas ||
      !store.rgbCtx
    )
      return

    const canvasWidth = store.maskCanvas.width
    const canvasHeight = store.maskCanvas.height

    // Upload canvas data to GPU
    const maskImageData = store.maskCtx!.getImageData(
      0,
      0,
      canvasWidth,
      canvasHeight
    )
    premultiplyData(maskImageData.data)
    device.queue.writeTexture(
      { texture: maskTexture },
      maskImageData.data,
      { bytesPerRow: canvasWidth * 4 },
      { width: canvasWidth, height: canvasHeight }
    )

    const rgbImageData = store.rgbCtx.getImageData(
      0,
      0,
      canvasWidth,
      canvasHeight
    )
    premultiplyData(rgbImageData.data)
    device.queue.writeTexture(
      { texture: rgbTexture },
      rgbImageData.data,
      { bytesPerRow: canvasWidth * 4 },
      { width: canvasWidth, height: canvasHeight }
    )
  }

  /**
   * Initializes all GPU resources including textures and the brush renderer.
   */
  async function initGPUResources(): Promise<void> {
    // Initialize TypeGPU
    await initTypeGPU()

    if (!store.tgpuRoot || !device) {
      console.warn('TypeGPU not initialized, skipping GPU resource setup')
      return
    }

    if (
      !store.maskCanvas ||
      !store.rgbCanvas ||
      !store.maskCtx ||
      !store.rgbCtx
    ) {
      console.warn('Canvas contexts not ready, skipping GPU resource setup')
      return
    }

    const canvasWidth = store.maskCanvas!.width
    const canvasHeight = store.maskCanvas!.height

    try {
      console.warn(
        `🎨 Initializing GPU resources for ${canvasWidth}x${canvasHeight} canvas`
      )

      // Create read/write textures
      maskTexture = device.createTexture({
        size: [canvasWidth, canvasHeight],
        format: 'rgba8unorm',
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.RENDER_ATTACHMENT |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.COPY_SRC
      })

      rgbTexture = device.createTexture({
        size: [canvasWidth, canvasHeight],
        format: 'rgba8unorm',
        usage:
          GPUTextureUsage.TEXTURE_BINDING |
          GPUTextureUsage.STORAGE_BINDING |
          GPUTextureUsage.RENDER_ATTACHMENT |
          GPUTextureUsage.COPY_DST |
          GPUTextureUsage.COPY_SRC
      })

      // Upload initial data
      await updateGPUFromCanvas()

      console.warn('✅ GPU resources initialized successfully')

      const preferredFormat = navigator.gpu.getPreferredCanvasFormat()
      renderer = new GPUBrushRenderer(device!, preferredFormat)
      console.warn('✅ Brush renderer initialized')
    } catch (error) {
      console.error('Failed to initialize GPU resources:', error)
      // Reset to null on failure
      maskTexture = null
      rgbTexture = null
    }
  }

  /**
   * Draws a shape on the appropriate canvas based on the current tool and layer.
   * @param point - The center point of the shape.
   * @param overrideOpacity - Optional opacity override.
   */
  function drawShape(point: Point, overrideOpacity?: number) {
    const brush = store.brushSettings
    const mask_ctx = store.maskCtx
    const rgb_ctx = store.rgbCtx

    if (!mask_ctx || !rgb_ctx) {
      throw new Error('Canvas contexts are required')
    }

    const brushType = brush.type
    const brushRadius = brush.size
    const hardness = brush.hardness
    const opacity = overrideOpacity ?? brush.opacity

    const isErasing = mask_ctx.globalCompositeOperation === 'destination-out'
    const currentTool = store.currentTool
    const isRgbLayer = store.activeLayer === 'rgb'

    if (
      isRgbLayer &&
      currentTool &&
      (currentTool === Tools.Eraser || currentTool === Tools.PaintPen)
    ) {
      // Calculate effective size and hardness
      const effectiveRadius = getEffectiveBrushSize(brushRadius, hardness)
      const effectiveHardness = getEffectiveHardness(
        brushRadius,
        hardness,
        effectiveRadius
      )

      drawRgbShape(
        rgb_ctx,
        point,
        brushType,
        effectiveRadius,
        effectiveHardness,
        opacity
      )
      return
    }

    // Calculate effective size and hardness
    const effectiveRadius = getEffectiveBrushSize(brushRadius, hardness)
    const effectiveHardness = getEffectiveHardness(
      brushRadius,
      hardness,
      effectiveRadius
    )

    drawMaskShape(
      mask_ctx,
      point,
      brushType,
      effectiveRadius,
      effectiveHardness,
      opacity,
      isErasing
    )

    updateDirtyRect(point.x, point.y, effectiveRadius)
  }

  /**
   * Draws a shape on the RGB canvas.
   * @param ctx - The canvas rendering context.
   * @param point - The center point.
   * @param brushType - The type of brush (circle/rect).
   * @param brushRadius - The radius of the brush.
   * @param hardness - The hardness of the brush.
   * @param opacity - The opacity of the brush.
   */
  function drawRgbShape(
    ctx: CanvasRenderingContext2D,
    point: Point,
    brushType: BrushShape,
    brushRadius: number,
    hardness: number,
    opacity: number
  ): void {
    const { x, y } = point
    const rgbColor = store.rgbColor

    if (brushType === BrushShape.Rect && hardness < 1) {
      const rgbaColor = formatRgba(rgbColor, opacity)
      const brushTexture = getCachedBrushTexture(
        brushRadius,
        hardness,
        rgbaColor,
        opacity
      )
      ctx.drawImage(brushTexture, x - brushRadius, y - brushRadius)
      updateDirtyRect(x, y, brushRadius)
      return
    }

    if (hardness === 1) {
      const rgbaColor = formatRgba(rgbColor, opacity)
      ctx.fillStyle = rgbaColor
      drawShapeOnContext(ctx, brushType, x, y, brushRadius)
      updateDirtyRect(x, y, brushRadius)
      return
    }

    const gradient = createBrushGradient(
      ctx,
      x,
      y,
      brushRadius,
      hardness,
      rgbColor,
      opacity,
      false
    )
    ctx.fillStyle = gradient
    drawShapeOnContext(ctx, brushType, x, y, brushRadius)
    updateDirtyRect(x, y, brushRadius)
  }

  /**
   * Draws a shape on the Mask canvas.
   * @param ctx - The canvas rendering context.
   * @param point - The center point.
   * @param brushType - The type of brush (circle/rect).
   * @param brushRadius - The radius of the brush.
   * @param hardness - The hardness of the brush.
   * @param opacity - The opacity of the brush.
   * @param isErasing - Whether the operation is erasing.
   */
  function drawMaskShape(
    ctx: CanvasRenderingContext2D,
    point: Point,
    brushType: BrushShape,
    brushRadius: number,
    hardness: number,
    opacity: number,
    isErasing: boolean
  ): void {
    const { x, y } = point
    const maskColor = store.maskColor

    if (brushType === BrushShape.Rect && hardness < 1) {
      const baseColor = isErasing
        ? `rgba(255, 255, 255, ${opacity})`
        : `rgba(${maskColor.r}, ${maskColor.g}, ${maskColor.b}, ${opacity})`

      const brushTexture = getCachedBrushTexture(
        brushRadius,
        hardness,
        baseColor,
        opacity
      )
      ctx.drawImage(brushTexture, x - brushRadius, y - brushRadius)
      updateDirtyRect(x, y, brushRadius)
      return
    }

    if (hardness === 1) {
      ctx.fillStyle = isErasing
        ? `rgba(255, 255, 255, ${opacity})`
        : `rgba(${maskColor.r}, ${maskColor.g}, ${maskColor.b}, ${opacity})`
      drawShapeOnContext(ctx, brushType, x, y, brushRadius)
      updateDirtyRect(x, y, brushRadius)
      return
    }

    const maskColorHex = `rgb(${maskColor.r}, ${maskColor.g}, ${maskColor.b})`
    const gradient = createBrushGradient(
      ctx,
      x,
      y,
      brushRadius,
      hardness,
      maskColorHex,
      opacity,
      isErasing
    )
    ctx.fillStyle = gradient
    drawShapeOnContext(ctx, brushType, x, y, brushRadius)
    updateDirtyRect(x, y, brushRadius)
  }

  /**
   * Helper to draw the path of the shape on the context.
   * @param ctx - The canvas rendering context.
   * @param brushType - The type of brush.
   * @param x - Center x.
   * @param y - Center y.
   * @param radius - Radius.
   */
  function drawShapeOnContext(
    ctx: CanvasRenderingContext2D,
    brushType: BrushShape,
    x: number,
    y: number,
    radius: number
  ): void {
    ctx.beginPath()
    if (brushType === BrushShape.Rect) {
      ctx.rect(x - radius, y - radius, radius * 2, radius * 2)
    } else {
      ctx.arc(x, y, radius, 0, Math.PI * 2, false)
    }
    ctx.fill()
  }

  /**
   * Formats a hex color and alpha into an rgba string.
   * @param hex - The hex color string.
   * @param alpha - The alpha value (0-1).
   * @returns The rgba string.
   */
  function formatRgba(hex: string, alpha: number): string {
    const { r, g, b } = hexToRgb(hex)
    return `rgba(${r}, ${g}, ${b}, ${alpha})`
  }

  /**
   * Creates a radial gradient for soft brushes.
   * @param ctx - The canvas context.
   * @param x - Center x.
   * @param y - Center y.
   * @param radius - Radius.
   * @param hardness - Hardness (0-1).
   * @param color - Color string.
   * @param opacity - Opacity (0-1).
   * @param isErasing - Whether erasing.
   * @returns The canvas gradient.
   */
  function createBrushGradient(
    ctx: CanvasRenderingContext2D,
    x: number,
    y: number,
    radius: number,
    hardness: number,
    color: string,
    opacity: number,
    isErasing: boolean
  ): CanvasGradient {
    if (
      !Number.isFinite(x) ||
      !Number.isFinite(y) ||
      !Number.isFinite(radius)
    ) {
      return ctx.createRadialGradient(0, 0, 0, 0, 0, 0)
    }

    const gradient = ctx.createRadialGradient(x, y, 0, x, y, radius)

    if (isErasing) {
      gradient.addColorStop(0, `rgba(255, 255, 255, ${opacity})`)
      gradient.addColorStop(hardness, `rgba(255, 255, 255, ${opacity})`)
      gradient.addColorStop(1, `rgba(255, 255, 255, 0)`)
    } else {
      const { r, g, b } = parseToRgb(color)
      gradient.addColorStop(0, `rgba(${r}, ${g}, ${b}, ${opacity})`)
      gradient.addColorStop(hardness, `rgba(${r}, ${g}, ${b}, ${opacity})`)
      gradient.addColorStop(1, `rgba(${r}, ${g}, ${b}, 0)`)
    }

    return gradient
  }

  /**
   * Draws a point using the stroke processor for smoothing.
   * @param point - The point to draw.
   */
  async function drawWithBetterSmoothing(point: Point): Promise<void> {
    if (!strokeProcessor) return

    // Process point to generate equidistant points
    const newPoints = strokeProcessor.addPoint(point)

    if (newPoints.length === 0) {
      // Update preview even if no points generated to ensure background visibility
      if (renderer && initialDraw.value) {
        gpuRender([], true)
      }
      return
    }

    // Render points on GPU
    if (renderer) {
      gpuRender(newPoints, true) // Skip resampling
    } else {
      // CPU fallback
      for (const p of newPoints) {
        drawShape(p)
      }
    }

    initialDraw.value = false
  }

  /**
   * Initializes the canvas context for a new shape.
   * @param compositionOperation - The composition operation to use.
   */
  function initShape(compositionOperation: CompositionOperation) {
    const blendMode = store.maskBlendMode
    const mask_ctx = store.maskCtx
    const rgb_ctx = store.rgbCtx

    if (!mask_ctx || !rgb_ctx) {
      throw new Error('Canvas contexts are required')
    }

    mask_ctx.beginPath()
    rgb_ctx.beginPath()

    if (compositionOperation === CompositionOperation.SourceOver) {
      mask_ctx.fillStyle = blendMode
      mask_ctx.globalCompositeOperation = CompositionOperation.SourceOver
      rgb_ctx.globalCompositeOperation = CompositionOperation.SourceOver
    } else if (compositionOperation === CompositionOperation.DestinationOut) {
      mask_ctx.globalCompositeOperation = CompositionOperation.DestinationOut
      rgb_ctx.globalCompositeOperation = CompositionOperation.DestinationOut
    }
  }

  /**
   * Draws a line between two points.
   * @param p1 - Start point.
   * @param p2 - End point.
   * @param compositionOp - Composition operation.
   * @param spacing - Spacing between points.
   */
  async function drawLine(
    p1: Point,
    p2: Point,
    compositionOp: CompositionOperation,
    spacing: number
  ): Promise<void> {
    // Generate equidistant points using segment resampling
    const { points, remainder } = resampleSegment(
      [p1, p2],
      spacing,
      lineRemainder.value
    )

    lineRemainder.value = remainder

    // Ensure context state is initialized (sets globalCompositeOperation for isErasing checks)
    initShape(compositionOp)

    if (renderer) {
      gpuRender(points)
    } else {
      // CPU fallback
      for (const point of points) {
        drawShape(point, 1) // Opacity handled by brush texture
      }
    }
  }

  /**
   * Starts the drawing process.
   * @param event - The pointer event.
   */
  async function startDrawing(event: PointerEvent): Promise<void> {
    isDrawing.value = true
    resetDirtyRect()

    try {
      // Initialize stroke accumulator
      if (renderer && store.maskCanvas) {
        renderer.prepareStroke(store.maskCanvas.width, store.maskCanvas.height)
      }

      let compositionOp: CompositionOperation
      const currentTool = store.currentTool
      const coords = { x: event.offsetX, y: event.offsetY }
      const coords_canvas = coordinateTransform.screenToCanvas(coords)

      if (currentTool === 'eraser' || event.buttons === 2) {
        compositionOp = CompositionOperation.DestinationOut
      } else {
        compositionOp = CompositionOperation.SourceOver
      }

      // Calculate target spacing based on step size percentage
      const stepPercentage = store.brushSettings.stepSize / 100
      const targetSpacing = Math.max(
        1.0,
        store.brushSettings.size * stepPercentage
      )

      if (event.shiftKey && lineStartPoint.value) {
        isDrawingLine.value = true
        await drawLine(
          lineStartPoint.value,
          coords_canvas,
          compositionOp,
          targetSpacing
        )
      } else {
        isDrawingLine.value = false
        initShape(compositionOp)
        // Reset remainder
        lineRemainder.value = targetSpacing
      }

      lineStartPoint.value = coords_canvas

      // Hide main canvas to prevent double rendering
      if (renderer) {
        const isRgb = store.activeLayer === 'rgb'
        if (isRgb && store.rgbCanvas) {
          store.rgbCanvas.style.opacity = '0'
          if (previewCanvas) previewCanvas.style.opacity = '1'
        } else if (!isRgb && store.maskCanvas) {
          store.maskCanvas.style.opacity = '0'
          if (previewCanvas)
            previewCanvas.style.opacity = String(store.maskOpacity)
        }
      }

      // Initialize stroke processor
      strokeProcessor = new StrokeProcessor(targetSpacing)

      // Process first point
      await drawWithBetterSmoothing(coords_canvas)

      smoothingLastDrawTime.value = new Date()
    } catch (error) {
      console.error('[useBrushDrawing] Failed to start drawing:', error)

      isDrawing.value = false
      isDrawingLine.value = false
    }
  }

  /**
   * Handles the drawing movement.
   * @param event - The pointer event.
   */
  async function handleDrawing(event: PointerEvent): Promise<void> {
    const diff = performance.now() - smoothingLastDrawTime.value.getTime()
    const coords = { x: event.offsetX, y: event.offsetY }
    const coords_canvas = coordinateTransform.screenToCanvas(coords)
    const currentTool = store.currentTool

    if (diff > 20 && !isDrawing.value) {
      requestAnimationFrame(async () => {
        if (!isDrawing.value) return // Fix: Prevent race condition
        try {
          initShape(CompositionOperation.SourceOver)
          await gpuDrawPoint(coords_canvas)
          // smoothingCordsArray.value.push(coords_canvas) // Removed in favor of StrokeProcessor
        } catch (error) {
          console.error('[useBrushDrawing] Drawing error:', error)
        }
      })
    } else {
      requestAnimationFrame(async () => {
        if (!isDrawing.value) return // Fix: Prevent race condition
        try {
          if (currentTool === 'eraser' || event.buttons === 2) {
            initShape(CompositionOperation.DestinationOut)
          } else {
            initShape(CompositionOperation.SourceOver)
          }
          await drawWithBetterSmoothing(coords_canvas)
        } catch (error) {
          console.error('[useBrushDrawing] Drawing error:', error)
        }
      })
    }

    smoothingLastDrawTime.value = new Date()
  }

  /**
   * Ends the drawing process.
   * @param event - The pointer event.
   */
  async function drawEnd(event: PointerEvent): Promise<void> {
    const coords = { x: event.offsetX, y: event.offsetY }
    const coords_canvas = coordinateTransform.screenToCanvas(coords)

    if (isDrawing.value) {
      isDrawing.value = false

      lineStartPoint.value = coords_canvas
      initialDraw.value = true

      // Flush remaining points from StrokeProcessor
      if (strokeProcessor) {
        const finalPoints = strokeProcessor.endStroke()
        if (finalPoints.length > 0) {
          if (renderer) {
            gpuRender(finalPoints, true)
          } else {
            for (const p of finalPoints) {
              drawShape(p)
            }
          }
        }
        strokeProcessor = null
      }

      // Composite the stroke accumulator into the main texture
      if (renderer && maskTexture && rgbTexture) {
        const isRgb = store.activeLayer === 'rgb'
        const targetTex = isRgb ? rgbTexture : maskTexture

        // Use the actual brush opacity for the composite pass
        const size = store.brushSettings.size
        const hardness = store.brushSettings.hardness
        const effectiveSize = getEffectiveBrushSize(size, hardness)
        const effectiveHardness = getEffectiveHardness(
          size,
          hardness,
          effectiveSize
        )

        const isErasing =
          store.currentTool === 'eraser' ||
          store.maskCtx?.globalCompositeOperation === 'destination-out'

        const brushShape = store.brushSettings.type === BrushShape.Rect ? 1 : 0

        renderer.compositeStroke(targetTex.createView(), {
          opacity: store.brushSettings.opacity,
          color: [0, 0, 0], // Color is handled by accumulator, this is just for uniforms if needed
          hardness: effectiveHardness,
          screenSize: [store.maskCanvas!.width, store.maskCanvas!.height],
          brushShape,
          isErasing
        })
      }

      let maskData: ImageData | undefined
      let rgbData: ImageData | undefined

      if (renderer && maskTexture && rgbTexture) {
        try {
          const result = await copyGpuToCanvas()
          maskData = result.maskData
          rgbData = result.rgbData
        } catch (error) {
          console.warn('GPU readback failed, falling back to CPU:', error)
        }
      }

      isSavingHistory.value = true
      store.canvasHistory.saveState(maskData, rgbData)
      // Wait for watcher to trigger (if any) before clearing flag
      await nextTick()

      isSavingHistory.value = false

      // Fix: Clear the preview canvas when drawing ends
      if (renderer && previewContext) {
        renderer.clearPreview(previewContext)
      }

      // Restore main canvas visibility
      if (store.activeLayer === 'rgb' && store.rgbCanvas) {
        store.rgbCanvas.style.opacity = '1'
      } else if (store.activeLayer === 'mask' && store.maskCanvas) {
        store.maskCanvas.style.opacity = String(store.maskOpacity)
      }

      // Reset preview canvas opacity to 1 (for hover preview)
      if (previewCanvas) {
        previewCanvas.style.opacity = '1'
      }
    }
  }

  /**
   * Starts the brush adjustment interaction.
   * @param event - The pointer event.
   */
  async function startBrushAdjustment(event: PointerEvent): Promise<void> {
    event.preventDefault()

    const coords = { x: event.offsetX, y: event.offsetY }
    const coords_canvas = coordinateTransform.screenToCanvas(coords)

    store.brushPreviewGradientVisible = true
    initialPoint.value = coords_canvas
  }

  /**
   * Handles the brush adjustment movement.
   * @param event - The pointer event.
   */
  async function handleBrushAdjustment(event: PointerEvent): Promise<void> {
    if (!initialPoint.value) {
      return
    }

    const coords = { x: event.offsetX, y: event.offsetY }
    const brushDeadZone = 5
    const coords_canvas = coordinateTransform.screenToCanvas(coords)

    const delta_x = coords_canvas.x - initialPoint.value.x
    const delta_y = coords_canvas.y - initialPoint.value.y

    const effectiveDeltaX = Math.abs(delta_x) < brushDeadZone ? 0 : delta_x
    const effectiveDeltaY = Math.abs(delta_y) < brushDeadZone ? 0 : delta_y

    let finalDeltaX = effectiveDeltaX
    let finalDeltaY = effectiveDeltaY

    if (useDominantAxis.value) {
      const ratio = Math.abs(effectiveDeltaX) / Math.abs(effectiveDeltaY)
      const threshold = 2.0

      if (ratio > threshold) {
        finalDeltaY = 0
      } else if (ratio < 1 / threshold) {
        finalDeltaX = 0
      }
    }

    const cappedDeltaX = Math.max(-100, Math.min(100, finalDeltaX))
    const cappedDeltaY = Math.max(-100, Math.min(100, finalDeltaY))

    const newSize = Math.max(
      1,
      Math.min(
        500,
        store.brushSettings.size +
          (cappedDeltaX / 35) * brushAdjustmentSpeed.value
      )
    )

    const newHardness = Math.max(
      0,
      Math.min(
        1,
        store.brushSettings.hardness -
          (cappedDeltaY / 4000) * brushAdjustmentSpeed.value
      )
    )

    store.setBrushSize(newSize)
    store.setBrushHardness(newHardness)
  }

  /**
   * Saves the current brush settings to cache.
   */
  function saveBrushSettings(): void {
    saveBrushToCache('maskeditor_brush_settings', store.brushSettings)
  }

  /**
   * Reads back the GPU textures to CPU ImageDatas.
   * @returns Object containing mask and rgb ImageDatas.
   */
  async function copyGpuToCanvas(): Promise<{
    maskData: ImageData
    rgbData: ImageData
  }> {
    if (
      !device ||
      !maskTexture ||
      !rgbTexture ||
      !store.maskCanvas ||
      !store.rgbCanvas ||
      !store.maskCtx ||
      !store.rgbCtx ||
      !renderer
    )
      throw new Error('GPU resources not ready')

    const width = store.maskCanvas.width
    const height = store.maskCanvas.height
    const bufferSize = width * height * 4

    // 1. Initialize/Resize Buffers if needed
    if (
      !readbackStorageMask ||
      !readbackStorageRgb ||
      !readbackStagingMask ||
      !readbackStagingRgb ||
      currentBufferSize !== bufferSize
    ) {
      // Destroy old buffers if they exist
      readbackStorageMask?.destroy()
      readbackStorageRgb?.destroy()
      readbackStagingMask?.destroy()
      readbackStagingRgb?.destroy()

      // Create Storage Buffers (for compute shader output)
      readbackStorageMask = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      })
      readbackStorageRgb = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
      })

      // Create Staging Buffers (for reading back)
      readbackStagingMask = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      })
      readbackStagingRgb = device.createBuffer({
        size: bufferSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
      })

      currentBufferSize = bufferSize
    }

    // 2. Run Compute Shaders (Un-premultiply and pack)
    renderer.prepareReadback(maskTexture, readbackStorageMask)
    renderer.prepareReadback(rgbTexture, readbackStorageRgb)

    // 3. Copy Storage -> Staging
    const encoder = device.createCommandEncoder()
    encoder.copyBufferToBuffer(
      readbackStorageMask,
      0,
      readbackStagingMask,
      0,
      bufferSize
    )
    encoder.copyBufferToBuffer(
      readbackStorageRgb,
      0,
      readbackStagingRgb,
      0,
      bufferSize
    )
    device.queue.submit([encoder.finish()])

    // 4. Map Staging Buffers
    await Promise.all([
      readbackStagingMask.mapAsync(GPUMapMode.READ),
      readbackStagingRgb.mapAsync(GPUMapMode.READ)
    ])

    // 5. Read Data & Update Canvas
    // We use slice(0) to copy data because unmap() invalidates the array
    const maskDataArr = new Uint8ClampedArray(
      readbackStagingMask.getMappedRange().slice(0)
    )
    const rgbDataArr = new Uint8ClampedArray(
      readbackStagingRgb.getMappedRange().slice(0)
    )

    // Unmap immediately after copying
    readbackStagingMask.unmap()
    readbackStagingRgb.unmap()

    const maskImageData = new ImageData(maskDataArr, width, height)
    const rgbImageData = new ImageData(rgbDataArr, width, height)

    // Calculate Dirty Rect
    let dx = 0
    let dy = 0
    let dw = width
    let dh = height

    if (
      dirtyRect.value.minX !== Infinity &&
      dirtyRect.value.maxX !== -Infinity
    ) {
      const r = dirtyRect.value
      dx = Math.floor(Math.max(0, r.minX))
      dy = Math.floor(Math.max(0, r.minY))
      const max_x = Math.ceil(Math.min(width, r.maxX))
      const max_y = Math.ceil(Math.min(height, r.maxY))
      dw = max_x - dx
      dh = max_y - dy
    }

    // Ensure valid dimensions
    if (dw > 0 && dh > 0) {
      store.maskCtx.putImageData(maskImageData, 0, 0, dx, dy, dw, dh)
      store.rgbCtx.putImageData(rgbImageData, 0, 0, dx, dy, dw, dh)
    } else {
      // Fallback to full update if rect is invalid (shouldn't happen if drawn)
      store.maskCtx.putImageData(maskImageData, 0, 0)
      store.rgbCtx.putImageData(rgbImageData, 0, 0)
    }

    return { maskData: maskImageData, rgbData: rgbImageData }
  }

  /**
   * Cleans up GPU resources and buffers.
   */
  function destroy(): void {
    renderer?.destroy()
    if (maskTexture) {
      maskTexture.destroy()
      maskTexture = null
    }
    if (rgbTexture) {
      rgbTexture.destroy()
      rgbTexture = null
    }
    // Cleanup Readback Buffers
    readbackStorageMask?.destroy()
    readbackStorageRgb?.destroy()
    readbackStagingMask?.destroy()
    readbackStagingRgb?.destroy()
    readbackStorageMask = null
    readbackStorageRgb = null
    readbackStagingMask = null
    readbackStagingRgb = null
    currentBufferSize = 0

    if (store.tgpuRoot) {
      store.tgpuRoot.destroy()
      store.tgpuRoot = null
    }
    device = null
  }

  /**
   * Draws a single point using the GPU renderer.
   * @param point - The point to draw.
   * @param opacity - The opacity of the point.
   */
  async function gpuDrawPoint(point: Point, opacity: number = 1) {
    if (renderer) {
      const width = store.maskCanvas!.width
      const height = store.maskCanvas!.height
      const strokePoints = [{ x: point.x, y: point.y, pressure: opacity }]

      const size = store.brushSettings.size
      const hardness = store.brushSettings.hardness
      const effectiveSize = getEffectiveBrushSize(size, hardness)
      const effectiveHardness = getEffectiveHardness(
        size,
        hardness,
        effectiveSize
      )

      const brushShape = store.brushSettings.type === BrushShape.Rect ? 1 : 0

      // Use accumulator with fixed high opacity to build shape
      renderer.renderStrokeToAccumulator(strokePoints, {
        size: effectiveSize,
        opacity: 0.5, // Fixed flow for smooth accumulation
        hardness: effectiveHardness,
        color: [1, 1, 1],
        width,
        height,
        brushShape
      })

      // Update preview with correct settings
      if (maskTexture && previewContext) {
        const isRgb = store.activeLayer === 'rgb'
        let color: [number, number, number] = [1, 1, 1]
        if (isRgb) {
          const c = parseToRgb(store.rgbColor)
          color = [c.r / 255, c.g / 255, c.b / 255]
        } else {
          const c = store.maskColor as { r: number; g: number; b: number }
          color = [c.r / 255, c.g / 255, c.b / 255]
        }

        const isErasing =
          store.currentTool === 'eraser' ||
          store.maskCtx?.globalCompositeOperation === 'destination-out'

        renderer.blitToCanvas(
          previewContext,
          {
            opacity: store.brushSettings.opacity,
            color,
            hardness: effectiveHardness,
            screenSize: [width, height],
            brushShape,
            isErasing
          },
          undefined // Do not draw background texture for preview to avoid double rendering
        )
      }
    } else {
      drawShape(point, opacity)
    }
  }

  /**
   * Initializes the preview canvas context for WebGPU.
   * @param canvas - The canvas element.
   */
  function initPreviewCanvas(canvas: HTMLCanvasElement) {
    if (!device) return

    const ctx = canvas.getContext('webgpu')
    if (!ctx) return

    ctx.configure({
      device,
      format: navigator.gpu.getPreferredCanvasFormat(),
      alphaMode: 'premultiplied'
    })
    previewContext = ctx
    previewCanvas = canvas
    console.warn('✅ Preview Canvas Initialized')
  }

  /**
   * Renders a set of points using the GPU.
   * @param points - The points to render.
   * @param skipResampling - Whether to skip resampling (if points are already spaced).
   */
  function gpuRender(points: Point[], skipResampling: boolean = false) {
    if (!renderer || !maskTexture || !rgbTexture) return

    const isRgb = store.activeLayer === 'rgb'

    // 1. Get Correct Color
    let color: [number, number, number] = [1, 1, 1]
    if (isRgb) {
      const c = parseToRgb(store.rgbColor)
      color = [c.r / 255, c.g / 255, c.b / 255]
    } else {
      // Mask color - properly typed
      const c = store.maskColor as { r: number; g: number; b: number }
      color = [c.r / 255, c.g / 255, c.b / 255]
    }

    // 2. Prepare stroke points
    let strokePoints: { x: number; y: number; pressure: number }[] = []

    if (skipResampling) {
      // Points are already properly spaced from Catmull-Rom spline interpolation
      strokePoints = points.map((p) => ({ x: p.x, y: p.y, pressure: 1.0 }))
    } else {
      // Legacy resampling for shift+click and other cases
      for (let i = 0; i < points.length - 1; i++) {
        const p1 = points[i]
        const p2 = points[i + 1]
        const dist = Math.hypot(p2.x - p1.x, p2.y - p1.y)

        // Calculate target spacing based on stepSize
        const stepPercentage = store.brushSettings.stepSize / 100
        const stepSize = Math.max(
          1.0,
          store.brushSettings.size * stepPercentage
        )
        const steps = Math.max(1, Math.ceil(dist / stepSize))

        for (let step = 0; step <= steps; step++) {
          const t = step / steps
          const pressure = 1.0

          const x = p1.x + (p2.x - p1.x) * t
          const y = p1.y + (p2.y - p1.y) * t

          strokePoints.push({ x, y, pressure })
        }
      }
    }

    // Render to Accumulator (SourceOver blending)
    // Use fixed opacity (0.5) to build up the shape smoothly without creases.
    // The final opacity is applied in the composite pass.
    const size = store.brushSettings.size
    const hardness = store.brushSettings.hardness
    const effectiveSize = getEffectiveBrushSize(size, hardness)
    const effectiveHardness = getEffectiveHardness(
      size,
      hardness,
      effectiveSize
    )

    const brushShape = store.brushSettings.type === BrushShape.Rect ? 1 : 0

    // Render to Accumulator (SourceOver blending)
    // Use fixed opacity (0.5) to build up the shape smoothly without creases.
    // The final opacity is applied in the composite pass.
    renderer.renderStrokeToAccumulator(strokePoints, {
      size: effectiveSize,
      opacity: 0.5,
      hardness: effectiveHardness,
      color: color,
      width: store.maskCanvas!.width,
      height: store.maskCanvas!.height,
      brushShape
    })

    // Update Dirty Rect
    for (const p of strokePoints) {
      updateDirtyRect(p.x, p.y, effectiveSize)
    }

    // 3. Blit to Preview with correct settings
    if (previewContext) {
      const isErasing =
        store.currentTool === 'eraser' ||
        store.maskCtx?.globalCompositeOperation === 'destination-out'

      const targetTex = isRgb ? rgbTexture : maskTexture
      renderer.blitToCanvas(
        previewContext,
        {
          opacity: store.brushSettings.opacity,
          color,
          hardness: effectiveHardness,
          screenSize: [store.maskCanvas!.width, store.maskCanvas!.height],
          brushShape,
          isErasing
        },
        targetTex ?? undefined
      )
    }
  }

  /**
   * Clears the GPU textures.
   */
  function clearGPU() {
    if (!device || !maskTexture || !rgbTexture || !store.maskCanvas) return

    const width = store.maskCanvas.width
    const height = store.maskCanvas.height

    // Clear Mask Texture
    device.queue.writeTexture(
      { texture: maskTexture },
      new Uint8Array(width * height * 4), // Zeros
      { bytesPerRow: width * 4 },
      { width, height }
    )

    // Clear RGB Texture
    device.queue.writeTexture(
      { texture: rgbTexture },
      new Uint8Array(width * height * 4), // Zeros
      { bytesPerRow: width * 4 },
      { width, height }
    )
  }

  return {
    startDrawing,
    handleDrawing,
    drawEnd,
    startBrushAdjustment,
    handleBrushAdjustment,
    saveBrushSettings,
    destroy,
    initGPUResources,
    initPreviewCanvas,
    clearGPU // Export this
  }
}
