import type { Ref } from 'vue'
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { useElementSize } from '@vueuse/core'
import { useI18n } from 'vue-i18n'

import {
  getEffectiveBrushSize,
  getEffectiveHardness
} from '@/composables/maskeditor/brushUtils'
import { StrokeProcessor } from '@/composables/maskeditor/StrokeProcessor'
import { hexToRgb } from '@/utils/colorUtil'
import type { Point } from '@/extensions/core/maskeditor/types'
import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'
import { isCloud } from '@/platform/distribution/types'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { api } from '@/scripts/api'
import { app } from '@/scripts/app'
import { useNodeOutputStore } from '@/stores/imagePreviewStore'

type PainterTool = 'brush' | 'eraser'

export const PAINTER_TOOLS: Record<string, PainterTool> = {
  BRUSH: 'brush',
  ERASER: 'eraser'
} as const

interface UsePainterOptions {
  canvasEl: Ref<HTMLCanvasElement | null>
  modelValue: Ref<string>
}

export function usePainter(nodeId: string, options: UsePainterOptions) {
  const { canvasEl, modelValue } = options
  const { t } = useI18n()
  const nodeOutputStore = useNodeOutputStore()
  const toastStore = useToastStore()

  const isDirty = ref(false)

  const canvasWidth = ref(512)
  const canvasHeight = ref(512)

  const cursorX = ref(0)
  const cursorY = ref(0)
  const cursorVisible = ref(false)

  const inputImageUrl = ref<string | null>(null)
  const isImageInputConnected = ref(false)

  let isDrawing = false
  let strokeProcessor: StrokeProcessor | null = null
  let lastPoint: Point | null = null

  let cachedRect: DOMRect | null = null

  let mainCtx: CanvasRenderingContext2D | null = null

  let strokeCanvas: HTMLCanvasElement | null = null
  let strokeCtx: CanvasRenderingContext2D | null = null

  let baseCanvas: HTMLCanvasElement | null = null
  let baseCtx: CanvasRenderingContext2D | null = null
  let hasBaseSnapshot = false
  let hasStrokes = false

  let dirtyX0 = 0
  let dirtyY0 = 0
  let dirtyX1 = 0
  let dirtyY1 = 0
  let hasDirtyRect = false

  let strokeBrush: {
    radius: number
    effectiveRadius: number
    effectiveHardness: number
    hardness: number
    r: number
    g: number
    b: number
  } | null = null

  const litegraphNode = computed(() => {
    if (!nodeId || !app.canvas.graph) return null
    return app.canvas.graph.getNodeById(nodeId) as LGraphNode | null
  })

  function getWidgetByName(name: string): IBaseWidget | undefined {
    return litegraphNode.value?.widgets?.find(
      (w: IBaseWidget) => w.name === name
    )
  }

  const tool = ref<PainterTool>(PAINTER_TOOLS.BRUSH)
  const brushSize = ref(20)
  const brushColor = ref('#ffffff')
  const brushOpacity = ref(1)
  const brushHardness = ref(1)
  const backgroundColor = ref('#000000')

  function restoreSettingsFromProperties() {
    const node = litegraphNode.value
    if (!node) return

    const props = node.properties
    if (props.painterTool != null) tool.value = props.painterTool as PainterTool
    if (props.painterBrushSize != null)
      brushSize.value = props.painterBrushSize as number
    if (props.painterBrushColor != null)
      brushColor.value = props.painterBrushColor as string
    if (props.painterBrushOpacity != null)
      brushOpacity.value = props.painterBrushOpacity as number
    if (props.painterBrushHardness != null)
      brushHardness.value = props.painterBrushHardness as number

    const bgColorWidget = getWidgetByName('bg_color')
    if (bgColorWidget) backgroundColor.value = bgColorWidget.value as string
  }

  function saveSettingsToProperties() {
    const node = litegraphNode.value
    if (!node) return

    node.properties.painterTool = tool.value
    node.properties.painterBrushSize = brushSize.value
    node.properties.painterBrushColor = brushColor.value
    node.properties.painterBrushOpacity = brushOpacity.value
    node.properties.painterBrushHardness = brushHardness.value
  }

  function syncCanvasSizeToWidgets() {
    const widthWidget = getWidgetByName('width')
    const heightWidget = getWidgetByName('height')

    if (widthWidget && widthWidget.value !== canvasWidth.value) {
      widthWidget.value = canvasWidth.value
      widthWidget.callback?.(canvasWidth.value)
    }
    if (heightWidget && heightWidget.value !== canvasHeight.value) {
      heightWidget.value = canvasHeight.value
      heightWidget.callback?.(canvasHeight.value)
    }
  }

  function syncBackgroundColorToWidget() {
    const bgColorWidget = getWidgetByName('bg_color')
    if (bgColorWidget && bgColorWidget.value !== backgroundColor.value) {
      bgColorWidget.value = backgroundColor.value
      bgColorWidget.callback?.(backgroundColor.value)
    }
  }

  function updateInputImageUrl() {
    const node = litegraphNode.value
    if (!node) {
      inputImageUrl.value = null
      isImageInputConnected.value = false
      return
    }

    isImageInputConnected.value = node.isInputConnected(0)

    const inputNode = node.getInputNode(0)
    if (!inputNode) {
      inputImageUrl.value = null
      return
    }

    const urls = nodeOutputStore.getNodeImageUrls(inputNode)
    inputImageUrl.value = urls?.length ? urls[0] : null
  }

  function syncCanvasSizeFromWidgets() {
    const w = getWidgetByName('width')
    const h = getWidgetByName('height')
    canvasWidth.value = (w?.value as number) ?? 512
    canvasHeight.value = (h?.value as number) ?? 512
  }

  function activeHardness(): number {
    return tool.value === PAINTER_TOOLS.ERASER ? 1 : brushHardness.value
  }

  const { width: canvasDisplayWidth } = useElementSize(canvasEl)

  const displayBrushSize = computed(() => {
    if (!canvasDisplayWidth.value || !canvasWidth.value) return brushSize.value

    const radius = brushSize.value / 2
    const effectiveRadius = getEffectiveBrushSize(radius, activeHardness())
    const effectiveDiameter = effectiveRadius * 2
    return effectiveDiameter * (canvasDisplayWidth.value / canvasWidth.value)
  })

  function getCtx() {
    if (!mainCtx && canvasEl.value) {
      mainCtx = canvasEl.value.getContext('2d') ?? null
    }
    return mainCtx
  }

  function cacheCanvasRect() {
    const el = canvasEl.value
    if (el) cachedRect = el.getBoundingClientRect()
  }

  function getCanvasPoint(e: PointerEvent): Point | null {
    const el = canvasEl.value
    if (!el) return null
    const rect = cachedRect ?? el.getBoundingClientRect()
    return {
      x: ((e.clientX - rect.left) / rect.width) * el.width,
      y: ((e.clientY - rect.top) / rect.height) * el.height
    }
  }

  function expandDirtyRect(cx: number, cy: number, r: number) {
    const x0 = cx - r
    const y0 = cy - r
    const x1 = cx + r
    const y1 = cy + r
    if (!hasDirtyRect) {
      dirtyX0 = x0
      dirtyY0 = y0
      dirtyX1 = x1
      dirtyY1 = y1
      hasDirtyRect = true
    } else {
      if (x0 < dirtyX0) dirtyX0 = x0
      if (y0 < dirtyY0) dirtyY0 = y0
      if (x1 > dirtyX1) dirtyX1 = x1
      if (y1 > dirtyY1) dirtyY1 = y1
    }
  }

  function snapshotBrush() {
    const radius = brushSize.value / 2
    const hardness = activeHardness()
    const effectiveRadius = getEffectiveBrushSize(radius, hardness)
    strokeBrush = {
      radius,
      effectiveRadius,
      effectiveHardness: getEffectiveHardness(
        radius,
        hardness,
        effectiveRadius
      ),
      hardness,
      ...hexToRgb(brushColor.value)
    }
  }

  function drawCircle(ctx: CanvasRenderingContext2D, point: Point) {
    const b = strokeBrush!

    expandDirtyRect(point.x, point.y, b.effectiveRadius)

    ctx.beginPath()
    ctx.arc(point.x, point.y, b.effectiveRadius, 0, Math.PI * 2)

    if (b.hardness < 1) {
      const gradient = ctx.createRadialGradient(
        point.x,
        point.y,
        0,
        point.x,
        point.y,
        b.effectiveRadius
      )
      gradient.addColorStop(0, `rgba(${b.r}, ${b.g}, ${b.b}, 1)`)
      gradient.addColorStop(
        b.effectiveHardness,
        `rgba(${b.r}, ${b.g}, ${b.b}, 1)`
      )
      gradient.addColorStop(1, `rgba(${b.r}, ${b.g}, ${b.b}, 0)`)
      ctx.fillStyle = gradient
    }

    ctx.fill()
  }

  function drawSegment(ctx: CanvasRenderingContext2D, from: Point, to: Point) {
    const b = strokeBrush!

    if (b.hardness < 1) {
      const dx = to.x - from.x
      const dy = to.y - from.y
      const dist = Math.hypot(dx, dy)
      const step = Math.max(1, b.effectiveRadius / 2)

      if (dist > 0) {
        const steps = Math.ceil(dist / step)
        const dabPoint: Point = { x: 0, y: 0 }
        for (let i = 1; i <= steps; i++) {
          const t = i / steps
          dabPoint.x = from.x + dx * t
          dabPoint.y = from.y + dy * t
          drawCircle(ctx, dabPoint)
        }
      }
    } else {
      expandDirtyRect(from.x, from.y, b.effectiveRadius)
      ctx.beginPath()
      ctx.moveTo(from.x, from.y)
      ctx.lineTo(to.x, to.y)
      ctx.stroke()
      drawCircle(ctx, to)
    }
  }

  function applyBrushStyle(ctx: CanvasRenderingContext2D) {
    const b = strokeBrush!
    const color = `rgb(${b.r}, ${b.g}, ${b.b})`

    ctx.globalCompositeOperation = 'source-over'
    ctx.globalAlpha = 1
    ctx.fillStyle = color
    ctx.strokeStyle = color
    ctx.lineWidth = b.effectiveRadius * 2
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'
  }

  function ensureStrokeCanvas() {
    const el = canvasEl.value
    if (!el) return null

    if (
      !strokeCanvas ||
      strokeCanvas.width !== el.width ||
      strokeCanvas.height !== el.height
    ) {
      strokeCanvas = document.createElement('canvas')
      strokeCanvas.width = el.width
      strokeCanvas.height = el.height
      strokeCtx = strokeCanvas.getContext('2d')
    }

    strokeCtx?.clearRect(0, 0, strokeCanvas.width, strokeCanvas.height)
    return strokeCtx
  }

  function ensureBaseCanvas() {
    const el = canvasEl.value
    if (!el) return null

    if (
      !baseCanvas ||
      baseCanvas.width !== el.width ||
      baseCanvas.height !== el.height
    ) {
      baseCanvas = document.createElement('canvas')
      baseCanvas.width = el.width
      baseCanvas.height = el.height
      baseCtx = baseCanvas.getContext('2d')
    }

    return baseCtx
  }

  function compositeStrokeToMain(isPreview: boolean = false) {
    const el = canvasEl.value
    const ctx = getCtx()
    if (!el || !ctx || !strokeCanvas) return

    const useDirty = hasDirtyRect
    const sx = Math.max(0, Math.floor(dirtyX0))
    const sy = Math.max(0, Math.floor(dirtyY0))
    const sr = Math.min(el.width, Math.ceil(dirtyX1))
    const sb = Math.min(el.height, Math.ceil(dirtyY1))
    const sw = sr - sx
    const sh = sb - sy
    hasDirtyRect = false

    if (hasBaseSnapshot && baseCanvas) {
      if (useDirty && sw > 0 && sh > 0) {
        ctx.clearRect(sx, sy, sw, sh)
        ctx.drawImage(baseCanvas, sx, sy, sw, sh, sx, sy, sw, sh)
      } else {
        ctx.clearRect(0, 0, el.width, el.height)
        ctx.drawImage(baseCanvas, 0, 0)
      }
    }

    ctx.save()
    const isEraser = tool.value === PAINTER_TOOLS.ERASER
    ctx.globalAlpha = isEraser ? 1 : brushOpacity.value
    ctx.globalCompositeOperation = isEraser ? 'destination-out' : 'source-over'
    if (useDirty && sw > 0 && sh > 0) {
      ctx.drawImage(strokeCanvas, sx, sy, sw, sh, sx, sy, sw, sh)
    } else {
      ctx.drawImage(strokeCanvas, 0, 0)
    }
    ctx.restore()

    if (!isPreview) {
      hasBaseSnapshot = false
    }
  }

  function startStroke(e: PointerEvent) {
    const point = getCanvasPoint(e)
    if (!point) return

    const el = canvasEl.value
    if (!el) return

    const bCtx = ensureBaseCanvas()
    if (bCtx) {
      bCtx.clearRect(0, 0, el.width, el.height)
      bCtx.drawImage(el, 0, 0)
      hasBaseSnapshot = true
    }

    isDrawing = true
    isDirty.value = true
    hasStrokes = true
    snapshotBrush()
    strokeProcessor = new StrokeProcessor(Math.max(1, strokeBrush!.radius / 2))
    strokeProcessor.addPoint(point)
    lastPoint = point

    const ctx = ensureStrokeCanvas()
    if (!ctx) return
    ctx.save()
    applyBrushStyle(ctx)
    drawCircle(ctx, point)
    ctx.restore()

    compositeStrokeToMain(true)
  }

  function continueStroke(e: PointerEvent) {
    if (!isDrawing || !strokeProcessor || !strokeCtx) return

    const point = getCanvasPoint(e)
    if (!point) return

    const points = strokeProcessor.addPoint(point)
    if (points.length === 0 && lastPoint) {
      points.push(point)
    }

    if (points.length === 0) return

    strokeCtx.save()
    applyBrushStyle(strokeCtx)

    let prev = lastPoint ?? points[0]
    for (const p of points) {
      drawSegment(strokeCtx, prev, p)
      prev = p
    }
    lastPoint = prev

    strokeCtx.restore()

    compositeStrokeToMain(true)
  }

  function endStroke() {
    if (!isDrawing || !strokeProcessor) return

    const points = strokeProcessor.endStroke()
    if (strokeCtx && points.length > 0) {
      strokeCtx.save()
      applyBrushStyle(strokeCtx)
      let prev = lastPoint ?? points[0]
      for (const p of points) {
        drawSegment(strokeCtx, prev, p)
        prev = p
      }
      strokeCtx.restore()
    }

    compositeStrokeToMain()

    isDrawing = false
    strokeProcessor = null
    strokeBrush = null
    lastPoint = null
  }

  function resizeCanvas() {
    const el = canvasEl.value
    if (!el) return

    let tmp: HTMLCanvasElement | null = null
    if (el.width > 0 && el.height > 0) {
      tmp = document.createElement('canvas')
      tmp.width = el.width
      tmp.height = el.height
      tmp.getContext('2d')!.drawImage(el, 0, 0)
    }

    el.width = canvasWidth.value
    el.height = canvasHeight.value
    mainCtx = null

    if (tmp) {
      getCtx()?.drawImage(tmp, 0, 0)
    }

    strokeCanvas = null
    strokeCtx = null
    baseCanvas = null
    baseCtx = null
    hasBaseSnapshot = false
  }

  function handleClear() {
    const el = canvasEl.value
    const ctx = getCtx()
    if (!el || !ctx) return
    ctx.clearRect(0, 0, el.width, el.height)
    isDirty.value = true
    hasStrokes = false
  }

  function updateCursorPos(e: PointerEvent) {
    cursorX.value = e.offsetX
    cursorY.value = e.offsetY
  }

  function handlePointerDown(e: PointerEvent) {
    if (e.button !== 0) return
    ;(e.target as HTMLElement).setPointerCapture(e.pointerId)
    cacheCanvasRect()
    updateCursorPos(e)
    startStroke(e)
  }

  let pendingMoveEvent: PointerEvent | null = null
  let rafId: number | null = null

  function flushPendingStroke() {
    if (pendingMoveEvent) {
      continueStroke(pendingMoveEvent)
      pendingMoveEvent = null
    }
    rafId = null
  }

  function handlePointerMove(e: PointerEvent) {
    updateCursorPos(e)
    if (!isDrawing) return

    pendingMoveEvent = e
    if (!rafId) {
      rafId = requestAnimationFrame(flushPendingStroke)
    }
  }

  function handlePointerUp(e: PointerEvent) {
    if (e.button !== 0) return
    if (rafId) {
      cancelAnimationFrame(rafId)
      flushPendingStroke()
    }
    ;(e.target as HTMLElement).releasePointerCapture(e.pointerId)
    endStroke()
  }

  function handlePointerLeave() {
    cursorVisible.value = false
    if (rafId) {
      cancelAnimationFrame(rafId)
      flushPendingStroke()
    }
    endStroke()
  }

  function handlePointerEnter() {
    cursorVisible.value = true
  }

  function handleInputImageLoad(e: Event) {
    const img = e.target as HTMLImageElement
    const widthWidget = getWidgetByName('width')
    const heightWidget = getWidgetByName('height')
    if (widthWidget) {
      widthWidget.value = img.naturalWidth
      widthWidget.callback?.(img.naturalWidth)
    }
    if (heightWidget) {
      heightWidget.value = img.naturalHeight
      heightWidget.callback?.(img.naturalHeight)
    }
    canvasWidth.value = img.naturalWidth
    canvasHeight.value = img.naturalHeight
  }

  function parseMaskFilename(value: string): {
    filename: string
    subfolder: string
    type: string
  } | null {
    const trimmed = value?.trim()
    if (!trimmed) return null

    const typeMatch = trimmed.match(/^(.+?) \[([^\]]+)\]$/)
    const pathPart = typeMatch ? typeMatch[1] : trimmed
    const type = typeMatch ? typeMatch[2] : 'input'

    const lastSlash = pathPart.lastIndexOf('/')
    const subfolder = lastSlash !== -1 ? pathPart.substring(0, lastSlash) : ''
    const filename =
      lastSlash !== -1 ? pathPart.substring(lastSlash + 1) : pathPart

    return { filename, subfolder, type }
  }

  function isCanvasEmpty(): boolean {
    return !hasStrokes
  }

  async function serializeValue(): Promise<string> {
    const el = canvasEl.value
    if (!el) return ''

    if (isCanvasEmpty()) return ''

    if (!isDirty.value) return modelValue.value

    const blob = await new Promise<Blob | null>((resolve) =>
      el.toBlob(resolve, 'image/png')
    )
    if (!blob) return modelValue.value

    const name = `painter-${nodeId}-${Date.now()}.png`
    const body = new FormData()
    body.append('image', blob, name)
    if (!isCloud) body.append('subfolder', 'painter')
    body.append('type', isCloud ? 'input' : 'temp')

    let resp: Response
    try {
      resp = await api.fetchApi('/upload/image', {
        method: 'POST',
        body
      })
    } catch (e) {
      const err = t('painter.uploadError', {
        status: 0,
        statusText: e instanceof Error ? e.message : String(e)
      })
      toastStore.addAlert(err)
      throw new Error(err)
    }

    if (resp.status !== 200) {
      const err = t('painter.uploadError', {
        status: resp.status,
        statusText: resp.statusText
      })
      toastStore.addAlert(err)
      throw new Error(err)
    }

    let data: { name: string }
    try {
      data = await resp.json()
    } catch (e) {
      const err = t('painter.uploadError', {
        status: resp.status,
        statusText: e instanceof Error ? e.message : String(e)
      })
      toastStore.addAlert(err)
      throw new Error(err)
    }

    const result = isCloud
      ? `${data.name} [input]`
      : `painter/${data.name} [temp]`
    modelValue.value = result
    isDirty.value = false
    return result
  }

  function registerWidgetSerialization() {
    const node = litegraphNode.value
    if (!node?.widgets) return
    const targetWidget = node.widgets.find(
      (w: IBaseWidget) => w.name === 'mask'
    )
    if (targetWidget) {
      targetWidget.serializeValue = serializeValue
    }
  }

  function restoreCanvas() {
    const parsed = parseMaskFilename(modelValue.value)
    if (!parsed) return

    const params = new URLSearchParams()
    params.set('filename', parsed.filename)
    if (parsed.subfolder) params.set('subfolder', parsed.subfolder)
    params.set('type', parsed.type)

    const url = api.apiURL('/view?' + params.toString())
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      const el = canvasEl.value
      if (!el) return
      canvasWidth.value = img.naturalWidth
      canvasHeight.value = img.naturalHeight
      mainCtx = null
      getCtx()?.drawImage(img, 0, 0)
      isDirty.value = false
      hasStrokes = true
    }
    img.onerror = () => {
      modelValue.value = ''
    }
    img.src = url
  }

  watch(() => nodeOutputStore.nodeOutputs, updateInputImageUrl, { deep: true })
  watch(() => nodeOutputStore.nodePreviewImages, updateInputImageUrl, {
    deep: true
  })
  watch([canvasWidth, canvasHeight], resizeCanvas)

  watch(
    [tool, brushSize, brushColor, brushOpacity, brushHardness],
    saveSettingsToProperties
  )

  watch([canvasWidth, canvasHeight], syncCanvasSizeToWidgets)

  watch(backgroundColor, syncBackgroundColorToWidget)

  function initialize() {
    syncCanvasSizeFromWidgets()
    resizeCanvas()
    registerWidgetSerialization()
    restoreSettingsFromProperties()
    updateInputImageUrl()
    restoreCanvas()
  }

  onMounted(initialize)

  onUnmounted(() => {
    if (rafId) {
      cancelAnimationFrame(rafId)
      rafId = null
    }
  })

  return {
    tool,
    brushSize,
    brushColor,
    brushOpacity,
    brushHardness,
    backgroundColor,

    canvasWidth,
    canvasHeight,

    cursorX,
    cursorY,
    cursorVisible,
    displayBrushSize,

    inputImageUrl,
    isImageInputConnected,

    handlePointerDown,
    handlePointerMove,
    handlePointerUp,
    handlePointerEnter,
    handlePointerLeave,

    handleInputImageLoad,
    handleClear
  }
}
