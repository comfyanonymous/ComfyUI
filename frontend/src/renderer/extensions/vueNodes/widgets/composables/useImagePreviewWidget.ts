import { BaseWidget, LiteGraph } from '@/lib/litegraph/src/litegraph'
import type { CanvasPointer, LGraphNode } from '@/lib/litegraph/src/litegraph'
import type {
  IBaseWidget,
  IWidgetOptions
} from '@/lib/litegraph/src/types/widgets'
import type { DrawWidgetOptions } from '@/lib/litegraph/src/widgets/BaseWidget'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import type { InputSpec } from '@/schemas/nodeDef/nodeDefSchemaV2'
import { app } from '@/scripts/app'
import { calculateImageGrid } from '@/scripts/ui/imagePreview'
import type { ComfyWidgetConstructorV2 } from '@/scripts/widgets'
import { is_all_same_aspect_ratio } from '@/utils/imageUtil'

/**
 * Workaround for Chrome GPU bug:
 * When Chrome is maximized with GPU acceleration and high DPR, calling
 * drawImage(canvas) + drawImage(img) in the same frame causes severe
 * performance degradation (FPS drops to 2-10, memory spikes ~18GB).
 *
 * Solution: Defer image rendering using queueMicrotask to separate
 * the two drawImage calls into different tasks.
 *
 * Note: As tested, requestAnimationFrame delays rendering to the next frame,
 * causing visible image flickering. queueMicrotask executes within the same
 * frame, avoiding flicker while still separating the drawImage calls.
 */
let deferredImageRenders: Array<() => void> = []
let deferredRenderScheduled = false

function scheduleDeferredImageRender() {
  if (deferredRenderScheduled) return
  deferredRenderScheduled = true

  queueMicrotask(() => {
    const renders = deferredImageRenders
    deferredImageRenders = []
    deferredRenderScheduled = false

    for (const render of renders) {
      render()
    }
  })
}

const TWO_PI = Math.PI * 2
const SPINNER_ARC_LENGTH = Math.PI * 1.5

function renderUploadSpinner(
  ctx: CanvasRenderingContext2D,
  node: LGraphNode,
  shiftY: number,
  computedHeight: number | undefined
) {
  const dw = node.size[0]
  const dh = computedHeight ?? 220
  const centerX = dw / 2
  const centerY = shiftY + dh / 2
  const radius = 16
  const angle = ((Date.now() % 1000) / 1000) * TWO_PI

  ctx.save()
  ctx.strokeStyle = LiteGraph.NODE_TEXT_COLOR
  ctx.lineWidth = 3
  ctx.lineCap = 'round'
  ctx.beginPath()
  ctx.arc(centerX, centerY, radius, angle, angle + SPINNER_ARC_LENGTH)
  ctx.stroke()
  ctx.restore()

  // Schedule next frame to keep spinner animating continuously.
  // Only runs while node.isUploading is true (checked by caller).
  node.graph?.setDirtyCanvas(true)
}

const renderPreview = (
  ctx: CanvasRenderingContext2D,
  node: LGraphNode,
  shiftY: number,
  computedHeight: number | undefined,
  imgs: HTMLImageElement[],
  width: number
) => {
  if (!node.size) return

  if (node.isUploading) {
    renderUploadSpinner(ctx, node, shiftY, computedHeight)
    return
  }

  const canvas = useCanvasStore().getCanvas()
  const mouse = canvas.graph_mouse

  if (!canvas.pointer_is_down && node.pointerDown) {
    if (
      mouse[0] === node.pointerDown.pos[0] &&
      mouse[1] === node.pointerDown.pos[1]
    ) {
      node.imageIndex = node.pointerDown.index
    }
    node.pointerDown = null
  }

  if (imgs.length === 0) return

  let { imageIndex } = node
  const numImages = imgs.length
  if (numImages === 1 && !imageIndex) {
    // This skips the thumbnail render section below
    node.imageIndex = imageIndex = 0
  }

  const settingStore = useSettingStore()
  const allowImageSizeDraw = settingStore.get('Comfy.Node.AllowImageSizeDraw')
  const IMAGE_TEXT_SIZE_TEXT_HEIGHT = allowImageSizeDraw ? 15 : 0
  const dw = width
  const dh = computedHeight ? computedHeight - IMAGE_TEXT_SIZE_TEXT_HEIGHT : 0

  if (imageIndex == null) {
    // No image selected; draw thumbnails of all
    let cellWidth: number
    let cellHeight: number
    let shiftX: number
    let cell_padding: number
    let cols: number

    const compact_mode = is_all_same_aspect_ratio(imgs)
    if (!compact_mode) {
      // use rectangle cell style and border line
      cell_padding = 2
      // Prevent infinite canvas2d scale-up
      const largestDimension = imgs.reduce(
        (acc, current) =>
          Math.max(acc, current.naturalWidth, current.naturalHeight),
        0
      )
      const fakeImgs = []
      fakeImgs.length = imgs.length
      fakeImgs[0] = {
        naturalWidth: largestDimension,
        naturalHeight: largestDimension
      }
      ;({ cellWidth, cellHeight, cols, shiftX } = calculateImageGrid(
        fakeImgs,
        dw,
        dh
      ))
    } else {
      cell_padding = 0
      ;({ cellWidth, cellHeight, cols, shiftX } = calculateImageGrid(
        imgs,
        dw,
        dh
      ))
    }

    let anyHovered = false
    node.imageRects = []
    for (let i = 0; i < numImages; i++) {
      const img = imgs[i]
      const row = Math.floor(i / cols)
      const col = i % cols
      const x = col * cellWidth + shiftX
      const y = row * cellHeight + shiftY
      if (!anyHovered) {
        anyHovered = LiteGraph.isInsideRectangle(
          mouse[0],
          mouse[1],
          x + node.pos[0],
          y + node.pos[1],
          cellWidth,
          cellHeight
        )
        if (anyHovered) {
          node.overIndex = i
          let value = 110
          if (canvas.pointer_is_down) {
            if (!node.pointerDown || node.pointerDown.index !== i) {
              node.pointerDown = { index: i, pos: [...mouse] }
            }
            value = 125
          }
          ctx.filter = `contrast(${value}%) brightness(${value}%)`
          canvas.canvas.style.cursor = 'pointer'
        }
      }
      node.imageRects.push([x, y, cellWidth, cellHeight])

      const wratio = cellWidth / img.width
      const hratio = cellHeight / img.height
      const ratio = Math.min(wratio, hratio)

      const imgHeight = ratio * img.height
      const imgY = row * cellHeight + shiftY + (cellHeight - imgHeight) / 2
      const imgWidth = ratio * img.width
      const imgX = col * cellWidth + shiftX + (cellWidth - imgWidth) / 2

      // Defer image rendering to work around Chrome GPU bug
      const transform = ctx.getTransform()
      const filter = ctx.filter
      const drawParams = {
        img,
        x: imgX + cell_padding,
        y: imgY + cell_padding,
        w: imgWidth - cell_padding * 2,
        h: imgHeight - cell_padding * 2
      }
      deferredImageRenders.push(() => {
        ctx.save()
        ctx.setTransform(transform)
        ctx.filter = filter
        ctx.drawImage(
          drawParams.img,
          drawParams.x,
          drawParams.y,
          drawParams.w,
          drawParams.h
        )
        ctx.restore()
      })
      scheduleDeferredImageRender()

      if (!compact_mode) {
        // rectangle cell and border line style
        ctx.strokeStyle = '#8F8F8F'
        ctx.lineWidth = 1
        ctx.strokeRect(
          x + cell_padding,
          y + cell_padding,
          cellWidth - cell_padding * 2,
          cellHeight - cell_padding * 2
        )
      }

      ctx.filter = 'none'
    }

    if (!anyHovered) {
      node.pointerDown = null
      node.overIndex = null
    }

    return
  }
  // Draw individual
  const img = imgs[imageIndex]
  let w = img.naturalWidth
  let h = img.naturalHeight

  const scaleX = dw / w
  const scaleY = dh / h
  const scale = Math.min(scaleX, scaleY, 1)

  w *= scale
  h *= scale

  const x = (dw - w) / 2
  const y = (dh - h) / 2 + shiftY

  // Defer image rendering to work around Chrome GPU bug
  const transform = ctx.getTransform()
  deferredImageRenders.push(() => {
    ctx.save()
    ctx.setTransform(transform)
    ctx.drawImage(img, x, y, w, h)
    ctx.restore()
  })
  scheduleDeferredImageRender()

  // Draw image size text below the image
  if (allowImageSizeDraw) {
    ctx.fillStyle = LiteGraph.NODE_TEXT_COLOR
    ctx.textAlign = 'center'
    ctx.font = '10px sans-serif'
    const sizeText = `${Math.round(img.naturalWidth)} × ${Math.round(img.naturalHeight)}`
    const textY = y + h + 10
    ctx.fillText(sizeText, x + w / 2, textY)
  }

  const drawButton = (
    x: number,
    y: number,
    sz: number,
    text: string
  ): boolean => {
    const hovered = LiteGraph.isInsideRectangle(
      mouse[0],
      mouse[1],
      x + node.pos[0],
      y + node.pos[1],
      sz,
      sz
    )
    let fill = '#333'
    let textFill = '#fff'
    let isClicking = false
    if (hovered) {
      canvas.canvas.style.cursor = 'pointer'
      if (canvas.pointer_is_down) {
        fill = '#1e90ff'
        isClicking = true
      } else {
        fill = '#eee'
        textFill = '#000'
      }
    }

    deferredImageRenders.push(() => {
      ctx.save()
      ctx.setTransform(transform)
      ctx.fillStyle = fill
      ctx.beginPath()
      ctx.roundRect(x, y, sz, sz, [4])
      ctx.fill()
      ctx.fillStyle = textFill
      ctx.font = '12px Inter, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(text, x + 15, y + 20)
      ctx.restore()
    })

    return isClicking
  }

  if (!(numImages > 1)) return

  const imageNum = (node.imageIndex ?? 0) + 1
  if (drawButton(dw - 40, dh + shiftY - 40, 30, `${imageNum}/${numImages}`)) {
    const i = imageNum >= numImages ? 0 : imageNum
    if (!node.pointerDown || node.pointerDown.index !== i) {
      node.pointerDown = { index: i, pos: [...mouse] }
    }
  }

  if (drawButton(dw - 40, shiftY + 10, 30, `x`)) {
    if (!node.pointerDown || node.pointerDown.index !== null) {
      node.pointerDown = { index: null, pos: [...mouse] }
    }
  }
}

class ImagePreviewWidget extends BaseWidget {
  constructor(
    node: LGraphNode,
    name: string,
    options: IWidgetOptions<string | object>
  ) {
    const widget: IBaseWidget = {
      name,
      options,
      type: 'custom',
      /** Dummy value to satisfy type requirements. */
      value: '',
      y: 0
    }
    super(widget, node)

    // Don't serialize the widget value
    this.serialize = false
  }

  override drawWidget(
    ctx: CanvasRenderingContext2D,
    options: DrawWidgetOptions
  ): void {
    const imgs = options.previewImages ?? this.node.imgs ?? []
    renderPreview(
      ctx,
      this.node,
      this.y,
      this.computedHeight,
      imgs,
      options.width
    )
  }

  override createCopyForNode(node: LGraphNode): this {
    const copy = new ImagePreviewWidget(
      node,
      this.name,
      this.options as IWidgetOptions<string | object>
    ) as this
    copy.value = this.value
    return copy
  }

  override onPointerDown(pointer: CanvasPointer, node: LGraphNode): boolean {
    pointer.onDragStart = () => {
      const { canvas } = app
      const { graph } = canvas
      canvas.emitBeforeChange()
      graph?.beforeChange()
      // Ensure that dragging is properly cleaned up, on success or failure.
      pointer.finally = () => {
        canvas.isDragging = false
        graph?.afterChange()
        canvas.emitAfterChange()
      }

      canvas.processSelect(node, pointer.eDown)
      canvas.isDragging = true
    }

    pointer.onDragEnd = (e) => {
      const { canvas } = app
      if (e.shiftKey || LiteGraph.alwaysSnapToGrid)
        canvas.graph?.snapToGrid(canvas.selectedItems)

      canvas.setDirty(true, true)
    }

    return true
  }

  override onClick(): void {}

  override computeLayoutSize() {
    return {
      minHeight: 220,
      minWidth: 1
    }
  }
}

export const useImagePreviewWidget = () => {
  const widgetConstructor: ComfyWidgetConstructorV2 = (
    node: LGraphNode,
    inputSpec: InputSpec
  ) => {
    return node.addCustomWidget(
      new ImagePreviewWidget(node, inputSpec.name, {
        serialize: false,
        canvasOnly: true
      })
    )
  }

  return widgetConstructor
}
