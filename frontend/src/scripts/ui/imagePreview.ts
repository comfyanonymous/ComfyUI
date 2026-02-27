import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'

import { app } from '../app'
import { $el } from '../ui'

export function calculateImageGrid(
  // @ts-expect-error fixme ts strict error
  imgs,
  // @ts-expect-error fixme ts strict error
  dw,
  // @ts-expect-error fixme ts strict error
  dh
): {
  cellWidth: number
  cellHeight: number
  cols: number
  rows: number
  shiftX: number
} {
  let best = 0
  let w = imgs[0].naturalWidth
  let h = imgs[0].naturalHeight
  const numImages = imgs.length

  let cellWidth, cellHeight, cols, rows, shiftX
  // compact style
  for (let c = 1; c <= numImages; c++) {
    const r = Math.ceil(numImages / c)
    const cW = dw / c
    const cH = dh / r
    const scaleX = cW / w
    const scaleY = cH / h

    const scale = Math.min(scaleX, scaleY, 1)
    const imageW = w * scale
    const imageH = h * scale
    const area = imageW * imageH * numImages

    if (area > best) {
      best = area
      cellWidth = imageW
      cellHeight = imageH
      cols = c
      rows = r
      shiftX = c * ((cW - imageW) / 2)
    }
  }

  // @ts-expect-error fixme ts strict error
  return { cellWidth, cellHeight, cols, rows, shiftX }
}

/** @knipIgnoreUnusedButUsedByCustomNodes */
export function createImageHost(node: LGraphNode) {
  const el = $el('div.comfy-img-preview')
  // @ts-expect-error fixme ts strict error
  let currentImgs
  let first = true

  function updateSize() {
    let w = null
    let h = null

    // @ts-expect-error fixme ts strict error
    if (currentImgs) {
      let elH = el.clientHeight
      if (first) {
        first = false
        // On first run, if we are small then grow a bit
        if (elH < 190) {
          elH = 190
        }
        el.style.setProperty('--comfy-widget-min-height', elH.toString())
      } else {
        el.style.setProperty('--comfy-widget-min-height', null)
      }

      const nw = node.size[0]
      ;({ cellWidth: w, cellHeight: h } = calculateImageGrid(
        currentImgs,
        nw - 20,
        elH
      ))
      // @ts-expect-error fixme ts strict error
      w += 'px'
      // @ts-expect-error fixme ts strict error
      h += 'px'

      // @ts-expect-error fixme ts strict error
      el.style.setProperty('--comfy-img-preview-width', w)
      // @ts-expect-error fixme ts strict error
      el.style.setProperty('--comfy-img-preview-height', h)
    }
  }
  return {
    el,
    getCurrentImage() {
      // @ts-expect-error fixme ts strict error
      return currentImgs?.[0]
    },
    // @ts-expect-error fixme ts strict error
    updateImages(imgs) {
      // @ts-expect-error fixme ts strict error
      if (imgs !== currentImgs) {
        // @ts-expect-error fixme ts strict error
        if (currentImgs == null) {
          requestAnimationFrame(() => {
            updateSize()
          })
        }
        el.replaceChildren(...imgs)
        currentImgs = imgs
        node.onResize?.(node.size)
        node.graph?.setDirtyCanvas(true, true)
      }
    },
    getHeight() {
      updateSize()
    },
    onDraw() {
      // Element from point uses a hittest find elements so we need to toggle pointer events
      el.style.pointerEvents = 'all'
      const over = document.elementFromPoint(
        app.canvas.mouse[0],
        app.canvas.mouse[1]
      )
      el.style.pointerEvents = 'none'

      if (!over) return
      // Set the overIndex so Open Image etc work
      // @ts-expect-error fixme ts strict error
      const idx = currentImgs.indexOf(over)
      node.overIndex = idx
    }
  }
}
