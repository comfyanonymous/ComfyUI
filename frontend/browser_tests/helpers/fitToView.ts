import type { ReadOnlyRect } from '../../src/lib/litegraph/src/interfaces'
import type { ComfyPage } from '../fixtures/ComfyPage'

interface FitToViewOptions {
  selectionOnly?: boolean
  zoom?: number
  padding?: number
}

/**
 * Instantly fits the canvas view to graph content without waiting for UI animation.
 *
 * Lives outside the shared fixture to keep the default ComfyPage interactions user-oriented.
 */
export async function fitToViewInstant(
  comfyPage: ComfyPage,
  options: FitToViewOptions = {}
) {
  const { selectionOnly = false, zoom = 0.75, padding = 10 } = options

  const rectangles = await comfyPage.page.evaluate<
    ReadOnlyRect[] | null,
    { selectionOnly: boolean }
  >(
    ({ selectionOnly }) => {
      const app = window.app
      if (!app?.canvas) return null

      const canvas = app.canvas
      const items = (() => {
        if (selectionOnly && canvas.selectedItems?.size) {
          return Array.from(canvas.selectedItems)
        }
        try {
          return Array.from(canvas.positionableItems ?? [])
        } catch {
          return []
        }
      })()

      if (!items.length) return null

      const rects: ReadOnlyRect[] = []

      for (const item of items) {
        const rect = item?.boundingRect
        if (!rect) continue

        const x = Number(rect[0])
        const y = Number(rect[1])
        const width = Number(rect[2])
        const height = Number(rect[3])

        rects.push([x, y, width, height] as const)
      }

      return rects.length ? rects : null
    },
    { selectionOnly }
  )

  if (!rectangles || rectangles.length === 0) return

  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity

  for (const [x, y, width, height] of rectangles) {
    minX = Math.min(minX, Number(x))
    minY = Math.min(minY, Number(y))
    maxX = Math.max(maxX, Number(x) + Number(width))
    maxY = Math.max(maxY, Number(y) + Number(height))
  }

  const hasFiniteBounds =
    Number.isFinite(minX) &&
    Number.isFinite(minY) &&
    Number.isFinite(maxX) &&
    Number.isFinite(maxY)

  if (!hasFiniteBounds) return

  const bounds: ReadOnlyRect = [
    minX - padding,
    minY - padding,
    maxX - minX + 2 * padding,
    maxY - minY + 2 * padding
  ]

  await comfyPage.page.evaluate(
    ({ bounds, zoom }) => {
      const app = window.app
      if (!app?.canvas) return

      const canvas = app.canvas
      canvas.ds.fitToBounds(bounds, { zoom })
      canvas.setDirty(true, true)
    },
    { bounds, zoom }
  )

  await comfyPage.nextFrame()
}
