import type { LGraphCanvas, Rectangle } from '@/lib/litegraph/src/litegraph'
import { createBounds } from '@/lib/litegraph/src/litegraph'
import { app } from '@/scripts/app'

/**
 * Draws a dashed border around selected items that maintains constant pixel size
 * regardless of zoom level, similar to the DOM selection overlay.
 */
function drawSelectionBorder(
  ctx: CanvasRenderingContext2D,
  canvas: LGraphCanvas
) {
  const selectedItems = canvas.selectedItems

  // Only draw if multiple items selected
  if (selectedItems.size <= 1) return

  // Use the same bounds calculation as the toolbox
  const bounds = createBounds(selectedItems, 10)
  if (!bounds) return

  const [x, y, width, height] = bounds

  // Save context state
  ctx.save()

  // Set up dashed line style that doesn't scale with zoom
  const borderWidth = 2 / canvas.ds.scale // Constant 2px regardless of zoom
  ctx.lineWidth = borderWidth
  ctx.strokeStyle =
    getComputedStyle(document.documentElement)
      .getPropertyValue('--border-color')
      .trim() || '#ffffff66'

  // Create dash pattern that maintains visual size
  const dashSize = 5 / canvas.ds.scale
  ctx.setLineDash([dashSize, dashSize])

  // Draw the border using the bounds directly
  ctx.beginPath()
  ctx.roundRect(x, y, width, height, 8 / canvas.ds.scale)
  ctx.stroke()

  // Restore context
  ctx.restore()
}

/**
 * Extension that adds a dashed selection border for multiple selected nodes
 */
const ext = {
  name: 'Comfy.SelectionBorder',

  async init() {
    // Hook into the canvas drawing
    const originalDrawForeground = app.canvas.onDrawForeground

    app.canvas.onDrawForeground = function (
      ctx: CanvasRenderingContext2D,
      visibleArea: Rectangle
    ) {
      // Call original if it exists
      originalDrawForeground?.call(this, ctx, visibleArea)

      // Draw our selection border
      drawSelectionBorder(ctx, app.canvas)
    }
  }
}

app.registerExtension(ext)
