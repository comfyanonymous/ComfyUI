import { LGraphBadge } from './LGraphBadge'
import type { LGraphBadgeOptions } from './LGraphBadge'
import { Rectangle } from './infrastructure/Rectangle'

export interface LGraphButtonOptions extends LGraphBadgeOptions {
  name?: string // To identify the button
}

export class LGraphButton extends LGraphBadge {
  name?: string
  _last_area: Rectangle = new Rectangle()

  constructor(options: LGraphButtonOptions) {
    super(options)
    this.name = options.name
  }

  override getWidth(ctx: CanvasRenderingContext2D): number {
    if (!this.visible) return 0

    const { font } = ctx
    ctx.font = `${this.fontSize}px 'PrimeIcons'`

    // For icon buttons, just measure the text width without padding
    const textWidth = this.text ? ctx.measureText(this.text).width : 0

    ctx.font = font
    return textWidth
  }

  /**
   * @internal
   *
   * Draws the button and updates its last rendered area for hit detection.
   * @param ctx The canvas rendering context.
   * @param x The x-coordinate to draw the button at.
   * @param y The y-coordinate to draw the button at.
   */
  override draw(ctx: CanvasRenderingContext2D, x: number, y: number): void {
    if (!this.visible) {
      return
    }

    const width = this.getWidth(ctx)

    // Update the hit area
    this._last_area[0] = x + this.xOffset
    this._last_area[1] = y + this.yOffset
    this._last_area[2] = width
    this._last_area[3] = this.height

    // Custom drawing for buttons - no background, just icon/text
    const adjustedX = x + this.xOffset
    const adjustedY = y + this.yOffset

    const { font, fillStyle, textBaseline, textAlign } = ctx

    // Use the same color as the title text (usually white)
    const titleTextColor = ctx.fillStyle || 'white'

    // Draw as icon-only without background
    ctx.font = `${this.fontSize}px 'PrimeIcons'`
    ctx.fillStyle = titleTextColor
    ctx.textBaseline = 'middle'
    ctx.textAlign = 'center'

    const centerX = adjustedX + width / 2
    const centerY = adjustedY + this.height / 2

    if (this.text) {
      ctx.fillText(this.text, centerX, centerY)
    }

    // Restore context
    ctx.font = font
    ctx.fillStyle = fillStyle
    ctx.textBaseline = textBaseline
    ctx.textAlign = textAlign
  }

  /**
   * Checks if a point is inside the button's last rendered area.
   * @param x The x-coordinate of the point.
   * @param y The y-coordinate of the point.
   * @returns `true` if the point is inside the button, otherwise `false`.
   */
  isPointInside(x: number, y: number): boolean {
    return this._last_area.containsPoint([x, y])
  }
}
