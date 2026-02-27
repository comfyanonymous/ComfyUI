import type { ReadOnlyRect } from '@/lib/litegraph/src/interfaces'
import { LGraphIcon } from './LGraphIcon'
import type { LGraphIconOptions } from './LGraphIcon'

export enum BadgePosition {
  TopLeft = 'top-left',
  TopRight = 'top-right'
}

export interface LGraphBadgeOptions {
  text: string
  fgColor?: string
  bgColor?: string
  fontSize?: number
  padding?: number
  height?: number
  cornerRadius?: number
  iconOptions?: LGraphIconOptions
  onClick?: (e: MouseEvent) => void
  xOffset?: number
  yOffset?: number
}

export class LGraphBadge {
  text: string
  fgColor: string
  bgColor: string
  fontSize: number
  padding: number
  height: number
  cornerRadius: number
  icon?: LGraphIcon
  onClick?: (e: MouseEvent) => void
  xOffset: number
  yOffset: number

  readonly _boundingRect: [number, number, number, number] = [0, 0, 0, 0]
  get boundingRect(): ReadOnlyRect {
    return this._boundingRect
  }

  constructor({
    text,
    fgColor = 'white',
    bgColor = '#0F1F0F',
    fontSize = 12,
    padding = 6,
    height = 20,
    cornerRadius = 5,
    iconOptions,
    onClick,
    xOffset = 0,
    yOffset = 0
  }: LGraphBadgeOptions) {
    this.text = text
    this.fgColor = fgColor
    this.bgColor = bgColor
    this.fontSize = fontSize
    this.padding = padding
    this.height = height
    this.cornerRadius = cornerRadius
    if (iconOptions) {
      this.icon = new LGraphIcon(iconOptions)
    }
    this.onClick = onClick
    this.xOffset = xOffset
    this.yOffset = yOffset
  }

  get visible() {
    return (this.text?.length ?? 0) > 0 || !!this.icon
  }

  getWidth(ctx: CanvasRenderingContext2D) {
    if (!this.visible) return 0
    const { font } = ctx
    let iconWidth = 0
    if (this.icon) {
      if (this.icon.image) {
        iconWidth = this.icon.size + this.padding
      } else if (this.icon.unicode) {
        ctx.font = `${this.icon.fontSize}px '${this.icon.fontFamily}'`
        iconWidth = ctx.measureText(this.icon.unicode).width + this.padding
      }
    }
    ctx.font = `${this.fontSize}px sans-serif`
    const textWidth = this.text ? ctx.measureText(this.text).width : 0
    ctx.font = font
    return iconWidth + textWidth + this.padding * 2
  }

  draw(ctx: CanvasRenderingContext2D, x: number, y: number): void {
    if (!this.visible) return

    x += this.xOffset
    y += this.yOffset

    const { font, fillStyle, textBaseline, textAlign } = ctx

    ctx.font = `${this.fontSize}px sans-serif`
    const badgeWidth = this.getWidth(ctx)
    const badgeX = 0

    this._boundingRect.splice(0, 4, x, y, badgeWidth, this.height)

    // Draw badge background
    ctx.fillStyle = this.bgColor
    ctx.beginPath()
    if (ctx.roundRect) {
      ctx.roundRect(x + badgeX, y, badgeWidth, this.height, this.cornerRadius)
    } else {
      // Fallback for browsers that don't support roundRect
      ctx.rect(x + badgeX, y, badgeWidth, this.height)
    }
    ctx.fill()

    let drawX = x + badgeX + this.padding
    const centerY = y + this.height / 2

    // Draw icon if present
    if (this.icon) {
      this.icon.draw(ctx, drawX, centerY)
      const iconWidth = this.icon.image ? this.icon.size : this.icon.fontSize
      drawX += iconWidth + this.padding / 2 + 4
    }

    // Draw badge text
    if (this.text) {
      ctx.fillStyle = this.fgColor
      ctx.textBaseline = 'middle'
      ctx.textAlign = 'left'
      ctx.fillText(this.text, drawX, centerY + 1)
    }

    ctx.font = font
    ctx.fillStyle = fillStyle
    ctx.textBaseline = textBaseline
    ctx.textAlign = textAlign
  }
}
