import type { IBooleanWidget } from '@/lib/litegraph/src/types/widgets'

import { BaseWidget } from './BaseWidget'
import type { DrawWidgetOptions, WidgetEventOptions } from './BaseWidget'

export class BooleanWidget
  extends BaseWidget<IBooleanWidget>
  implements IBooleanWidget
{
  override type = 'toggle' as const

  override drawWidget(
    ctx: CanvasRenderingContext2D,
    options: DrawWidgetOptions
  ) {
    const { width, showText = true } = options
    const { height, y } = this
    const { margin } = BaseWidget

    this.drawWidgetShape(ctx, options)

    ctx.fillStyle = this.value ? '#89A' : '#333'
    ctx.beginPath()
    ctx.arc(width - margin * 2, y + height * 0.5, height * 0.36, 0, Math.PI * 2)
    ctx.fill()

    if (showText) {
      this.drawLabel(ctx, margin * 2)
      this.drawValue(ctx, width - 40)
    }
  }

  drawLabel(ctx: CanvasRenderingContext2D, x: number): void {
    // Draw label
    ctx.fillStyle = this.secondary_text_color
    const { displayName } = this
    if (displayName) ctx.fillText(displayName, x, this.labelBaseline)
  }

  drawValue(ctx: CanvasRenderingContext2D, x: number): void {
    // Draw value
    ctx.fillStyle = this.value ? this.text_color : this.secondary_text_color
    ctx.textAlign = 'right'
    const value = this.value
      ? this.options.on || 'true'
      : this.options.off || 'false'
    ctx.fillText(value, x, this.labelBaseline)
  }

  override onClick(options: WidgetEventOptions) {
    this.setValue(!this.value, options)
  }
}
