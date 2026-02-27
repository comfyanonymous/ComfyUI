import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { IButtonWidget } from '@/lib/litegraph/src/types/widgets'

import { BaseWidget } from './BaseWidget'
import type { DrawWidgetOptions, WidgetEventOptions } from './BaseWidget'

export class ButtonWidget
  extends BaseWidget<IButtonWidget>
  implements IButtonWidget
{
  override type = 'button' as const
  clicked: boolean

  constructor(widget: IButtonWidget, node: LGraphNode) {
    super(widget, node)
    this.clicked ??= false
  }

  /**
   * Draws the widget
   * @param ctx The canvas context
   * @param options The options for drawing the widget
   */
  override drawWidget(
    ctx: CanvasRenderingContext2D,
    { width, showText = true, suppressPromotedOutline }: DrawWidgetOptions
  ) {
    // Store original context attributes
    const { fillStyle, strokeStyle, textAlign } = ctx

    const { height, y } = this
    const { margin } = BaseWidget

    // Draw button background
    ctx.fillStyle = this.background_color
    if (this.clicked) {
      ctx.fillStyle = '#AAA'
      this.clicked = false
    }
    ctx.fillRect(margin, y, width - margin * 2, height)

    // Draw button outline if not disabled
    if (showText && !this.computedDisabled) {
      ctx.strokeStyle = this.getOutlineColor(suppressPromotedOutline)
      ctx.strokeRect(margin, y, width - margin * 2, height)
    }

    // Draw button text
    if (showText) this.drawLabel(ctx, width * 0.5)

    // Restore original context attributes
    Object.assign(ctx, { textAlign, strokeStyle, fillStyle })
  }

  drawLabel(ctx: CanvasRenderingContext2D, x: number): void {
    ctx.textAlign = 'center'
    ctx.fillStyle = this.text_color
    ctx.fillText(this.displayName, x, this.y + this.height * 0.7)
  }

  override onClick({ e, node, canvas }: WidgetEventOptions) {
    const pos = canvas.graph_mouse

    // Set clicked state and mark canvas as dirty
    this.clicked = true
    canvas.setDirty(true)

    // Call the callback with widget value
    this.callback?.(this.value, canvas, node, pos, e)
  }
}
