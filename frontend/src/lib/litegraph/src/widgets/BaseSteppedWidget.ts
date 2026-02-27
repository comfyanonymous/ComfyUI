import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'

import { BaseWidget } from './BaseWidget'
import type { DrawWidgetOptions, WidgetEventOptions } from './BaseWidget'

/**
 * Base class for widgets that have increment and decrement buttons.
 */
export abstract class BaseSteppedWidget<
  TWidget extends IBaseWidget = IBaseWidget
> extends BaseWidget<TWidget> {
  /**
   * Whether the widget can increment its value
   * @returns `true` if the widget can increment its value, otherwise `false`
   */
  abstract canIncrement(): boolean
  /**
   * Whether the widget can decrement its value
   * @returns `true` if the widget can decrement its value, otherwise `false`
   */
  abstract canDecrement(): boolean
  /**
   * Increment the value of the widget
   * @param options The options for the widget event
   */
  abstract incrementValue(options: WidgetEventOptions): void
  /**
   * Decrement the value of the widget
   * @param options The options for the widget event
   */
  abstract decrementValue(options: WidgetEventOptions): void

  /**
   * Draw the arrow buttons for the widget
   * @param ctx The canvas rendering context
   * @param width The width of the widget
   */
  drawArrowButtons(ctx: CanvasRenderingContext2D, width: number) {
    const { height, text_color, disabledTextColor, y } = this
    const { arrowMargin, arrowWidth, margin } = BaseWidget
    const arrowTipX = margin + arrowMargin
    const arrowInnerX = arrowTipX + arrowWidth

    // Draw left arrow
    ctx.fillStyle = this.canDecrement() ? text_color : disabledTextColor
    ctx.beginPath()
    ctx.moveTo(arrowInnerX, y + 5)
    ctx.lineTo(arrowTipX, y + height * 0.5)
    ctx.lineTo(arrowInnerX, y + height - 5)
    ctx.fill()

    // Draw right arrow
    ctx.fillStyle = this.canIncrement() ? text_color : disabledTextColor
    ctx.beginPath()
    ctx.moveTo(width - arrowInnerX, y + 5)
    ctx.lineTo(width - arrowTipX, y + height * 0.5)
    ctx.lineTo(width - arrowInnerX, y + height - 5)
    ctx.fill()
  }

  override drawWidget(
    ctx: CanvasRenderingContext2D,
    options: DrawWidgetOptions
  ) {
    // Store original context attributes
    const { fillStyle, strokeStyle, textAlign } = ctx

    this.drawWidgetShape(ctx, options)
    if (options.showText) {
      if (!this.computedDisabled) this.drawArrowButtons(ctx, options.width)

      this.drawTruncatingText({ ctx, width: options.width })
    }

    // Restore original context attributes
    Object.assign(ctx, { textAlign, strokeStyle, fillStyle })
  }
}
