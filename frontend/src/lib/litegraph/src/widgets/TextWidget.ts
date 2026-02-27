import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { IStringWidget } from '@/lib/litegraph/src/types/widgets'

import { BaseWidget } from './BaseWidget'
import type { DrawWidgetOptions, WidgetEventOptions } from './BaseWidget'

export class TextWidget
  extends BaseWidget<IStringWidget>
  implements IStringWidget
{
  constructor(widget: IStringWidget, node: LGraphNode) {
    super(widget, node)
    this.type ??= 'string'
    this.value = widget.value?.toString() ?? ''
  }

  /**
   * Draws the widget
   * @param ctx The canvas context
   * @param options The options for drawing the widget
   */
  override drawWidget(
    ctx: CanvasRenderingContext2D,
    options: DrawWidgetOptions
  ) {
    const { width, showText = true } = options
    // Store original context attributes
    const { fillStyle, strokeStyle, textAlign } = ctx

    this.drawWidgetShape(ctx, options)

    if (showText) {
      this.drawTruncatingText({ ctx, width, leftPadding: 0, rightPadding: 0 })
    }

    // Restore original context attributes
    Object.assign(ctx, { textAlign, strokeStyle, fillStyle })
  }

  override onClick({ e, node, canvas }: WidgetEventOptions) {
    // Show prompt dialog for text input
    canvas.prompt(
      'Value',
      this.value,
      (v: string) => {
        if (v !== null) {
          this.setValue(v, { e, node, canvas })
        }
      },
      e,
      this.options?.multiline ?? false
    )
  }
}
