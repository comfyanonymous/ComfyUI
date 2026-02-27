import type { ITextareaWidget } from '../types/widgets'
import { BaseWidget } from './BaseWidget'
import type { DrawWidgetOptions, WidgetEventOptions } from './BaseWidget'

/**
 * Widget for multi-line text input.
 * This widget only has a Vue implementation.
 */
export class TextareaWidget
  extends BaseWidget<ITextareaWidget>
  implements ITextareaWidget
{
  override type = 'textarea' as const

  drawWidget(ctx: CanvasRenderingContext2D, options: DrawWidgetOptions): void {
    this.drawVueOnlyWarning(ctx, options, 'Textarea')
  }

  onClick(_options: WidgetEventOptions): void {
    // This widget only has a Vue implementation
  }
}
