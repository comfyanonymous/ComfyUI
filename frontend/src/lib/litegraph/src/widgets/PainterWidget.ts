import type { IPainterWidget } from '../types/widgets'
import { BaseWidget } from './BaseWidget'
import type { DrawWidgetOptions, WidgetEventOptions } from './BaseWidget'

/**
 * Widget for the Painter node canvas drawing tool.
 * This is a widget that only has a Vue widgets implementation.
 */
export class PainterWidget
  extends BaseWidget<IPainterWidget>
  implements IPainterWidget
{
  override type = 'painter' as const

  drawWidget(ctx: CanvasRenderingContext2D, options: DrawWidgetOptions): void {
    this.drawVueOnlyWarning(ctx, options, 'Painter')
  }

  onClick(_options: WidgetEventOptions): void {
    // This is a widget that only has a Vue widgets implementation
  }
}
