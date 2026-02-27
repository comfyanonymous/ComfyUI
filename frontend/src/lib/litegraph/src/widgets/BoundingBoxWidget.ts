import type { IBoundingBoxWidget } from '../types/widgets'
import { BaseWidget } from './BaseWidget'
import type { DrawWidgetOptions, WidgetEventOptions } from './BaseWidget'

/**
 * Widget for defining bounding box regions.
 * This widget only has a Vue implementation.
 */
export class BoundingBoxWidget
  extends BaseWidget<IBoundingBoxWidget>
  implements IBoundingBoxWidget
{
  override type = 'boundingbox' as const

  drawWidget(ctx: CanvasRenderingContext2D, options: DrawWidgetOptions): void {
    this.drawVueOnlyWarning(ctx, options, 'BoundingBox')
  }

  onClick(_options: WidgetEventOptions): void {
    // This widget only has a Vue implementation
  }
}
