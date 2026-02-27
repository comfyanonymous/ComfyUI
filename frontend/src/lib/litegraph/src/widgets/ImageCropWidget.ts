import type { IImageCropWidget } from '../types/widgets'
import { BaseWidget } from './BaseWidget'
import type { DrawWidgetOptions, WidgetEventOptions } from './BaseWidget'

/**
 * Widget for displaying an image crop preview.
 * This widget only has a Vue implementation.
 */
export class ImageCropWidget
  extends BaseWidget<IImageCropWidget>
  implements IImageCropWidget
{
  override type = 'imagecrop' as const

  drawWidget(ctx: CanvasRenderingContext2D, options: DrawWidgetOptions): void {
    this.drawVueOnlyWarning(ctx, options, 'ImageCrop')
  }

  onClick(_options: WidgetEventOptions): void {
    // This widget only has a Vue implementation
  }
}
