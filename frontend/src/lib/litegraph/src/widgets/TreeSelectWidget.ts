import type { ITreeSelectWidget } from '../types/widgets'
import { BaseWidget } from './BaseWidget'
import type { DrawWidgetOptions, WidgetEventOptions } from './BaseWidget'

/**
 * Widget for hierarchical tree selection.
 * This widget only has a Vue implementation.
 */
export class TreeSelectWidget
  extends BaseWidget<ITreeSelectWidget>
  implements ITreeSelectWidget
{
  override type = 'treeselect' as const

  drawWidget(ctx: CanvasRenderingContext2D, options: DrawWidgetOptions): void {
    this.drawVueOnlyWarning(ctx, options, 'TreeSelect')
  }

  onClick(_options: WidgetEventOptions): void {
    // This widget only has a Vue implementation
  }
}
