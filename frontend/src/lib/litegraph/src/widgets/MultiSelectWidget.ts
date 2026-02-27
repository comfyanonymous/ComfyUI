import { t } from '@/i18n'

import type { IMultiSelectWidget } from '../types/widgets'
import { BaseWidget } from './BaseWidget'
import type { DrawWidgetOptions, WidgetEventOptions } from './BaseWidget'

/**
 * Widget for selecting multiple options
 * This is a widget that only has a Vue widgets implementation
 */
export class MultiSelectWidget
  extends BaseWidget<IMultiSelectWidget>
  implements IMultiSelectWidget
{
  override type = 'multiselect' as const

  drawWidget(ctx: CanvasRenderingContext2D, options: DrawWidgetOptions): void {
    const { width } = options
    const { y, height } = this

    const { fillStyle, strokeStyle, textAlign, textBaseline, font } = ctx

    ctx.fillStyle = this.background_color
    ctx.fillRect(15, y, width - 30, height)

    ctx.strokeStyle = this.getOutlineColor(options.suppressPromotedOutline)
    ctx.strokeRect(15, y, width - 30, height)

    ctx.fillStyle = this.text_color
    ctx.font = '11px monospace'
    ctx.textAlign = 'center'
    ctx.textBaseline = 'middle'

    const text = `MultiSelect: ${t('widgets.node2only')}`
    ctx.fillText(text, width / 2, y + height / 2)

    Object.assign(ctx, {
      fillStyle,
      strokeStyle,
      textAlign,
      textBaseline,
      font
    })
  }

  onClick(_options: WidgetEventOptions): void {
    // This is a widget that only has a Vue widgets implementation
  }
}
