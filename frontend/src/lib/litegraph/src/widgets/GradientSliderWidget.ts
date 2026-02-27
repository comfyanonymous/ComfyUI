import { clamp } from 'es-toolkit/compat'

import type { IGradientSliderWidget } from '@/lib/litegraph/src/types/widgets'

import { BaseWidget } from './BaseWidget'
import type { DrawWidgetOptions, WidgetEventOptions } from './BaseWidget'

export class GradientSliderWidget
  extends BaseWidget<IGradientSliderWidget>
  implements IGradientSliderWidget
{
  override type = 'gradientslider' as const

  override drawWidget(
    ctx: CanvasRenderingContext2D,
    { width, showText = true }: DrawWidgetOptions
  ) {
    ctx.save()

    const { height, y } = this
    const { margin } = BaseWidget

    ctx.fillStyle = this.background_color
    ctx.fillRect(margin, y, width - margin * 2, height)

    const range = this.options.max - this.options.min
    let nvalue = (this.value - this.options.min) / range
    nvalue = clamp(nvalue, 0, 1)

    ctx.fillStyle = '#678'
    ctx.fillRect(margin, y, nvalue * (width - margin * 2), height)

    if (showText && !this.computedDisabled) {
      ctx.strokeStyle = this.outline_color
      ctx.strokeRect(margin, y, width - margin * 2, height)
    }

    if (showText) {
      ctx.textAlign = 'center'
      ctx.fillStyle = this.text_color
      const fixedValue = Number(this.value).toFixed(this.options.precision ?? 3)
      ctx.fillText(
        `${this.label || this.name}  ${fixedValue}`,
        width * 0.5,
        y + height * 0.7
      )
    }

    ctx.restore()
  }

  override onClick(options: WidgetEventOptions) {
    if (this.options.read_only) return

    const { e, node } = options
    const width = this.width || node.size[0]
    const x = e.canvasX - node.pos[0]

    const { margin } = BaseWidget
    const slideFactor = clamp((x - margin) / (width - margin * 2), 0, 1)
    const newValue =
      this.options.min + (this.options.max - this.options.min) * slideFactor

    if (newValue !== this.value) {
      this.setValue(newValue, options)
    }
  }

  override onDrag(options: WidgetEventOptions) {
    if (this.options.read_only) return false

    const { e, node } = options
    const width = this.width || node.size[0]
    const x = e.canvasX - node.pos[0]

    const { margin } = BaseWidget
    const slideFactor = clamp((x - margin) / (width - margin * 2), 0, 1)
    const newValue =
      this.options.min + (this.options.max - this.options.min) * slideFactor

    if (newValue !== this.value) {
      this.setValue(newValue, options)
    }
  }
}
