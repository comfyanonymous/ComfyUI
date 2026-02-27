import type { INumericWidget } from '@/lib/litegraph/src/types/widgets'
import { evaluateInput, getWidgetStep } from '@/lib/litegraph/src/utils/widget'

import { BaseSteppedWidget } from './BaseSteppedWidget'
import type { WidgetEventOptions } from './BaseWidget'

export class NumberWidget
  extends BaseSteppedWidget<INumericWidget>
  implements INumericWidget
{
  override type = 'number' as const

  override get _displayValue() {
    if (this.computedDisabled) return ''
    return Number(this.value).toFixed(
      this.options.precision !== undefined ? this.options.precision : 3
    )
  }

  override canIncrement(): boolean {
    const { max } = this.options
    return max == null || this.value < max
  }

  override canDecrement(): boolean {
    const { min } = this.options
    return min == null || this.value > min
  }

  override incrementValue(options: WidgetEventOptions): void {
    this.setValue(this.value + getWidgetStep(this.options), options)
  }

  override decrementValue(options: WidgetEventOptions): void {
    this.setValue(this.value - getWidgetStep(this.options), options)
  }

  override setValue(value: number, options: WidgetEventOptions) {
    let newValue = value
    if (this.options.min != null && newValue < this.options.min) {
      newValue = this.options.min
    }
    if (this.options.max != null && newValue > this.options.max) {
      newValue = this.options.max
    }
    super.setValue(newValue, options)
  }

  override onClick({ e, node, canvas }: WidgetEventOptions) {
    const x = e.canvasX - node.pos[0]
    const width = this.width || node.size[0]

    // Determine if clicked on left/right arrows
    const delta = x < 40 ? -1 : x > width - 40 ? 1 : 0

    if (delta) {
      // Handle left/right arrow clicks
      this.setValue(this.value + delta * getWidgetStep(this.options), {
        e,
        node,
        canvas
      })
      return
    }

    // Handle center click - show prompt
    canvas.prompt(
      'Value',
      this.value,
      (v: string) => {
        const parsed = evaluateInput(v)
        if (parsed !== undefined) this.setValue(parsed, { e, node, canvas })
      },
      e
    )
  }

  /**
   * Handles drag events for the number widget
   * @param options The options for handling the drag event
   */
  override onDrag({ e, node, canvas }: WidgetEventOptions) {
    const width = this.width || node.width
    const x = e.canvasX - node.pos[0]
    const delta = x < 40 ? -1 : x > width - 40 ? 1 : 0

    if (delta && x > -3 && x < width + 3) return
    this.setValue(this.value + (e.deltaX ?? 0) * getWidgetStep(this.options), {
      e,
      node,
      canvas
    })
  }
}
