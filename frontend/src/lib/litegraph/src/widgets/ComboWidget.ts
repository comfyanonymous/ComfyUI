import { clamp } from 'es-toolkit/compat'

import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import { LiteGraph } from '@/lib/litegraph/src/litegraph'
import type {
  IComboWidget,
  IStringComboWidget
} from '@/lib/litegraph/src/types/widgets'
import { warnDeprecated } from '@/lib/litegraph/src/utils/feedback'

import { BaseSteppedWidget } from './BaseSteppedWidget'
import type { WidgetEventOptions } from './BaseWidget'

/**
 * This is used as an (invalid) assertion to resolve issues with legacy duck-typed values.
 *
 * Function style in use by:
 * https://github.com/kijai/ComfyUI-KJNodes/blob/c3dc82108a2a86c17094107ead61d63f8c76200e/web/js/setgetnodes.js#L401-L404
 */
type Values =
  | string[]
  | Record<string, string>
  | ((widget?: ComboWidget, node?: LGraphNode) => string[])

function toArray(values: Values): string[] {
  return Array.isArray(values) ? values : Object.keys(values)
}

export class ComboWidget
  extends BaseSteppedWidget<IStringComboWidget | IComboWidget>
  implements IComboWidget
{
  override type = 'combo' as const

  override get _displayValue() {
    if (this.computedDisabled) return ''

    const getOptionLabel = this.options.getOptionLabel
    if (getOptionLabel) {
      try {
        return getOptionLabel(this.value ? String(this.value) : null)
      } catch (e) {
        console.error('Failed to map value:', e)
        return this.value ? String(this.value) : ''
      }
    }

    const { values: rawValues } = this.options
    if (rawValues) {
      const values = typeof rawValues === 'function' ? rawValues() : rawValues

      if (values && !Array.isArray(values)) {
        return values[this.value]
      }
    }
    return typeof this.value === 'number' ? String(this.value) : this.value
  }

  private getValues(node: LGraphNode): Values {
    const { values } = this.options
    if (values == null) throw new Error('[ComboWidget]: values is required')

    return typeof values === 'function' ? values(this, node) : values
  }

  /**
   * Checks if the value is {@link Array.at at} the given index in the combo list.
   * @param increment `true` if checking the use of the increment button, `false` for decrement
   * @returns `true` if the value is at the given index, otherwise `false`.
   */
  private canUseButton(increment: boolean): boolean {
    const { values } = this.options
    // If using legacy duck-typed method, false is the most permissive return value
    if (typeof values === 'function') return false

    const valuesArray = toArray(values)
    if (!(valuesArray.length > 1)) return false

    // Edge case where the value is both the first and last item in the list
    const firstValue = valuesArray.at(0)
    const lastValue = valuesArray.at(-1)
    if (firstValue === lastValue) return true

    return this.value !== (increment ? lastValue : firstValue)
  }

  /**
   * Returns `true` if the current value is not the last value in the list.
   * Handles edge case where the value is both the first and last item in the list.
   */
  override canIncrement(): boolean {
    return this.canUseButton(true)
  }

  override canDecrement(): boolean {
    return this.canUseButton(false)
  }

  override incrementValue(options: WidgetEventOptions): void {
    this.tryChangeValue(1, options)
  }

  override decrementValue(options: WidgetEventOptions): void {
    this.tryChangeValue(-1, options)
  }

  private tryChangeValue(delta: number, options: WidgetEventOptions): void {
    const values = this.getValues(options.node)
    const indexedValues = toArray(values)

    // avoids double click event
    options.canvas.last_mouseclick = 0

    const foundIndex =
      typeof values === 'object'
        ? indexedValues.indexOf(String(this.value)) + delta
        : // @ts-expect-error handle non-string values
          indexedValues.indexOf(this.value) + delta

    const index = clamp(foundIndex, 0, indexedValues.length - 1)

    const value = Array.isArray(values) ? values[index] : index
    this.setValue(value, options)
  }

  override onClick({ e, node, canvas }: WidgetEventOptions) {
    const x = e.canvasX - node.pos[0]
    const width = this.width || node.size[0]

    // Deprecated functionality (warning as of v0.14.5)
    if (typeof this.options.values === 'function') {
      warnDeprecated(
        'Using a function for values is deprecated. Use an array of unique values instead.'
      )
    }

    // Determine if clicked on left/right arrows
    if (x < 40) return this.decrementValue({ e, node, canvas })
    if (x > width - 40) return this.incrementValue({ e, node, canvas })

    // Otherwise, show dropdown menu
    const values = this.getValues(node)
    const values_list = toArray(values)

    // Use addItem to solve duplicate filename issues
    if (this.options.getOptionLabel) {
      const menuOptions = {
        scale: Math.max(1, canvas.ds.scale),
        event: e,
        className: 'dark',
        callback: (value: string) => {
          this.setValue(value, { e, node, canvas })
        }
      }
      const menu = new LiteGraph.ContextMenu([], menuOptions)

      const getOptionLabel = this.options.getOptionLabel
      for (const value of values_list) {
        try {
          const label = getOptionLabel
            ? getOptionLabel(String(value))
            : String(value)
          menu.addItem(label, value, menuOptions)
        } catch (err) {
          console.error('Failed to map value:', err)
          menu.addItem(String(value), value, menuOptions)
        }
      }
      return
    }

    // Show dropdown menu when user clicks on widget label
    const text_values = values != values_list ? Object.values(values) : values
    new LiteGraph.ContextMenu(text_values, {
      scale: Math.max(1, canvas.ds.scale),
      event: e,
      className: 'dark',
      callback: (value: string) => {
        this.setValue(
          values != values_list ? text_values.indexOf(value) : value,
          { e, node, canvas }
        )
      }
    })
  }
}
