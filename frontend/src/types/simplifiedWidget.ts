/**
 * Simplified widget interface for Vue-based node rendering
 * Removes all DOM manipulation and positioning concerns
 */
import type { InputSpec as InputSpecV2 } from '@/schemas/nodeDef/nodeDefSchemaV2'
import type { IWidgetOptions } from '@/lib/litegraph/src/types/widgets'

/** Valid types for widget values */
export type WidgetValue =
  | string
  | number
  | boolean
  | object
  | undefined
  | null
  | void
  | File[]

export const CONTROL_OPTIONS = [
  'fixed',
  'increment',
  'decrement',
  'randomize'
] as const
export type ControlOptions = (typeof CONTROL_OPTIONS)[number]

function isControlOption(val: WidgetValue): val is ControlOptions {
  return CONTROL_OPTIONS.includes(val as ControlOptions)
}

export function normalizeControlOption(val: WidgetValue): ControlOptions {
  if (isControlOption(val)) return val
  return 'randomize'
}

export type SafeControlWidget = {
  value: ControlOptions
  update: (value: WidgetValue) => void
}

export interface SimplifiedWidget<
  T extends WidgetValue = WidgetValue,
  O extends IWidgetOptions = IWidgetOptions
> {
  /** Display name of the widget */
  name: string

  /** Widget type identifier (e.g., 'STRING', 'INT', 'COMBO') */
  type: string

  /** Current value of the widget */
  value: T

  borderStyle?: string

  /** Callback fired when value changes */
  callback?: (value: T) => void

  /** Optional method to compute widget size requirements */
  computeSize?: () => { minHeight: number; maxHeight?: number }

  /** Localized display label (falls back to name if not provided) */
  label?: string

  /** Widget options including filtered PrimeVue props */
  options?: O

  /** Override for use with subgraph promoted asset widgets*/
  nodeType?: string

  /** Optional serialization method for custom value handling */
  serializeValue?: () => unknown

  /** Optional input specification backing this widget */
  spec?: InputSpecV2

  controlWidget?: SafeControlWidget
}

export interface SimplifiedControlWidget<
  T extends WidgetValue = WidgetValue,
  O extends IWidgetOptions = IWidgetOptions
> extends SimplifiedWidget<T, O> {
  controlWidget: SafeControlWidget
}
