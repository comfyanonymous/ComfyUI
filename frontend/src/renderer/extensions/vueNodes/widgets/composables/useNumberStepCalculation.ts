import { computed, toValue } from 'vue'
import type { MaybeRefOrGetter } from 'vue'

interface NumberWidgetOptions {
  step?: number
  step2?: number
  precision?: number
}

/**
 * Shared composable for calculating step values in number input widgets
 * Handles both explicit step2 values and precision-derived steps
 */
export function useNumberStepCalculation(
  options: NumberWidgetOptions | undefined,
  precisionArg: MaybeRefOrGetter<number | undefined>,
  returnUndefinedForDefault = false
) {
  return computed(() => {
    const precision = toValue(precisionArg)
    // Use step2 (correct input spec value) if available
    if (options?.step2 !== undefined) {
      return Number(options.step2)
    }
    // Use step / 10 for custom large step values (> 10) to match litegraph behavior
    // This is important for extensions like Impact Pack that use custom step values (e.g., 640)
    // We skip default step values (1, 10) to avoid affecting normal widgets
    const step = options?.step
    if (step !== undefined && step > 10) {
      return Number(step) / 10
    }

    if (precision === undefined) {
      return returnUndefinedForDefault ? undefined : 0
    }

    if (precision === 0) return 1

    // For precision > 0, step = 1 / (10^precision)
    const calculatedStep = 1 / Math.pow(10, precision)
    return returnUndefinedForDefault
      ? calculatedStep
      : Number(calculatedStep.toFixed(precision))
  })
}
