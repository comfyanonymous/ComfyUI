<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import ScrubableNumberInput from '@/components/common/ScrubableNumberInput.vue'
import { evaluateInput } from '@/lib/litegraph/src/utils/widget'
import type { SimplifiedWidget } from '@/types/simplifiedWidget'
import { cn } from '@/utils/tailwindUtil'
import {
  INPUT_EXCLUDED_PROPS,
  filterWidgetProps
} from '@/utils/widgetPropFilter'

import { WidgetInputBaseClass } from './layout'
import WidgetLayoutField from './layout/WidgetLayoutField.vue'

const { locale } = useI18n()

const props = defineProps<{
  widget: SimplifiedWidget<number>
  rootClass?: string
}>()

function formatNumber(value: number, options?: Intl.NumberFormatOptions) {
  return new Intl.NumberFormat(locale.value, options).format(value)
}

const decimalSeparator = computed(() =>
  formatNumber(1.1).replace(/\p{Number}/gu, '')
)
const groupSeparator = computed(() =>
  formatNumber(11111).replace(/\p{Number}/gu, '')
)
function unformatValue(value: string) {
  return value
    .replaceAll(groupSeparator.value, '')
    .replaceAll(decimalSeparator.value, '.')
}

const modelValue = defineModel<number>({ default: 0 })

const formattedValue = computed(() => {
  const value = modelValue.value
  if ((value as unknown) === '' || !isFinite(value)) return `${value}`

  const options: Intl.NumberFormatOptions = {
    useGrouping: useGrouping.value
  }
  if (precision.value !== undefined) {
    options.minimumFractionDigits = precision.value
    options.maximumFractionDigits = precision.value
  }
  return formatNumber(value, options)
})

function parseWidgetValue(raw: string): number | undefined {
  return evaluateInput(unformatValue(raw))
}

interface NumericWidgetOptions {
  min: number
  max: number
  step?: number
  step2?: number
  precision?: number
  disabled?: boolean
  useGrouping?: boolean
}

const filteredProps = computed(() => {
  const filtered = filterWidgetProps(props.widget.options, INPUT_EXCLUDED_PROPS)
  return filtered as Partial<NumericWidgetOptions>
})

const isDisabled = computed(() => props.widget.options?.disabled ?? false)

// Get the precision value for proper number formatting
const precision = computed(() => {
  const p = props.widget.options?.precision
  // Treat negative or non-numeric precision as undefined
  return typeof p === 'number' && p >= 0 ? p : undefined
})

// Calculate the step value based on precision or widget options
const stepValue = computed(() => {
  // Use step2 (correct input spec value) if available
  if (props.widget.options?.step2 !== undefined) {
    return Number(props.widget.options.step2)
  }
  // Use step / 10 for custom large step values (> 10) to match litegraph behavior
  // This is important for extensions like Impact Pack that use custom step values (e.g., 640)
  // We skip default step values (1, 10) to avoid affecting normal widgets
  const step = props.widget.options?.step as number | undefined
  if (step !== undefined && step > 10) {
    return Number(step) / 10
  }
  // Otherwise, derive from precision
  if (precision.value !== undefined) {
    if (precision.value === 0) {
      return 1
    }
    // For precision > 0, step = 1 / (10^precision)
    // precision 1 → 0.1, precision 2 → 0.01, etc.
    return Number((1 / Math.pow(10, precision.value)).toFixed(precision.value))
  }
  // Default to 'any' for unrestricted stepping
  return 0
})

// Disable grouping separators by default unless explicitly enabled by the node author
const useGrouping = computed(() => {
  return props.widget.options?.useGrouping === true
})

// Check if increment/decrement buttons should be disabled due to precision limits
const buttonsDisabled = computed(() => {
  const currentValue = modelValue.value ?? 0
  return (
    !Number.isFinite(currentValue) ||
    Math.abs(currentValue) > Number.MAX_SAFE_INTEGER
  )
})

function updateValueBy(delta: number) {
  const max = filteredProps.value.max ?? Number.MAX_VALUE
  const min = filteredProps.value.min ?? -Number.MAX_VALUE
  modelValue.value = Math.min(max, Math.max(min, modelValue.value + delta))
}

const buttonTooltip = computed(() => {
  if (buttonsDisabled.value) {
    return 'Increment/decrement disabled: value exceeds JavaScript precision limit (±2^53)'
  }
  return null
})

const sliderWidth = computed(() => {
  const { max, min, step } = filteredProps.value
  if (
    min === undefined ||
    max === undefined ||
    step === undefined ||
    (max - min) / step >= 100
  )
    return 0
  const ratio = (modelValue.value - min) / (max - min)
  return (ratio * 100).toFixed(0)
})

const inputAriaAttrs = computed(() => ({
  'aria-valuenow': modelValue.value,
  'aria-valuemin': filteredProps.value.min,
  'aria-valuemax': filteredProps.value.max,
  role: 'spinbutton',
  tabindex: 0
}))
</script>

<template>
  <WidgetLayoutField :widget :root-class="props.rootClass">
    <ScrubableNumberInput
      v-model="modelValue"
      v-tooltip="buttonTooltip"
      :aria-label="widget.name"
      :min="filteredProps.min"
      :max="filteredProps.max"
      :step="stepValue"
      :display-value="formattedValue"
      :disabled="isDisabled"
      :hide-buttons="buttonsDisabled"
      :parse-value="parseWidgetValue"
      :input-attrs="inputAriaAttrs"
      :class="cn(WidgetInputBaseClass, 'grow text-xs flex h-7 relative')"
      @keydown.up.prevent="updateValueBy(stepValue)"
      @keydown.down.prevent="updateValueBy(-stepValue)"
      @keydown.page-up.prevent="updateValueBy(10 * stepValue)"
      @keydown.page-down.prevent="updateValueBy(-10 * stepValue)"
    >
      <template #background>
        <div
          class="absolute size-full rounded-lg pointer-events-none overflow-clip"
        >
          <div
            class="bg-primary-background/15 size-full"
            :style="{ width: `${sliderWidth}%` }"
          />
        </div>
      </template>
      <slot />
    </ScrubableNumberInput>
  </WidgetLayoutField>
</template>
