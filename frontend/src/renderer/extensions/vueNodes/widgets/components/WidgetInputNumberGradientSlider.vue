<template>
  <WidgetLayoutField :widget="widget">
    <div :class="cn(WidgetInputBaseClass, 'flex items-center gap-2 pl-3 pr-2')">
      <GradientSlider
        v-model="modelValue"
        :stops="gradientStops"
        :min="widget.options?.min ?? 0"
        :max="widget.options?.max ?? 100"
        :step="stepValue"
        :disabled="widget.options?.disabled"
        :aria-label="widget.name"
        class="flex-1 min-w-0"
      />
      <InputNumber
        :key="timesEmptied"
        :model-value="modelValue"
        v-bind="filteredProps"
        :step="stepValue"
        :min-fraction-digits="precision"
        :max-fraction-digits="precision"
        :aria-label="widget.name"
        size="small"
        pt:pc-input-text:root="min-w-[4ch] bg-transparent border-none text-center truncate"
        class="w-16 shrink-0"
        :pt="numberPt"
        @update:model-value="handleNumberInputUpdate"
      />
    </div>
  </WidgetLayoutField>
</template>

<script setup lang="ts">
import InputNumber from 'primevue/inputnumber'
import { computed, ref } from 'vue'

import GradientSlider from '@/components/gradientslider/GradientSlider.vue'
import type { ColorStop } from '@/lib/litegraph/src/interfaces'
import type { IWidgetGradientSliderOptions } from '@/lib/litegraph/src/types/widgets'
import type { SimplifiedWidget } from '@/types/simplifiedWidget'
import { cn } from '@/utils/tailwindUtil'
import {
  STANDARD_EXCLUDED_PROPS,
  filterWidgetProps
} from '@/utils/widgetPropFilter'

import { useNumberStepCalculation } from '../composables/useNumberStepCalculation'
import { useNumberWidgetButtonPt } from '../composables/useNumberWidgetButtonPt'
import { WidgetInputBaseClass } from './layout'
import WidgetLayoutField from './layout/WidgetLayoutField.vue'

const DEFAULT_GRADIENT_STOPS: ColorStop[] = [
  { offset: 0, color: [0, 0, 0] },
  { offset: 1, color: [255, 255, 255] }
]

const { widget } = defineProps<{
  widget: SimplifiedWidget<number, IWidgetGradientSliderOptions>
}>()

const modelValue = defineModel<number>({ default: 0 })

const timesEmptied = ref(0)

const handleNumberInputUpdate = (newValue: number | undefined) => {
  if (newValue !== undefined) {
    modelValue.value = newValue
    return
  }
  timesEmptied.value += 1
}

const gradientStops = computed<ColorStop[]>(() => {
  const stops = widget.options?.gradient_stops
  if (stops && stops.length >= 2) return stops
  return DEFAULT_GRADIENT_STOPS
})

const filteredProps = computed(() =>
  filterWidgetProps(widget.options, STANDARD_EXCLUDED_PROPS)
)

const precision = computed(() => {
  const p = widget.options?.precision
  return typeof p === 'number' && p >= 0 ? p : undefined
})

const stepValue = useNumberStepCalculation(widget.options, precision, true)

const numberPt = useNumberWidgetButtonPt({
  roundedLeft: true,
  roundedRight: true
})
</script>
