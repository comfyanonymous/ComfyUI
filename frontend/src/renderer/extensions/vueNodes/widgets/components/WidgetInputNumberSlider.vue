<template>
  <WidgetLayoutField :widget="widget">
    <div :class="cn(WidgetInputBaseClass, 'flex items-center gap-2 pl-3 pr-2')">
      <Slider
        :model-value="[modelValue]"
        v-bind="filteredProps"
        class="flex-grow text-xs"
        :step="stepValue"
        :aria-label="widget.name"
        @update:model-value="updateLocalValue"
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
        class="w-16"
        :pt="sliderNumberPt"
        @update:model-value="handleNumberInputUpdate"
      />
    </div>
  </WidgetLayoutField>
</template>

<script setup lang="ts">
import InputNumber from 'primevue/inputnumber'
import { computed, ref } from 'vue'

import Slider from '@/components/ui/slider/Slider.vue'
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

const { widget } = defineProps<{
  widget: SimplifiedWidget<number>
}>()

const modelValue = defineModel<number>({ default: 0 })

const timesEmptied = ref(0)

const updateLocalValue = (newValue: number[] | undefined): void => {
  if (newValue?.length) modelValue.value = newValue[0]
}

const handleNumberInputUpdate = (newValue: number | undefined) => {
  if (newValue !== undefined) {
    updateLocalValue([newValue])
    return
  }
  timesEmptied.value += 1
}

const filteredProps = computed(() =>
  filterWidgetProps(widget.options, STANDARD_EXCLUDED_PROPS)
)

const p = widget.options?.precision
const precision = typeof p === 'number' && p >= 0 ? p : undefined

// Calculate the step value based on precision or widget options
const stepValue = useNumberStepCalculation(widget.options, precision, true)

const sliderNumberPt = useNumberWidgetButtonPt({
  roundedLeft: true,
  roundedRight: true
})
</script>

<style scoped>
:deep(.p-inputnumber-button.p-disabled .pi),
:deep(.p-inputnumber-button.p-disabled .p-icon) {
  color: var(--color-node-icon-disabled) !important;
}
</style>
