<!-- Needs custom color picker for alpha support -->
<template>
  <WidgetLayoutField :widget="widget">
    <label
      :class="
        cn(WidgetInputBaseClass, 'flex items-center gap-2 w-full px-4 py-2')
      "
    >
      <ColorPicker
        v-model="localValue"
        v-bind="filteredProps"
        class="h-4 w-8 overflow-hidden !rounded-full border-none"
        :aria-label="widget.name"
        :pt="{
          preview: '!w-full !h-full !border-none'
        }"
        @update:model-value="onPickerUpdate"
      />
      <span
        class="text-xs truncate min-w-[4ch]"
        data-testid="widget-color-text"
        >{{ toHexFromFormat(localValue, format) }}</span
      >
    </label>
  </WidgetLayoutField>
</template>

<script setup lang="ts">
import ColorPicker from 'primevue/colorpicker'
import { computed, ref, watch } from 'vue'

import type { SimplifiedWidget } from '@/types/simplifiedWidget'
import { isColorFormat, toHexFromFormat } from '@/utils/colorUtil'
import type { ColorFormat, HSB } from '@/utils/colorUtil'
import { cn } from '@/utils/tailwindUtil'
import {
  PANEL_EXCLUDED_PROPS,
  filterWidgetProps
} from '@/utils/widgetPropFilter'

import type { IWidgetOptions } from '@/lib/litegraph/src/types/widgets'

import { WidgetInputBaseClass } from './layout'
import WidgetLayoutField from './layout/WidgetLayoutField.vue'

type WidgetOptions = IWidgetOptions & { format?: ColorFormat }

const props = defineProps<{
  widget: SimplifiedWidget<string, WidgetOptions>
  modelValue: string
}>()

const emit = defineEmits<{
  'update:modelValue': [value: string]
}>()

const format = computed<ColorFormat>(() => {
  const optionFormat = props.widget.options?.format
  return isColorFormat(optionFormat) ? optionFormat : 'hex'
})

type PickerValue = string | HSB
const localValue = ref<PickerValue>(
  toHexFromFormat(
    props.modelValue || '#000000',
    isColorFormat(props.widget.options?.format)
      ? props.widget.options.format
      : 'hex'
  )
)

watch(
  () => props.modelValue,
  (newVal) => {
    localValue.value = toHexFromFormat(newVal || '#000000', format.value)
  }
)

function onPickerUpdate(val: unknown) {
  localValue.value = val as PickerValue
  emit('update:modelValue', toHexFromFormat(val, format.value))
}

// ColorPicker specific excluded props include panel/overlay classes
const COLOR_PICKER_EXCLUDED_PROPS = [...PANEL_EXCLUDED_PROPS] as const

const filteredProps = computed(() =>
  filterWidgetProps(props.widget.options, COLOR_PICKER_EXCLUDED_PROPS)
)
</script>
