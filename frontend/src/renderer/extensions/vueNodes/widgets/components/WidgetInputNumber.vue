<script setup lang="ts">
import { computed } from 'vue'

import type {
  SimplifiedControlWidget,
  SimplifiedWidget
} from '@/types/simplifiedWidget'

import WidgetInputNumberGradientSlider from './WidgetInputNumberGradientSlider.vue'
import WidgetInputNumberInput from './WidgetInputNumberInput.vue'
import WidgetInputNumberSlider from './WidgetInputNumberSlider.vue'
import WidgetWithControl from './WidgetWithControl.vue'

const props = defineProps<{
  widget: SimplifiedWidget<number>
}>()

const modelValue = defineModel<number>({ default: 0 })

const controlWidget = computed<SimplifiedControlWidget<number> | null>(() =>
  props.widget.controlWidget
    ? (props.widget as SimplifiedControlWidget<number>)
    : null
)

const widgetComponent = computed(() => {
  switch (props.widget.type) {
    case 'gradientslider':
      return WidgetInputNumberGradientSlider
    case 'slider':
      return WidgetInputNumberSlider
    default:
      return WidgetInputNumberInput
  }
})
</script>

<template>
  <WidgetWithControl
    v-if="controlWidget"
    v-model="modelValue"
    :widget="controlWidget"
    :component="widgetComponent"
  />
  <component
    :is="widgetComponent"
    v-else
    v-model="modelValue"
    :widget="widget"
    v-bind="$attrs"
  />
</template>
