<template>
  <WidgetLayoutField :widget="widget">
    <InputText
      v-model="modelValue"
      v-bind="filteredProps"
      :class="cn(WidgetInputBaseClass, 'w-full text-xs py-2 px-4')"
      :aria-label="widget.name"
      size="small"
      :pt="{ root: 'truncate min-w-[4ch]' }"
    />
  </WidgetLayoutField>
</template>

<script setup lang="ts">
import InputText from 'primevue/inputtext'
import { computed } from 'vue'

import type { SimplifiedWidget } from '@/types/simplifiedWidget'
import { cn } from '@/utils/tailwindUtil'
import {
  INPUT_EXCLUDED_PROPS,
  filterWidgetProps
} from '@/utils/widgetPropFilter'

import { WidgetInputBaseClass } from './layout'
import WidgetLayoutField from './layout/WidgetLayoutField.vue'

const props = defineProps<{
  widget: SimplifiedWidget<string>
}>()

const modelValue = defineModel<string>({ default: '' })

const filteredProps = computed(() =>
  filterWidgetProps(props.widget.options, INPUT_EXCLUDED_PROPS)
)
</script>
