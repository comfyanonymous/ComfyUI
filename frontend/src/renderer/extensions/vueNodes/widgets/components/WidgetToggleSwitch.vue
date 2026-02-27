<template>
  <WidgetLayoutField :widget>
    <!-- Use ToggleGroup when explicit labels are provided -->
    <ToggleGroup
      v-if="hasLabels"
      type="single"
      :model-value="modelValue ? 'on' : 'off'"
      :disabled="Boolean(widget.options?.read_only)"
      :class="
        cn(
          WidgetInputBaseClass,
          'w-full min-w-0 p-1 flex items-center justify-center gap-1'
        )
      "
      @update:model-value="(v) => handleOptionChange(v as string)"
    >
      <ToggleGroupItem value="off" size="sm">
        {{ widget.options?.off ?? t('widgets.boolean.false') }}
      </ToggleGroupItem>
      <ToggleGroupItem value="on" size="sm">
        {{ widget.options?.on ?? t('widgets.boolean.true') }}
      </ToggleGroupItem>
    </ToggleGroup>

    <!-- Use ToggleSwitch for implicit boolean states -->
    <div
      v-else
      :class="cn('flex w-fit items-center gap-2', hideLayoutField || 'ml-auto')"
    >
      <ToggleSwitch
        v-model="modelValue"
        v-bind="filteredProps"
        :aria-label="widget.name"
      />
    </div>
  </WidgetLayoutField>
</template>

<script setup lang="ts">
import ToggleSwitch from 'primevue/toggleswitch'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import type { IWidgetOptions } from '@/lib/litegraph/src/types/widgets'
import type { SimplifiedWidget } from '@/types/simplifiedWidget'
import { useHideLayoutField } from '@/types/widgetTypes'
import { cn } from '@/utils/tailwindUtil'
import {
  STANDARD_EXCLUDED_PROPS,
  filterWidgetProps
} from '@/utils/widgetPropFilter'

import { WidgetInputBaseClass } from './layout'
import WidgetLayoutField from './layout/WidgetLayoutField.vue'

const { widget } = defineProps<{
  widget: SimplifiedWidget<boolean, IWidgetOptions>
}>()

const modelValue = defineModel<boolean>()

const hideLayoutField = useHideLayoutField()
const { t } = useI18n()

const filteredProps = computed(() =>
  filterWidgetProps(widget.options, STANDARD_EXCLUDED_PROPS)
)

const hasLabels = computed(() => {
  return widget.options?.on != null || widget.options?.off != null
})

function handleOptionChange(value: string | undefined) {
  if (value) {
    modelValue.value = value === 'on'
  }
}
</script>
