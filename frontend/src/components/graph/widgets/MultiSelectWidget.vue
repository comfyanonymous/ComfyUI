<template>
  <div>
    <MultiSelect
      v-model="selectedItems"
      :options="options"
      filter
      :placeholder="placeholder"
      :max-selected-labels="3"
      :display="display"
      class="w-full"
      :pt="{
        dropdownIcon: 'text-button-icon'
      }"
    />
  </div>
</template>

<script setup lang="ts">
import MultiSelect from 'primevue/multiselect'

import type { ComboInputSpec } from '@/schemas/nodeDef/nodeDefSchemaV2'
import type { ComponentWidget } from '@/scripts/domWidget'

const selectedItems = defineModel<string[]>({ required: true })
const { widget } = defineProps<{
  widget: ComponentWidget<string[]>
}>()

const inputSpec = widget.inputSpec as ComboInputSpec
const options = inputSpec.options ?? []
const placeholder = inputSpec.multi_select?.placeholder ?? 'Select items'
const display = inputSpec.multi_select?.chip ? 'chip' : 'comma'
</script>
