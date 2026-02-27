<script setup lang="ts">
import ToggleSwitch from 'primevue/toggleswitch'

import LayoutField from './LayoutField.vue'

defineProps<{
  label: string
  tooltip?: string
}>()

const modelValue = defineModel<boolean>({ default: false })
</script>

<template>
  <LayoutField singleline :label :tooltip>
    <ToggleSwitch
      v-model="modelValue"
      class="transition-transform active:scale-90"
    />
  </LayoutField>
</template>
