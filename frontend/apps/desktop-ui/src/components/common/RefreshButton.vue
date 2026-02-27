<!--
  A refresh button that disables and shows a progress spinner whilst active.

  Usage:
  ```vue
    <RefreshButton
      v-model="isRefreshing"
      :outlined="false"
      @refresh="refresh"
    />
  ```
-->
<template>
  <Button
    class="relative p-button-icon-only"
    :outlined="outlined"
    :severity="severity"
    :disabled="active || disabled"
    @click="(event) => $emit('refresh', event)"
  >
    <span
      class="p-button-icon pi pi-refresh transition-all"
      :class="{ 'opacity-0': active }"
      data-pc-section="icon"
    />
    <span class="p-button-label" data-pc-section="label">&nbsp;</span>
    <ProgressSpinner v-show="active" class="absolute w-1/2 h-1/2" />
  </Button>
</template>

<script setup lang="ts">
import Button from 'primevue/button'
import ProgressSpinner from 'primevue/progressspinner'

import type { PrimeVueSeverity } from '@/types/primeVueTypes'

const {
  disabled,
  outlined = true,
  severity = 'secondary'
} = defineProps<{
  disabled?: boolean
  outlined?: boolean
  severity?: PrimeVueSeverity
}>()

// Model
const active = defineModel<boolean>({ required: true })

// Emits
defineEmits(['refresh'])
</script>
