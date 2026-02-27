<template>
  <div role="tablist" class="flex w-full items-center gap-2 pb-1">
    <slot />
  </div>
</template>

<script setup lang="ts" generic="T extends string = string">
import { provide } from 'vue'

const modelValue = defineModel<T>({ required: true })

// Provide for child Tab components
provide('tabs-value', modelValue)
provide('tabs-update', (value: T) => {
  modelValue.value = value
})
</script>
