<template>
  <div
    :class="
      cn(
        'flex justify-center items-center shrink-0 outline-hidden border-none p-0 rounded-lg shadow-sm transition-all duration-200 cursor-pointer bg-secondary-background',
        backgroundClass
      )
    "
  >
    <slot></slot>
  </div>
</template>

<script setup lang="ts">
import { cn } from '@/utils/tailwindUtil'

const { backgroundClass } = defineProps<{
  backgroundClass?: string
}>()
</script>
