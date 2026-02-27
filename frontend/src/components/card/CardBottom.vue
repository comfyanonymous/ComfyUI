<template>
  <div :class="containerClasses">
    <slot></slot>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import { cn } from '@/utils/tailwindUtil'

const { fullHeight = true } = defineProps<{
  fullHeight?: boolean
}>()

const containerClasses = computed(() =>
  cn('flex-1 w-full', fullHeight && 'h-full')
)
</script>
