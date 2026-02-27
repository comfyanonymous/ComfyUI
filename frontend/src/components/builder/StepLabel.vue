<template>
  <div class="flex flex-col gap-0.5 text-left">
    <span class="text-sm font-medium text-base-foreground">
      {{ step.title }}
    </span>
    <span
      class="hidden whitespace-nowrap text-xs text-muted-foreground sm:inline"
    >
      {{ step.subtitle }}
    </span>
  </div>
</template>

<script setup lang="ts">
import type { BuilderToolbarStep } from './types'

const { step } = defineProps<{
  step: BuilderToolbarStep
}>()
</script>
