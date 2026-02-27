<template>
  <div class="flex h-full shrink-0 items-center gap-1 empty:hidden">
    <Button
      v-for="(button, index) in actionBarButtonStore.buttons"
      :key="index"
      v-tooltip.bottom="button.tooltip"
      :aria-label="button.tooltip || button.label"
      :class="button.class"
      variant="muted-textonly"
      size="sm"
      class="h-7 rounded-full"
      @click="button.onClick"
    >
      <i :class="button.icon" />
      <span v-if="!isMobile && button.label">{{ button.label }}</span>
    </Button>
  </div>
</template>

<script lang="ts" setup>
import { breakpointsTailwind, useBreakpoints } from '@vueuse/core'

import Button from '@/components/ui/button/Button.vue'
import { useActionBarButtonStore } from '@/stores/actionBarButtonStore'

const actionBarButtonStore = useActionBarButtonStore()

const breakpoints = useBreakpoints(breakpointsTailwind)
const isMobile = breakpoints.smaller('sm')
</script>
