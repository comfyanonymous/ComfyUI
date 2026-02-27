<template>
  <div class="flex h-full shrink-0 items-center">
    <TopbarBadge
      v-for="badge in topbarBadgeStore.badges"
      :key="badge.text"
      :badge
      :display-mode="displayMode"
      :reverse-order="reverseOrder"
      :no-padding="noPadding"
    />
  </div>
</template>

<script lang="ts" setup>
import { breakpointsTailwind, useBreakpoints } from '@vueuse/core'
import { computed } from 'vue'

import { useTopbarBadgeStore } from '@/stores/topbarBadgeStore'

import TopbarBadge from './TopbarBadge.vue'

const { reverseOrder = false, noPadding = false } = defineProps<{
  reverseOrder?: boolean
  noPadding?: boolean
}>()

const breakpoints = useBreakpoints(breakpointsTailwind)
const isXl = breakpoints.greaterOrEqual('xl')
const isLg = breakpoints.greaterOrEqual('lg')

const displayMode = computed<'full' | 'compact' | 'icon-only'>(() => {
  if (isXl.value) return 'full'
  if (isLg.value) return 'compact'
  return 'icon-only'
})

const topbarBadgeStore = useTopbarBadgeStore()
</script>
