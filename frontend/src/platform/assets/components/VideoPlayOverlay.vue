<template>
  <template v-if="visible">
    <div :class="cn('pointer-events-none absolute inset-0', overlayClass)" />
    <div
      class="pointer-events-none absolute inset-0 flex items-center justify-center"
    >
      <i
        aria-hidden="true"
        :class="cn('icon-[lucide--play] text-white', iconSizeClass)"
      />
    </div>
  </template>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import { cn } from '@/utils/tailwindUtil'

const {
  visible = true,
  size = 'md',
  overlayClass = 'bg-black/15'
} = defineProps<{
  visible?: boolean
  size?: 'sm' | 'md'
  overlayClass?: string
}>()

const iconSizeClass = computed(() => (size === 'sm' ? 'size-3' : 'size-6'))
</script>
