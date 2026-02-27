<template>
  <div
    v-if="text"
    class="min-w-max rounded-sm bg-node-component-surface px-1 py-0.5 text-xs flex items-center gap-1"
    :style="{
      color: fgColor,
      backgroundColor: bgColor
    }"
    @click="(e) => onClick?.(e)"
  >
    {{ text }}
    <i v-if="cssIcon" :class="cn(cssIcon)" />
  </div>
</template>
<script setup lang="ts">
import type { LGraphBadge } from '@/lib/litegraph/src/LGraphBadge'
import { cn } from '@/utils/tailwindUtil'

export interface NodeBadgeProps {
  text: LGraphBadge['text']
  fgColor?: LGraphBadge['fgColor']
  bgColor?: LGraphBadge['bgColor']
  cssIcon?: string
  onClick?: (e: MouseEvent) => void
}

const {
  text,
  fgColor = 'currentColor',
  bgColor = '',
  cssIcon
} = defineProps<NodeBadgeProps>()
</script>
