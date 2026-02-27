<template>
  <span class="relative inline-flex items-center justify-center size-[1em]">
    <i :class="mainIcon" class="text-[1em]" />
    <i
      :class="
        cn(
          subIcon,
          'absolute leading-none pointer-events-none',
          positionX === 'left' ? 'left-0' : 'right-0',
          positionY === 'top' ? 'top-0' : 'bottom-0'
        )
      "
      :style="subIconStyle"
    />
  </span>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import { cn } from '@/utils/tailwindUtil'

type Position = 'top' | 'bottom' | 'left' | 'right'

export interface OverlayIconProps {
  mainIcon: string
  subIcon: string
  positionX?: Position
  positionY?: Position
  offsetX?: number
  offsetY?: number
  subIconScale?: number
}
const {
  mainIcon,
  subIcon,
  positionX = 'right',
  positionY = 'bottom',
  offsetX = 0,
  offsetY = 0,
  subIconScale = 0.6
} = defineProps<OverlayIconProps>()

const textShadow = [
  `-1px -1px 0 rgba(0, 0, 0, 0.7)`,
  `1px -1px 0 rgba(0, 0, 0, 0.7)`,
  `-1px 1px 0 rgba(0, 0, 0, 0.7)`,
  `1px 1px 0 rgba(0, 0, 0, 0.7)`,
  `-1px 0 0 rgba(0, 0, 0, 0.7)`,
  `1px 0 0 rgba(0, 0, 0, 0.7)`,
  `0 -1px 0 rgba(0, 0, 0, 0.7)`,
  `0 1px 0 rgba(0, 0, 0, 0.7)`
].join(', ')

const subIconStyle = computed(() => ({
  fontSize: `${subIconScale}em`,
  textShadow,
  ...(offsetX !== 0 && {
    [positionX === 'left' ? 'left' : 'right']: `${offsetX}px`
  }),
  ...(offsetY !== 0 && {
    [positionY === 'top' ? 'top' : 'bottom']: `${offsetY}px`
  })
}))
</script>
