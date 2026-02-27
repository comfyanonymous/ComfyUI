<template>
  <div
    ref="containerRef"
    :class="
      cn('overflow-hidden whitespace-nowrap', !isScrolling && 'text-ellipsis')
    "
  >
    <slot />
  </div>
</template>

<script setup lang="ts">
import { useElementHover, useElementSize, useRafFn } from '@vueuse/core'
import { ref, watch } from 'vue'

import { cn } from '@/utils/tailwindUtil'

const { speed = 70 } = defineProps<{
  /** Scroll speed in pixels per second */
  speed?: number
}>()

const containerRef = ref<HTMLElement | null>(null)
const isHovered = useElementHover(containerRef, { delayEnter: 350 })
const { width: containerWidth } = useElementSize(containerRef)
const isScrolling = ref(false)
let scrollStartTime = 0
let overflowAmount = 0

const { pause, resume } = useRafFn(
  ({ timestamp }) => {
    const el = containerRef.value
    if (!el || overflowAmount <= 0) return pause()

    const elapsed = timestamp - scrollStartTime
    const duration = (overflowAmount / speed) * 1000
    const progress = Math.min(elapsed / duration, 1)
    el.scrollLeft = overflowAmount * progress

    if (progress >= 1) pause()
  },
  { immediate: false }
)

function startScroll() {
  const el = containerRef.value
  if (!el) return

  overflowAmount = el.scrollWidth - containerWidth.value
  if (overflowAmount <= 0) return

  isScrolling.value = true
  scrollStartTime = performance.now()
  resume()
}

function stopScroll() {
  pause()
  if (containerRef.value) {
    containerRef.value.scrollLeft = 0
  }
  isScrolling.value = false
}

watch(isHovered, (hovered) => {
  if (hovered) startScroll()
  else stopScroll()
})
</script>
