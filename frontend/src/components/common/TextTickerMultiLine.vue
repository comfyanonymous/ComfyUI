<template>
  <div>
    <!-- Hidden single-line measurement element for overflow detection -->
    <div
      ref="measureRef"
      class="invisible absolute inset-x-0 top-0 overflow-hidden whitespace-nowrap pointer-events-none"
      aria-hidden="true"
    >
      <slot />
    </div>

    <MarqueeLine v-if="!secondLine">
      <slot />
    </MarqueeLine>

    <div v-else class="flex flex-col w-full">
      <MarqueeLine>{{ firstLine }}</MarqueeLine>
      <MarqueeLine>{{ secondLine }}</MarqueeLine>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useMutationObserver, useResizeObserver } from '@vueuse/core'
import { ref } from 'vue'

import MarqueeLine from './MarqueeLine.vue'
import { splitTextAtWordBoundary } from '@/utils/textTickerUtils'

const measureRef = ref<HTMLElement | null>(null)
const firstLine = ref('')
const secondLine = ref('')

function splitLines() {
  const el = measureRef.value
  const text = el?.textContent?.trim()
  if (!el || !text) {
    firstLine.value = ''
    secondLine.value = ''
    return
  }

  const containerWidth = el.clientWidth
  const textWidth = el.scrollWidth

  if (textWidth <= containerWidth) {
    firstLine.value = text
    secondLine.value = ''
    return
  }

  const [first, second] = splitTextAtWordBoundary(
    text,
    containerWidth / textWidth
  )
  firstLine.value = first
  secondLine.value = second
}

useResizeObserver(measureRef, splitLines)
useMutationObserver(measureRef, splitLines, {
  childList: true,
  characterData: true,
  subtree: true
})
</script>
