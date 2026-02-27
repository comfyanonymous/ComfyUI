<template>
  <div
    data-testid="transform-pane"
    :class="
      cn(
        'absolute inset-0 w-full h-full pointer-events-none',
        isInteracting ? 'transform-pane--interacting' : 'will-change-auto'
      )
    "
    :style="transformStyle"
  >
    <!-- Vue nodes will be rendered here -->
    <slot />
  </div>
</template>

<script setup lang="ts">
import { useRafFn } from '@vueuse/core'
import { computed } from 'vue'

import type { LGraphCanvas } from '@/lib/litegraph/src/litegraph'
import { useTransformSettling } from '@/renderer/core/layout/transform/useTransformSettling'
import { useTransformState } from '@/renderer/core/layout/transform/useTransformState'
import { cn } from '@/utils/tailwindUtil'

interface TransformPaneProps {
  canvas?: LGraphCanvas
}

const props = defineProps<TransformPaneProps>()

const { transformStyle, syncWithCanvas } = useTransformState()

const canvasElement = computed(() => props.canvas?.canvas)
const { isTransforming: isInteracting } = useTransformSettling(canvasElement, {
  settleDelay: 16
})

useRafFn(
  () => {
    if (!props.canvas) {
      return
    }
    syncWithCanvas(props.canvas)
  },
  { immediate: true }
)
</script>

<style scoped>
.transform-pane--interacting {
  will-change: transform;
}
</style>
