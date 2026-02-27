<template>
  <canvas
    ref="canvasRef"
    class="pointer-events-none absolute inset-0 size-full"
  />
</template>

<script setup lang="ts">
import type { LGraphCanvas } from '@/lib/litegraph/src/LGraphCanvas'

import { useRafFn } from '@vueuse/core'
import { onBeforeUnmount, onMounted, ref } from 'vue'

const { canvas } = defineProps<{
  canvas: LGraphCanvas
}>()

const emit = defineEmits<{
  ready: [canvas: HTMLCanvasElement]
  dispose: []
}>()

const canvasRef = ref<HTMLCanvasElement | null>(null)

useRafFn(() => {
  const el = canvasRef.value
  const mainCanvas = canvas.canvas
  if (!el || !mainCanvas) return

  if (el.width !== mainCanvas.width || el.height !== mainCanvas.height) {
    el.width = mainCanvas.width
    el.height = mainCanvas.height
  }
})

onMounted(() => {
  if (canvasRef.value) emit('ready', canvasRef.value)
})

onBeforeUnmount(() => {
  emit('dispose')
})
</script>
