<template>
  <div
    v-if="isVisible"
    class="pointer-events-none absolute z-9999 border border-blue-400 bg-blue-500/20"
    :style="rectangleStyle"
  />
</template>

<script setup lang="ts">
import { useRafFn } from '@vueuse/core'
import { computed, ref } from 'vue'

import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'

const canvasStore = useCanvasStore()

const selectionRect = ref<{
  x: number
  y: number
  w: number
  h: number
} | null>(null)

useRafFn(() => {
  const canvas = canvasStore.canvas
  if (!canvas) {
    selectionRect.value = null
    return
  }

  const { pointer, dragging_rectangle } = canvas

  if (dragging_rectangle && pointer.eDown && pointer.eMove) {
    const x = pointer.eDown.safeOffsetX
    const y = pointer.eDown.safeOffsetY
    const w = pointer.eMove.safeOffsetX - x
    const h = pointer.eMove.safeOffsetY - y

    selectionRect.value = { x, y, w, h }
  } else {
    selectionRect.value = null
  }
})

const isVisible = computed(() => selectionRect.value !== null)

const rectangleStyle = computed(() => {
  const rect = selectionRect.value
  if (!rect) return {}

  const left = rect.w >= 0 ? rect.x : rect.x + rect.w
  const top = rect.h >= 0 ? rect.y : rect.y + rect.h
  const width = Math.abs(rect.w)
  const height = Math.abs(rect.h)

  return {
    left: `${left}px`,
    top: `${top}px`,
    width: `${width}px`,
    height: `${height}px`
  }
})
</script>
