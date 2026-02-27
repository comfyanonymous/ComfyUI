<script setup lang="ts">
import { computed, ref, useTemplateRef } from 'vue'

const zoomPane = useTemplateRef('zoomPane')

const zoom = ref(1.0)
const panX = ref(0.0)
const panY = ref(0.0)

function handleWheel(e: WheelEvent) {
  const zoomPaneEl = zoomPane.value
  if (!zoomPaneEl) return

  zoom.value -= e.deltaY
  const { x, y, width, height } = zoomPaneEl.getBoundingClientRect()
  const offsetX = e.clientX - x - width / 2
  const offsetY = e.clientY - y - height / 2
  const scaler = 1.1 ** (e.deltaY / -30)

  panY.value = panY.value * scaler - offsetY * (scaler - 1)
  panX.value = panX.value * scaler - offsetX * (scaler - 1)
}

let dragging = false
function handleDown(e: PointerEvent) {
  if (e.button !== 0) return

  const zoomPaneEl = zoomPane.value
  if (!zoomPaneEl) return
  zoomPaneEl.parentElement?.focus()

  zoomPaneEl.setPointerCapture(e.pointerId)
  dragging = true
}
function handleMove(e: PointerEvent) {
  if (!dragging) return
  panX.value += e.movementX
  panY.value += e.movementY
}

const transform = computed(() => {
  const scale = 1.1 ** (zoom.value / 30)
  const matrix = [scale, 0, 0, scale, panX.value, panY.value]
  return `matrix(${matrix.join(',')})`
})
</script>
<template>
  <div
    ref="zoomPane"
    class="contain-size place-content-center"
    @wheel="handleWheel"
    @pointerdown.prevent="handleDown"
    @pointermove="handleMove"
    @pointerup="dragging = false"
    @pointercancel="dragging = false"
  >
    <slot :style="{ transform }" />
  </div>
</template>
