<template>
  <div
    ref="pointerZoneRef"
    class="w-[calc(100%-4rem-220px)] h-full"
    @pointerdown="handlePointerDown"
    @pointermove="handlePointerMove"
    @pointerup="handlePointerUp"
    @pointerleave="handlePointerLeave"
    @pointerenter="handlePointerEnter"
    @touchstart="handleTouchStart"
    @touchmove="handleTouchMove"
    @touchend="handleTouchEnd"
    @wheel="handleWheel"
    @contextmenu.prevent
  />
</template>

<script setup lang="ts">
import { onMounted, ref, watch } from 'vue'

import type { usePanAndZoom } from '@/composables/maskeditor/usePanAndZoom'
import type { useToolManager } from '@/composables/maskeditor/useToolManager'
import { useMaskEditorStore } from '@/stores/maskEditorStore'

const { toolManager, panZoom } = defineProps<{
  toolManager: ReturnType<typeof useToolManager>
  panZoom: ReturnType<typeof usePanAndZoom>
}>()

const store = useMaskEditorStore()
const pointerZoneRef = ref<HTMLDivElement>()

onMounted(() => {
  if (!pointerZoneRef.value) {
    console.error('[PointerZone] Pointer zone ref not initialized')
    return
  }

  store.pointerZone = pointerZoneRef.value
})

watch(
  () => store.isPanning,
  (isPanning) => {
    if (!pointerZoneRef.value) return

    if (isPanning) {
      pointerZoneRef.value.style.cursor = 'grabbing'
    } else {
      toolManager.updateCursor()
    }
  }
)

const handlePointerDown = async (event: PointerEvent) => {
  await toolManager.handlePointerDown(event)
}

const handlePointerMove = async (event: PointerEvent) => {
  await toolManager.handlePointerMove(event)
}

const handlePointerUp = (event: PointerEvent) => {
  void toolManager.handlePointerUp(event)
}

const handlePointerLeave = () => {
  store.brushVisible = false
  if (pointerZoneRef.value) {
    pointerZoneRef.value.style.cursor = ''
  }
}

const handlePointerEnter = () => {
  toolManager.updateCursor()
}

const handleTouchStart = (event: TouchEvent) => {
  panZoom.handleTouchStart(event)
}

const handleTouchMove = async (event: TouchEvent) => {
  await panZoom.handleTouchMove(event)
}

const handleTouchEnd = (event: TouchEvent) => {
  panZoom.handleTouchEnd(event)
}

const handleWheel = async (event: WheelEvent) => {
  await panZoom.zoom(event)
  const newCursorPoint = { x: event.clientX, y: event.clientY }
  panZoom.updateCursorPosition(newCursorPoint)
}
</script>
