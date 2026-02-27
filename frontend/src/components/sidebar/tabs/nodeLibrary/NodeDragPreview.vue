<template>
  <Teleport to="body">
    <div
      v-if="isDragging && draggedNode && showPreview"
      class="pointer-events-none fixed z-[10000]"
      :style="{
        left: `${previewPosition.x + 12}px`,
        top: `${previewPosition.y + 12}px`
      }"
    >
      <div class="origin-top-left scale-50 opacity-80">
        <LGraphNodePreview :node-def="draggedNode" position="relative" />
      </div>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref } from 'vue'

import { useNodeDragToCanvas } from '@/composables/node/useNodeDragToCanvas'
import LGraphNodePreview from '@/renderer/extensions/vueNodes/components/LGraphNodePreview.vue'

const {
  isDragging,
  draggedNode,
  cursorPosition,
  dragMode,
  setupGlobalListeners,
  cleanupGlobalListeners
} = useNodeDragToCanvas()

const nativeDragPosition = ref({ x: 0, y: 0 })

const previewPosition = computed(() => {
  if (dragMode.value === 'native') {
    return nativeDragPosition.value
  }
  return cursorPosition.value
})

const showPreview = computed(() => {
  if (dragMode.value === 'native') {
    return nativeDragPosition.value.x > 0 || nativeDragPosition.value.y > 0
  }
  return true
})

function handleDrag(e: DragEvent) {
  if (e.clientX === 0 && e.clientY === 0) return
  nativeDragPosition.value = { x: e.clientX, y: e.clientY }
}

function handleDragEnd() {
  nativeDragPosition.value = { x: 0, y: 0 }
}

onMounted(() => {
  setupGlobalListeners()
  document.addEventListener('drag', handleDrag)
  document.addEventListener('dragend', handleDragEnd)
})

onUnmounted(() => {
  cleanupGlobalListeners()
  document.removeEventListener('drag', handleDrag)
  document.removeEventListener('dragend', handleDragEnd)
})
</script>
