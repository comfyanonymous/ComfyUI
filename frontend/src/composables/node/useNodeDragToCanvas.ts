import { ref, shallowRef } from 'vue'

import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useLitegraphService } from '@/services/litegraphService'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'

type DragMode = 'click' | 'native'

const isDragging = ref(false)
const draggedNode = shallowRef<ComfyNodeDefImpl | null>(null)
const cursorPosition = ref({ x: 0, y: 0 })
const dragMode = ref<DragMode>('click')
let listenersSetup = false

function updatePosition(e: PointerEvent) {
  cursorPosition.value = { x: e.clientX, y: e.clientY }
}

function cancelDrag() {
  isDragging.value = false
  draggedNode.value = null
  dragMode.value = 'click'
}

function addNodeAtPosition(clientX: number, clientY: number): boolean {
  if (!draggedNode.value) return false

  const canvasStore = useCanvasStore()
  const canvas = canvasStore.canvas
  if (!canvas) return false

  const canvasElement = canvas.canvas as HTMLCanvasElement
  const rect = canvasElement.getBoundingClientRect()
  const isOverCanvas =
    clientX >= rect.left &&
    clientX <= rect.right &&
    clientY >= rect.top &&
    clientY <= rect.bottom

  if (isOverCanvas) {
    const pos = canvas.convertEventToCanvasOffset({
      clientX,
      clientY
    } as PointerEvent)
    const litegraphService = useLitegraphService()
    litegraphService.addNodeOnGraph(draggedNode.value, { pos })
    return true
  }
  return false
}

function endDrag(e: PointerEvent) {
  if (!isDragging.value || !draggedNode.value) return
  if (dragMode.value !== 'click') return

  try {
    addNodeAtPosition(e.clientX, e.clientY)
  } finally {
    cancelDrag()
  }
}

function handleKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape') cancelDrag()
}

function setupGlobalListeners() {
  if (listenersSetup) return
  listenersSetup = true

  document.addEventListener('pointermove', updatePosition)
  document.addEventListener('pointerup', endDrag, true)
  document.addEventListener('keydown', handleKeydown)
}

function cleanupGlobalListeners() {
  if (!listenersSetup) return
  listenersSetup = false

  document.removeEventListener('pointermove', updatePosition)
  document.removeEventListener('pointerup', endDrag, true)
  document.removeEventListener('keydown', handleKeydown)

  if (isDragging.value && dragMode.value === 'click') {
    cancelDrag()
  }
}

export function useNodeDragToCanvas() {
  function startDrag(nodeDef: ComfyNodeDefImpl, mode: DragMode = 'click') {
    isDragging.value = true
    draggedNode.value = nodeDef
    dragMode.value = mode
  }

  function handleNativeDrop(clientX: number, clientY: number) {
    if (dragMode.value !== 'native') return
    try {
      addNodeAtPosition(clientX, clientY)
    } finally {
      cancelDrag()
    }
  }

  return {
    isDragging,
    draggedNode,
    cursorPosition,
    dragMode,
    startDrag,
    cancelDrag,
    handleNativeDrop,
    setupGlobalListeners,
    cleanupGlobalListeners
  }
}
