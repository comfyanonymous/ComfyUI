import type { CSSProperties, Ref } from 'vue'
import { computed, ref } from 'vue'

import { useNodeDragToCanvas } from '@/composables/node/useNodeDragToCanvas'
import { useSettingStore } from '@/platform/settings/settingStore'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'

const PREVIEW_WIDTH = 200
const PREVIEW_MARGIN = 16

export function useNodePreviewAndDrag(
  nodeDef: Ref<ComfyNodeDefImpl | undefined>,
  options?: { panelRef?: Ref<HTMLElement | null> }
) {
  const { startDrag, handleNativeDrop } = useNodeDragToCanvas()
  const settingStore = useSettingStore()
  const sidebarLocation = computed<'left' | 'right'>(() =>
    settingStore.get('Comfy.Sidebar.Location')
  )

  const previewRef = ref<HTMLElement | null>(null)
  const isHovered = ref(false)
  const isDragging = ref(false)
  const showPreview = computed(() => isHovered.value && !isDragging.value)

  const nodePreviewStyle = ref<CSSProperties>({
    position: 'fixed',
    top: '0px',
    left: '0px',
    pointerEvents: 'none',
    zIndex: 1000
  })

  function calculatePreviewPosition(rect: DOMRect) {
    const viewportHeight = window.innerHeight
    const viewportWidth = window.innerWidth

    let left: number
    if (sidebarLocation.value === 'left') {
      left = rect.right + PREVIEW_MARGIN
      if (left + PREVIEW_WIDTH > viewportWidth) {
        left = rect.left - PREVIEW_MARGIN - PREVIEW_WIDTH
      }
    } else {
      left = rect.left - PREVIEW_MARGIN - PREVIEW_WIDTH
      if (left < 0) {
        left = rect.right + PREVIEW_MARGIN
      }
    }

    return { left, viewportHeight }
  }

  function handleMouseEnter(e: MouseEvent) {
    if (!nodeDef.value) return

    const target = e.currentTarget as HTMLElement
    const rect = target.getBoundingClientRect()
    const horizontalRect =
      options?.panelRef?.value?.getBoundingClientRect() ?? rect
    const { left, viewportHeight } = calculatePreviewPosition(horizontalRect)

    let top = rect.top

    nodePreviewStyle.value = {
      position: 'fixed',
      top: `${top}px`,
      left: `${left}px`,
      pointerEvents: 'none',
      zIndex: 1000,
      opacity: 0
    }
    isHovered.value = true

    requestAnimationFrame(() => {
      if (previewRef.value) {
        const previewRect = previewRef.value.getBoundingClientRect()
        const previewHeight = previewRect.height

        const mouseY = rect.top + rect.height / 2
        top = mouseY - previewHeight * 0.3

        const minTop = PREVIEW_MARGIN
        const maxTop = viewportHeight - previewHeight - PREVIEW_MARGIN
        top = Math.max(minTop, Math.min(top, maxTop))

        nodePreviewStyle.value = {
          ...nodePreviewStyle.value,
          top: `${top}px`,
          opacity: 1
        }
      }
    })
  }

  function handleMouseLeave() {
    isHovered.value = false
  }

  function createEmptyDragImage(): HTMLElement {
    const el = document.createElement('div')
    el.style.position = 'absolute'
    el.style.left = '-9999px'
    el.style.top = '-9999px'
    el.style.width = '1px'
    el.style.height = '1px'
    return el
  }

  function handleDragStart(e: DragEvent) {
    if (!nodeDef.value) return

    isDragging.value = true
    isHovered.value = false

    startDrag(nodeDef.value, 'native')

    if (e.dataTransfer) {
      e.dataTransfer.effectAllowed = 'copy'
      e.dataTransfer.setData('application/x-comfy-node', nodeDef.value.name)

      const dragImage = createEmptyDragImage()
      document.body.appendChild(dragImage)
      e.dataTransfer.setDragImage(dragImage, 0, 0)

      requestAnimationFrame(() => {
        document.body.removeChild(dragImage)
      })
    }
  }

  function handleDragEnd(e: DragEvent) {
    isDragging.value = false
    handleNativeDrop(e.clientX, e.clientY)
  }

  return {
    previewRef,
    isHovered,
    isDragging,
    showPreview,
    nodePreviewStyle,
    sidebarLocation,
    handleMouseEnter,
    handleMouseLeave,
    handleDragStart,
    handleDragEnd
  }
}
