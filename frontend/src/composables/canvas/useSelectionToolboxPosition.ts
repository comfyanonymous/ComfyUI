import { useElementBounding, useRafFn } from '@vueuse/core'
import { computed, onUnmounted, ref, watch, watchEffect } from 'vue'
import type { Ref } from 'vue'

import { useSelectedLiteGraphItems } from '@/composables/canvas/useSelectedLiteGraphItems'
import { useVueFeatureFlags } from '@/composables/useVueFeatureFlags'
import type { ReadOnlyRect } from '@/lib/litegraph/src/interfaces'
import {
  LGraphGroup,
  LGraphNode,
  LiteGraph
} from '@/lib/litegraph/src/litegraph'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { layoutStore } from '@/renderer/core/layout/store/layoutStore'
import { isLGraphGroup, isLGraphNode } from '@/utils/litegraphUtil'
import { computeUnionBounds } from '@/utils/mathUtil'

/**
 * Manages the position of the selection toolbox independently.
 * Uses CSS custom properties for performant transform updates.
 */

// Shared signals for auxiliary UI (e.g., MoreOptions) to coordinate hide/restore
const moreOptionsOpen = ref(false)
const forceCloseMoreOptionsSignal = ref(0)
const restoreMoreOptionsSignal = ref(0)
const moreOptionsRestorePending = ref(false)
let moreOptionsWasOpenBeforeDrag = false
let moreOptionsSelectionSignature: string | null = null

function buildSelectionSignature(
  store: ReturnType<typeof useCanvasStore>
): string | null {
  const c = store.canvas
  if (!c) return null
  const items = Array.from(c.selectedItems)
  if (items.length !== 1) return null
  const item = items[0]
  if (isLGraphNode(item)) return `N:${item.id}`
  if (isLGraphGroup(item)) return `G:${item.id}`
  return null
}

function currentSelectionMatchesSignature(
  store: ReturnType<typeof useCanvasStore>
) {
  if (!moreOptionsSelectionSignature) return false
  return buildSelectionSignature(store) === moreOptionsSelectionSignature
}

export function useSelectionToolboxPosition(
  toolboxRef: Ref<HTMLElement | undefined>
) {
  const canvasStore = useCanvasStore()
  const lgCanvas = canvasStore.getCanvas()
  const { getSelectableItems } = useSelectedLiteGraphItems()
  const { shouldRenderVueNodes } = useVueFeatureFlags()

  // World position of selection center
  const worldPosition = ref({ x: 0, y: 0 })

  const visible = ref(false)

  // Use VueUse to reactively track canvas bounding rect
  const { left: canvasLeft, top: canvasTop } = useElementBounding(
    lgCanvas.canvas
  )

  // Unified dragging state - combines both LiteGraph and Vue node dragging
  const isDragging = computed((): boolean => {
    const litegraphDragging = canvasStore.canvas?.state?.draggingItems ?? false
    const vueNodeDragging =
      shouldRenderVueNodes.value && layoutStore.isDraggingVueNodes.value
    return litegraphDragging || vueNodeDragging
  })

  /**
   * Update position based on selection
   */
  const updateSelectionBounds = () => {
    const selectableItems = getSelectableItems()

    if (!selectableItems.size) {
      visible.value = false
      return
    }

    // Don't show toolbox while dragging
    if (isDragging.value) {
      visible.value = false
      return
    }

    visible.value = true

    // Get bounds for all selected items
    const allBounds: ReadOnlyRect[] = []
    for (const item of selectableItems) {
      // Skip items without valid IDs
      if (item.id == null) continue

      if (shouldRenderVueNodes.value && typeof item.id === 'string') {
        // Use layout store for Vue nodes (only works with string IDs)
        const layout = layoutStore.getNodeLayoutRef(item.id).value
        if (layout) {
          allBounds.push([
            layout.bounds.x,
            layout.bounds.y,
            layout.bounds.width,
            layout.bounds.height
          ])
        }
      } else {
        // Fallback to LiteGraph bounds for regular nodes or non-string IDs
        if (item instanceof LGraphNode || item instanceof LGraphGroup) {
          allBounds.push([
            item.pos[0],
            item.pos[1] - LiteGraph.NODE_TITLE_HEIGHT,
            item.size[0],
            item.size[1] + LiteGraph.NODE_TITLE_HEIGHT
          ])
        }
      }
    }

    // Compute union bounds
    const unionBounds = computeUnionBounds(allBounds)
    if (!unionBounds) return

    worldPosition.value = {
      x: unionBounds.x + unionBounds.width / 2,
      // createBounds() applied a default padding of 10px
      // so adjust Y to maintain visual consistency
      y: unionBounds.y - 10
    }

    updateTransform()
  }

  const updateTransform = () => {
    if (!visible.value) return

    const { scale, offset } = lgCanvas.ds

    const screenX =
      (worldPosition.value.x + offset[0]) * scale + canvasLeft.value
    const screenY =
      (worldPosition.value.y + offset[1]) * scale + canvasTop.value

    // Update CSS custom properties directly for best performance
    if (toolboxRef.value) {
      toolboxRef.value.style.setProperty('--tb-x', `${screenX}px`)
      toolboxRef.value.style.setProperty('--tb-y', `${screenY}px`)
    }
  }

  // Sync with canvas transform
  const { resume: startSync, pause: stopSync } = useRafFn(updateTransform)

  watchEffect(() => {
    if (visible.value) {
      startSync()
    } else {
      stopSync()
    }
  })

  // Watch for selection changes
  watch(
    () => canvasStore.getCanvas().state.selectionChanged,
    (changed) => {
      if (changed) {
        if (moreOptionsRestorePending.value || moreOptionsSelectionSignature) {
          moreOptionsRestorePending.value = false
          moreOptionsWasOpenBeforeDrag = false
          if (!moreOptionsOpen.value) {
            moreOptionsSelectionSignature = null
          } else {
            moreOptionsSelectionSignature = buildSelectionSignature(canvasStore)
          }
        }
        updateSelectionBounds()
        canvasStore.getCanvas().state.selectionChanged = false
      }
    },
    { immediate: true }
  )
  watch(
    () => moreOptionsOpen.value,
    (v) => {
      if (v) {
        moreOptionsSelectionSignature = buildSelectionSignature(canvasStore)
      } else if (!canvasStore.canvas?.state?.draggingItems) {
        moreOptionsSelectionSignature = null
        if (moreOptionsRestorePending.value)
          moreOptionsRestorePending.value = false
      }
    }
  )

  const handleDragStateChange = (dragging: boolean) => {
    if (dragging) {
      handleDragStart()
      return
    }

    handleDragEnd()
  }

  const handleDragStart = () => {
    visible.value = false

    // Early return if more options wasn't open
    if (!moreOptionsOpen.value) {
      moreOptionsRestorePending.value = false
      moreOptionsWasOpenBeforeDrag = false
      return
    }

    // Handle more options cleanup
    const currentSig = buildSelectionSignature(canvasStore)
    const selectionChanged = currentSig !== moreOptionsSelectionSignature

    if (selectionChanged) {
      moreOptionsSelectionSignature = null
    }
    moreOptionsOpen.value = false
    moreOptionsWasOpenBeforeDrag = true
    moreOptionsRestorePending.value = !!moreOptionsSelectionSignature

    if (moreOptionsRestorePending.value) {
      forceCloseMoreOptionsSignal.value++
      return
    }

    moreOptionsWasOpenBeforeDrag = false
  }

  const handleDragEnd = () => {
    requestAnimationFrame(() => {
      updateSelectionBounds()

      const selectionMatches = currentSelectionMatchesSignature(canvasStore)
      const shouldRestore =
        moreOptionsWasOpenBeforeDrag &&
        visible.value &&
        moreOptionsRestorePending.value &&
        selectionMatches

      // Single point of assignment for each ref
      moreOptionsRestorePending.value =
        shouldRestore && moreOptionsRestorePending.value
      moreOptionsWasOpenBeforeDrag = false

      if (shouldRestore) {
        restoreMoreOptionsSignal.value++
      }
    })
  }

  watch(isDragging, handleDragStateChange)

  onUnmounted(() => {
    resetMoreOptionsState()
  })

  return {
    visible
  }
}

// External cleanup utility to be called when SelectionToolbox component unmounts
function resetMoreOptionsState() {
  moreOptionsOpen.value = false
  moreOptionsRestorePending.value = false
  moreOptionsWasOpenBeforeDrag = false
  moreOptionsSelectionSignature = null
}
