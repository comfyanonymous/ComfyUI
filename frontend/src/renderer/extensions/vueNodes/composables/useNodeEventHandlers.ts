/**
 * Node Event Handlers Composable
 *
 * Handles all Vue node interaction events including:
 * - Node selection with multi-select support
 * - Node collapse/expand state management
 * - Node title editing and updates
 * - Layout mutations for visual feedback
 * - Integration with LiteGraph canvas selection system
 */
import { createSharedComposable } from '@vueuse/core'

import { useVueNodeLifecycle } from '@/composables/graph/useVueNodeLifecycle'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useCanvasInteractions } from '@/renderer/core/canvas/useCanvasInteractions'
import { useNodeZIndex } from '@/renderer/extensions/vueNodes/composables/useNodeZIndex'
import { isMultiSelectKey } from '@/renderer/extensions/vueNodes/utils/selectionUtils'
import type { NodeId } from '@/renderer/core/layout/types'

function useNodeEventHandlersIndividual() {
  const canvasStore = useCanvasStore()
  const { nodeManager } = useVueNodeLifecycle()
  const { bringNodeToFront } = useNodeZIndex()
  const { shouldHandleNodePointerEvents } = useCanvasInteractions()

  /**
   * Handle node selection events
   * Supports single selection and multi-select with Ctrl/Cmd
   */
  function handleNodeSelect(event: PointerEvent, nodeId: NodeId) {
    if (!shouldHandleNodePointerEvents.value) return

    if (!canvasStore.canvas || !nodeManager.value) return

    const node = nodeManager.value.getNode(nodeId)
    if (!node) return

    const multiSelect = isMultiSelectKey(event)
    const selectedItemsCount = canvasStore.selectedItems.length
    const preserveExistingSelection =
      !multiSelect && node.selected && selectedItemsCount > 1

    if (multiSelect) {
      if (!node.selected) {
        canvasStore.canvas.select(node)
      }
    } else if (!preserveExistingSelection) {
      // Regular click -> single select
      canvasStore.canvas.deselectAll()
      canvasStore.canvas.select(node)
    }

    // Bring node to front when clicked (similar to LiteGraph behavior)
    // Skip if node is pinned to avoid unwanted movement
    if (!node.flags?.pinned) {
      bringNodeToFront(nodeId)
    }

    // Update canvas selection tracking
    canvasStore.updateSelectedItems()
  }

  /**
   * Handle node collapse/expand state changes
   * Uses LiteGraph's native collapse method for proper state management
   */
  function handleNodeCollapse(nodeId: NodeId, collapsed: boolean) {
    if (!shouldHandleNodePointerEvents.value) return

    if (!nodeManager.value) return

    const node = nodeManager.value.getNode(nodeId)
    if (!node) return

    // Use LiteGraph's collapse method if the state needs to change
    const currentCollapsed = node.flags?.collapsed ?? false
    if (currentCollapsed !== collapsed) {
      node.collapse()
    }
  }

  /**
   * Handle node title updates
   * Updates the title in LiteGraph for persistence across sessions
   */
  function handleNodeTitleUpdate(nodeId: NodeId, newTitle: string) {
    if (!shouldHandleNodePointerEvents.value) return

    if (!nodeManager.value) return

    const node = nodeManager.value.getNode(nodeId)
    if (!node) return

    // Update the node title in LiteGraph for persistence
    node.title = newTitle
  }

  /**
   * Handle node right-click context menu events
   * Integrates with LiteGraph's context menu system
   */
  function handleNodeRightClick(event: PointerEvent, nodeId: NodeId) {
    if (!shouldHandleNodePointerEvents.value) return

    if (!canvasStore.canvas || !nodeManager.value) return

    const node = nodeManager.value.getNode(nodeId)
    if (!node) return

    // Prevent default context menu
    event.preventDefault()

    // Select the node if not already selected
    if (!node.selected) {
      handleNodeSelect(event, nodeId)
    }

    // Let LiteGraph handle the context menu
    // The canvas will handle showing the appropriate context menu
  }

  function toggleNodeSelectionAfterPointerUp(
    nodeId: NodeId,
    multiSelect: boolean
  ) {
    if (!shouldHandleNodePointerEvents.value) return

    if (!canvasStore.canvas || !nodeManager.value) return

    const node = nodeManager.value.getNode(nodeId)
    if (!node) return

    if (!multiSelect) {
      canvasStore.canvas.deselectAll()
      canvasStore.canvas.select(node)
      canvasStore.updateSelectedItems()
      // Bring node to front when selected (unless pinned)
      if (!node.flags?.pinned) {
        bringNodeToFront(nodeId)
      }
      return
    }

    if (node.selected) {
      canvasStore.canvas.deselect(node)
    } else {
      canvasStore.canvas.select(node)
      // Bring node to front when selected (unless pinned)
      if (!node.flags?.pinned) {
        bringNodeToFront(nodeId)
      }
    }

    canvasStore.updateSelectedItems()
  }

  return {
    // Core event handlers
    handleNodeSelect,
    handleNodeCollapse,
    handleNodeTitleUpdate,
    handleNodeRightClick,

    // Batch operations
    toggleNodeSelectionAfterPointerUp
  }
}

export const useNodeEventHandlers = createSharedComposable(
  useNodeEventHandlersIndividual
)
