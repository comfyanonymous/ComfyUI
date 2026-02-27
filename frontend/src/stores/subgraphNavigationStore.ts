import QuickLRU from '@alloc/quick-lru'
import { defineStore } from 'pinia'
import { computed, ref, shallowRef, watch } from 'vue'

import type { DragAndScaleState } from '@/lib/litegraph/src/DragAndScale'
import type { Subgraph } from '@/lib/litegraph/src/litegraph'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { app } from '@/scripts/app'
import { findSubgraphPathById } from '@/utils/graphTraversalUtil'
import { isNonNullish } from '@/utils/typeGuardUtil'

/**
 * Stores the current subgraph navigation state; a stack representing subgraph
 * navigation history from the root graph to the subgraph that is currently
 * open.
 */
export const useSubgraphNavigationStore = defineStore(
  'subgraphNavigation',
  () => {
    const workflowStore = useWorkflowStore()
    const canvasStore = useCanvasStore()

    /** The currently opened subgraph. */
    const activeSubgraph = shallowRef<Subgraph>()

    /** The stack of subgraph IDs from the root graph to the currently opened subgraph. */
    const idStack = ref<string[]>([])

    /** LRU cache for viewport states. Key: subgraph ID or 'root' for root graph */
    const viewportCache = new QuickLRU<string, DragAndScaleState>({
      maxSize: 32
    })

    /**
     * Get the ID of the root graph for the currently active workflow.
     * @returns The ID of the root graph for the currently active workflow.
     */
    const getCurrentRootGraphId = () => {
      const canvas = canvasStore.getCanvas()
      if (!canvas) return 'root'

      return canvas.graph?.rootGraph?.id ?? 'root'
    }

    /**
     * A stack representing subgraph navigation history from the root graph to
     * the current opened subgraph.
     */
    const navigationStack = computed(() =>
      idStack.value
        .map((id) => app.rootGraph.subgraphs.get(id))
        .filter(isNonNullish)
    )

    /**
     * Restore the navigation stack from a list of subgraph IDs.
     * @param subgraphIds The list of subgraph IDs to restore the navigation stack from.
     * @see exportState
     */
    const restoreState = (subgraphIds: string[]) => {
      idStack.value.length = 0
      for (const id of subgraphIds) idStack.value.push(id)
    }

    /**
     * Export the navigation stack as a list of subgraph IDs.
     * @returns The list of subgraph IDs, ending with the currently active subgraph.
     * @see restoreState
     */
    const exportState = () => [...idStack.value]

    /**
     * Get the current viewport state.
     * @returns The current viewport state, or null if the canvas is not available.
     */
    const getCurrentViewport = (): DragAndScaleState | null => {
      const canvas = canvasStore.getCanvas()
      if (!canvas) return null

      return {
        scale: canvas.ds.state.scale,
        offset: [...canvas.ds.state.offset]
      }
    }

    /**
     * Save the current viewport state.
     * @param graphId The graph ID to save for. Use 'root' for root graph, or omit to use current context.
     */
    const saveViewport = (graphId: string) => {
      const viewport = getCurrentViewport()
      if (!viewport) return

      viewportCache.set(graphId, viewport)
    }

    /**
     * Restore viewport state for a graph.
     * @param graphId The graph ID to restore. Use 'root' for root graph, or omit to use current context.
     */
    const restoreViewport = (graphId: string) => {
      const viewport = viewportCache.get(graphId)
      if (!viewport) return

      const canvas = app.canvas
      if (!canvas) return

      canvas.ds.scale = viewport.scale
      canvas.ds.offset[0] = viewport.offset[0]
      canvas.ds.offset[1] = viewport.offset[1]
      canvas.setDirty(true, true)
    }

    /**
     * Update the navigation stack when the active subgraph changes.
     * @param subgraph The new active subgraph.
     * @param prevSubgraph The previous active subgraph.
     */
    const onNavigated = (
      subgraph: Subgraph | undefined,
      prevSubgraph: Subgraph | undefined
    ) => {
      // Save viewport state for the graph we're leaving
      if (prevSubgraph) {
        // Leaving a subgraph
        saveViewport(prevSubgraph.id)
      } else if (!prevSubgraph && subgraph) {
        // Leaving root graph to enter a subgraph
        saveViewport(getCurrentRootGraphId())
      }

      const isInRootGraph = !subgraph
      if (isInRootGraph) {
        idStack.value.length = 0
        restoreViewport(getCurrentRootGraphId())
        return
      }

      const path = findSubgraphPathById(subgraph.rootGraph, subgraph.id)
      const isInReachableSubgraph = !!path
      if (isInReachableSubgraph) {
        idStack.value = [...path]
      } else {
        // Treat as if opening a new subgraph
        idStack.value = [subgraph.id]
      }

      // Always try to restore viewport for the target subgraph
      restoreViewport(subgraph.id)
    }

    // Update navigation stack when opened subgraph changes (also triggers when switching workflows)
    watch(
      () => workflowStore.activeSubgraph,
      (newValue, oldValue) => {
        onNavigated(newValue, oldValue)
      }
    )

    return {
      activeSubgraph,
      navigationStack,
      restoreState,
      exportState,
      saveViewport,
      restoreViewport,
      viewportCache
    }
  }
)
