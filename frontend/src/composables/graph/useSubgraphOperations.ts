import { useSelectedLiteGraphItems } from '@/composables/canvas/useSelectedLiteGraphItems'
import { SubgraphNode } from '@/lib/litegraph/src/litegraph'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useNodeOutputStore } from '@/stores/imagePreviewStore'
import { useSubgraphStore } from '@/stores/subgraphStore'

/**
 * Composable for handling subgraph-related operations
 */
export function useSubgraphOperations() {
  const { getSelectedNodes } = useSelectedLiteGraphItems()
  const canvasStore = useCanvasStore()
  const workflowStore = useWorkflowStore()
  const nodeOutputStore = useNodeOutputStore()
  const subgraphStore = useSubgraphStore()

  const convertToSubgraph = () => {
    const canvas = canvasStore.getCanvas()
    const graph = canvas.subgraph ?? canvas.graph
    if (!graph) {
      return null
    }

    const res = graph.convertToSubgraph(canvas.selectedItems)
    if (!res) {
      return
    }

    const { node } = res
    canvas.select(node)
    canvasStore.updateSelectedItems()
    // Trigger change tracking
    workflowStore.activeWorkflow?.changeTracker?.checkState()
  }

  const doUnpack = (
    subgraphNodes: SubgraphNode[],
    skipMissingNodes: boolean
  ) => {
    const canvas = canvasStore.getCanvas()
    const graph = canvas.subgraph ?? canvas.graph
    if (!graph) return

    for (const subgraphNode of subgraphNodes) {
      nodeOutputStore.revokeSubgraphPreviews(subgraphNode)
      graph.unpackSubgraph(subgraphNode, { skipMissingNodes })
    }
    workflowStore.activeWorkflow?.changeTracker?.checkState()
  }

  const unpackSubgraph = () => {
    const canvas = canvasStore.getCanvas()
    const graph = canvas.subgraph ?? canvas.graph

    if (!graph) {
      return
    }

    const selectedItems = Array.from(canvas.selectedItems)
    const subgraphNodes = selectedItems.filter(
      (item): item is SubgraphNode => item instanceof SubgraphNode
    )

    if (subgraphNodes.length === 0) {
      return
    }
    doUnpack(subgraphNodes, true)
  }

  const addSubgraphToLibrary = async () => {
    const selectedItems = Array.from(canvasStore.selectedItems)
    const subgraphNodes = selectedItems.filter(
      (item): item is SubgraphNode => item instanceof SubgraphNode
    )
    if (subgraphNodes.length !== 1) {
      return
    }
    await subgraphStore.publishSubgraph()
  }

  const isSubgraphSelected = (): boolean => {
    const selectedItems = Array.from(canvasStore.selectedItems)
    return selectedItems.some((item) => item instanceof SubgraphNode)
  }

  const hasSelectableNodes = (): boolean => {
    return getSelectedNodes().length > 0
  }

  return {
    convertToSubgraph,
    unpackSubgraph,
    addSubgraphToLibrary,
    isSubgraphSelected,
    hasSelectableNodes
  }
}
