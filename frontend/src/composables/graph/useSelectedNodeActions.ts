import { useSelectedLiteGraphItems } from '@/composables/canvas/useSelectedLiteGraphItems'
import { LGraphEventMode } from '@/lib/litegraph/src/litegraph'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { app } from '@/scripts/app'
import { useCommandStore } from '@/stores/commandStore'
import { filterOutputNodes } from '@/utils/nodeFilterUtil'

/**
 * Composable for handling node information and utility operations
 */
export function useSelectedNodeActions() {
  const { getSelectedNodes, toggleSelectedNodesMode } =
    useSelectedLiteGraphItems()
  const commandStore = useCommandStore()
  const workflowStore = useWorkflowStore()

  const adjustNodeSize = () => {
    const selectedNodes = getSelectedNodes()

    selectedNodes.forEach((node) => {
      const optimalSize = node.computeSize()
      node.setSize([optimalSize[0], optimalSize[1]])
    })

    app.canvas.setDirty(true, true)
    workflowStore.activeWorkflow?.changeTracker?.checkState()
  }

  const toggleNodeCollapse = () => {
    const selectedNodes = getSelectedNodes()
    selectedNodes.forEach((node) => {
      node.collapse()
    })

    app.canvas.setDirty(true, true)
    workflowStore.activeWorkflow?.changeTracker?.checkState()
  }

  const toggleNodePin = () => {
    const selectedNodes = getSelectedNodes()
    selectedNodes.forEach((node) => {
      node.pin(!node.pinned)
    })

    app.canvas.setDirty(true, true)
    workflowStore.activeWorkflow?.changeTracker?.checkState()
  }

  const toggleNodeBypass = () => {
    toggleSelectedNodesMode(LGraphEventMode.BYPASS)
    app.canvas.setDirty(true, true)
  }

  const runBranch = async () => {
    const selectedNodes = getSelectedNodes()
    const selectedOutputNodes = filterOutputNodes(selectedNodes)
    if (selectedOutputNodes.length === 0) return
    await commandStore.execute('Comfy.QueueSelectedOutputNodes')
  }

  return {
    adjustNodeSize,
    toggleNodeCollapse,
    toggleNodePin,
    toggleNodeBypass,
    runBranch
  }
}
