/**
 * Node Z-Index Management Composable
 *
 * Provides focused functionality for managing node layering through z-index.
 * Integrates with the layout system to ensure proper visual ordering.
 */
import type { NodeId } from '@/platform/workflow/validation/schemas/workflowSchema'
import { useLayoutMutations } from '@/renderer/core/layout/operations/layoutMutations'
import { LayoutSource } from '@/renderer/core/layout/types'

interface NodeZIndexOptions {
  /**
   * Layout source for z-index mutations
   * @default LayoutSource.Vue
   */
  layoutSource?: LayoutSource
}

export function useNodeZIndex(options: NodeZIndexOptions = {}) {
  const { layoutSource = LayoutSource.Vue } = options
  const layoutMutations = useLayoutMutations()

  /**
   * Bring node to front (highest z-index)
   * @param nodeId - The node to bring to front
   * @param source - Optional source override
   */
  function bringNodeToFront(nodeId: NodeId, source?: LayoutSource) {
    layoutMutations.setSource(source ?? layoutSource)
    layoutMutations.bringNodeToFront(nodeId)
  }

  return {
    bringNodeToFront
  }
}
