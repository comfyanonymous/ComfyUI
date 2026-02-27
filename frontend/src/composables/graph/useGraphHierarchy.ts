import type { LGraphGroup, LGraphNode } from '@/lib/litegraph/src/litegraph'
import { containsCentre, containsRect } from '@/lib/litegraph/src/measure'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'

/**
 * Composable for working with graph hierarchy, specifically group containment.
 */
export function useGraphHierarchy() {
  const canvasStore = useCanvasStore()

  /**
   * Finds the smallest group that contains the center of a node's bounding box.
   * When multiple groups contain the node, returns the one with the smallest area.
   *
   * TODO: This traverses the entire graph and could be very slow; needs optimization.
   * Consider spatial indexing or caching for large graphs.
   *
   * @param node - The node to find the parent group for
   * @returns The parent group if found, otherwise null
   */
  function findParentGroup(node: LGraphNode): LGraphGroup | null {
    const graphGroups = (canvasStore.canvas?.graph?.groups ??
      []) as LGraphGroup[]

    let parent: LGraphGroup | null = null

    for (const group of graphGroups) {
      const groupRect = group.boundingRect
      if (!containsCentre(groupRect, node.boundingRect)) continue

      if (!parent) {
        parent = group
        continue
      }

      const parentRect = parent.boundingRect
      const candidateInsideParent = containsRect(parentRect, groupRect)
      const parentInsideCandidate = containsRect(groupRect, parentRect)

      if (candidateInsideParent && !parentInsideCandidate) {
        parent = group
        continue
      }

      const candidateArea = groupRect[2] * groupRect[3]
      const parentArea = parentRect[2] * parentRect[3]

      if (candidateArea < parentArea) parent = group
    }

    return parent
  }

  return {
    findParentGroup
  }
}
