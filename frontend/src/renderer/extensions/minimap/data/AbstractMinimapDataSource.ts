import type { LGraph } from '@/lib/litegraph/src/litegraph'
import { calculateNodeBounds } from '@/renderer/core/spatial/boundsCalculator'
import type { PositionedNode } from '@/renderer/core/spatial/boundsCalculator'

import type {
  IMinimapDataSource,
  MinimapBounds,
  MinimapGroupData,
  MinimapLinkData,
  MinimapNodeData
} from '../types'

/**
 * Abstract base class for minimap data sources
 * Provides common functionality and shared implementation
 */
export abstract class AbstractMinimapDataSource implements IMinimapDataSource {
  constructor(protected graph: LGraph | null) {}

  // Abstract methods that must be implemented by subclasses
  abstract getNodes(): MinimapNodeData[]
  abstract getNodeCount(): number
  abstract hasData(): boolean

  // Shared implementation using calculateNodeBounds
  getBounds(): MinimapBounds {
    const nodes = this.getNodes()
    if (nodes.length === 0) {
      return { minX: 0, minY: 0, maxX: 100, maxY: 100, width: 100, height: 100 }
    }

    // Convert MinimapNodeData to the format expected by calculateNodeBounds
    const compatibleNodes = nodes.map(
      (node): PositionedNode => ({
        pos: [node.x, node.y],
        size: [node.width, node.height]
      })
    )

    const bounds = calculateNodeBounds(compatibleNodes)
    if (!bounds) {
      return { minX: 0, minY: 0, maxX: 100, maxY: 100, width: 100, height: 100 }
    }

    return bounds
  }

  // Shared implementation for groups
  getGroups(): MinimapGroupData[] {
    if (!this.graph?._groups) return []
    return this.graph._groups.map((group) => ({
      x: group.pos[0],
      y: group.pos[1],
      width: group.size[0],
      height: group.size[1],
      color: group.color
    }))
  }

  // TODO: update when Layoutstore supports links
  getLinks(): MinimapLinkData[] {
    if (!this.graph) return []
    return this.extractLinksFromGraph(this.graph)
  }

  protected extractLinksFromGraph(graph: LGraph): MinimapLinkData[] {
    const links: MinimapLinkData[] = []
    const nodeMap = new Map(this.getNodes().map((n) => [n.id, n]))

    for (const node of graph._nodes) {
      if (!node.outputs) continue

      const sourceNodeData = nodeMap.get(String(node.id))
      if (!sourceNodeData) continue

      for (const output of node.outputs) {
        if (!output.links) continue

        for (const linkId of output.links) {
          const link = graph.links[linkId]
          if (!link) continue

          const targetNodeData = nodeMap.get(String(link.target_id))
          if (!targetNodeData) continue

          links.push({
            sourceNode: sourceNodeData,
            targetNode: targetNodeData,
            sourceSlot: link.origin_slot,
            targetSlot: link.target_slot
          })
        }
      }
    }

    return links
  }
}
