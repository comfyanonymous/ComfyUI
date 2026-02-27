import { layoutStore } from '@/renderer/core/layout/store/layoutStore'

import type { MinimapNodeData } from '../types'
import { AbstractMinimapDataSource } from './AbstractMinimapDataSource'

/**
 * Layout Store data source implementation
 */
export class LayoutStoreDataSource extends AbstractMinimapDataSource {
  getNodes(): MinimapNodeData[] {
    const allNodes = layoutStore.getAllNodes().value
    if (allNodes.size === 0) return []

    const nodes: MinimapNodeData[] = []

    for (const [nodeId, layout] of allNodes) {
      // Find corresponding LiteGraph node for additional properties
      const graphNode = this.graph?._nodes?.find((n) => String(n.id) === nodeId)

      nodes.push({
        id: nodeId,
        x: layout.position.x,
        y: layout.position.y,
        width: layout.size.width,
        height: layout.size.height,
        bgcolor: graphNode?.bgcolor,
        mode: graphNode?.mode,
        hasErrors: graphNode?.has_errors
      })
    }

    return nodes
  }

  getNodeCount(): number {
    return layoutStore.getAllNodes().value.size
  }

  hasData(): boolean {
    return this.getNodeCount() > 0
  }
}
