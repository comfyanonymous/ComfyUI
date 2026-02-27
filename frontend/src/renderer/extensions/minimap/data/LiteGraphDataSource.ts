import type { MinimapNodeData } from '../types'
import { AbstractMinimapDataSource } from './AbstractMinimapDataSource'

/**
 * LiteGraph data source implementation
 */
export class LiteGraphDataSource extends AbstractMinimapDataSource {
  getNodes(): MinimapNodeData[] {
    if (!this.graph?._nodes) return []

    return this.graph._nodes.map((node) => ({
      id: String(node.id),
      x: node.pos[0],
      y: node.pos[1],
      width: node.size[0],
      height: node.size[1],
      bgcolor: node.bgcolor,
      mode: node.mode,
      hasErrors: node.has_errors
    }))
  }

  getNodeCount(): number {
    return this.graph?._nodes?.length ?? 0
  }

  hasData(): boolean {
    return this.getNodeCount() > 0
  }
}
