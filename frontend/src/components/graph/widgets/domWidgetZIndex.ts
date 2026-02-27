import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'

export function getDomWidgetZIndex(
  node: LGraphNode,
  currentGraph: LGraphNode['graph'] | undefined
): number {
  if (!currentGraph) return node.order ?? -1

  const graphOrder = currentGraph.nodes.indexOf(node)
  if (graphOrder === -1) return node.order ?? -1

  return graphOrder
}
