import type { SubgraphNode } from '@/lib/litegraph/src/subgraph/SubgraphNode'
import { usePromotionStore } from '@/stores/promotionStore'

export function hasUnpromotedWidgets(subgraphNode: SubgraphNode): boolean {
  const promotionStore = usePromotionStore()
  const { id: subgraphNodeId, rootGraph, subgraph } = subgraphNode

  return subgraph.nodes.some((interiorNode) =>
    (interiorNode.widgets ?? []).some(
      (widget) =>
        !widget.computedDisabled &&
        !promotionStore.isPromoted(
          rootGraph.id,
          subgraphNodeId,
          String(interiorNode.id),
          widget.name
        )
    )
  )
}
