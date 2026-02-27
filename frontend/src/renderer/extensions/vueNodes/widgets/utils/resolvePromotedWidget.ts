import { isPromotedWidgetView } from '@/core/graph/subgraph/promotedWidgetTypes'
import { resolvePromotedWidgetSource } from '@/core/graph/subgraph/resolvePromotedWidgetSource'
import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'

type ResolvedWidget = {
  node: LGraphNode
  widget: IBaseWidget
}

export function resolveWidgetFromHostNode(
  hostNode: LGraphNode | undefined,
  widgetName: string
): ResolvedWidget | undefined {
  if (!hostNode) return undefined

  const widget = hostNode.widgets?.find((entry) => entry.name === widgetName)
  if (!widget) return undefined

  const sourceWidget = resolvePromotedWidgetSource(hostNode, widget)
  if (sourceWidget) return sourceWidget

  if (isPromotedWidgetView(widget) && hostNode.isSubgraphNode())
    return undefined

  return {
    node: hostNode,
    widget
  }
}
