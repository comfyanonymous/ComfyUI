import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'
import type { SubgraphNode } from '@/lib/litegraph/src/subgraph/SubgraphNode'

export interface PromotedWidgetView extends IBaseWidget {
  readonly node: SubgraphNode
  readonly sourceNodeId: string
  readonly sourceWidgetName: string
}

export function isPromotedWidgetView(
  widget: IBaseWidget
): widget is PromotedWidgetView {
  return 'sourceNodeId' in widget && 'sourceWidgetName' in widget
}
