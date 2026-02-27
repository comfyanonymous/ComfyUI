import type { PrimitiveNode } from '@/extensions/core/widgetInputs'
import type { LGraphNode } from '@/lib/litegraph/src/litegraph'

export const isPrimitiveNode = (
  node: LGraphNode
): node is PrimitiveNode & LGraphNode => node.type === 'PrimitiveNode'
