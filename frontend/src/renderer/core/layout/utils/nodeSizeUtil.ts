import { LiteGraph } from '@/lib/litegraph/src/litegraph'

export const removeNodeTitleHeight = (height: number) =>
  Math.max(0, height - (LiteGraph.NODE_TITLE_HEIGHT || 0))
