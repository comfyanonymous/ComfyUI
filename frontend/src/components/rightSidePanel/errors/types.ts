export interface ErrorItem {
  message: string
  details?: string
  isRuntimeError?: boolean
}

export interface ErrorCardData {
  id: string
  title: string
  nodeId?: string
  nodeTitle?: string
  graphNodeId?: string
  isSubgraphNode?: boolean
  errors: ErrorItem[]
}

export type ErrorGroup =
  | {
      type: 'execution'
      title: string
      cards: ErrorCardData[]
      priority: number
    }
  | { type: 'missing_node'; title: string; priority: number }
  | { type: 'swap_nodes'; title: string; priority: number }
