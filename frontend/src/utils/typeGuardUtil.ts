import type {
  INodeSlot,
  LGraph,
  LGraphNode,
  Subgraph
} from '@/lib/litegraph/src/litegraph'
import type { ResultItemType } from '@/schemas/apiSchema'

/**
 * Check if an error is an AbortError triggered by `AbortController#abort`
 * when cancelling a request.
 */
export const isAbortError = (
  err: unknown
): err is DOMException & { name: 'AbortError' } =>
  err instanceof DOMException && err.name === 'AbortError'

export const isSubgraph = (
  item: LGraph | Subgraph | undefined | null
): item is Subgraph => item?.isRootGraph === false

/**
 * Check if an item is non-nullish.
 */
export const isNonNullish = <T>(item: T | undefined | null): item is T =>
  item != null

/**
 * Type guard to check if a node is a subgraph input/output node.
 * These nodes are essential to subgraph structure and should not be removed.
 */
export const isSubgraphIoNode = (
  node: LGraphNode
): node is LGraphNode & {
  constructor: { comfyClass: 'SubgraphInputNode' | 'SubgraphOutputNode' }
} => {
  const nodeClass = node.constructor?.comfyClass
  return nodeClass === 'SubgraphInputNode' || nodeClass === 'SubgraphOutputNode'
}

/**
 * Type guard for slot objects (inputs/outputs)
 */
export const isSlotObject = (obj: unknown): obj is INodeSlot => {
  return (
    obj !== null &&
    typeof obj === 'object' &&
    'name' in obj &&
    'type' in obj &&
    'boundingRect' in obj
  )
}

/**
 * Type guard to check if a string is a valid ResultItemType
 * ResultItemType is used for asset categorization (input/output/temp)
 */
export const isResultItemType = (
  value: string | undefined
): value is ResultItemType => {
  return value === 'input' || value === 'output' || value === 'temp'
}
