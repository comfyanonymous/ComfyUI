import type { NodeId } from '@/platform/workflow/validation/schemas/workflowSchema'

/**
 * A globally unique identifier for nodes that maintains consistency across
 * multiple instances of the same subgraph.
 *
 * Format:
 * - For subgraph nodes: `<immediate-contained-subgraph-uuid>:<local-node-id>`
 * - For root graph nodes: `<local-node-id>`
 *
 * Examples:
 * - "a1b2c3d4-e5f6-7890-abcd-ef1234567890:123" (node in subgraph)
 * - "456" (node in root graph)
 *
 * Unlike execution IDs which change based on the instance path,
 * NodeLocatorId remains the same for all instances of a particular node.
 */
export type NodeLocatorId = string

/**
 * An execution identifier representing a node's position in nested subgraphs.
 * Also known as ExecutionId in some contexts.
 *
 * Format: Colon-separated path of node IDs
 * Example: "123:456:789" (node 789 in subgraph 456 in subgraph 123)
 */
export type NodeExecutionId = string

/**
 * Type guard to check if a value is a NodeLocatorId
 */
export function isNodeLocatorId(value: unknown): value is NodeLocatorId {
  if (typeof value !== 'string') return false

  // Check if it's a simple node ID (root graph node)
  const parts = value.split(':')
  if (parts.length === 1) {
    // Simple node ID - must be non-empty
    return value.length > 0
  }

  // Check for UUID:nodeId format
  if (parts.length !== 2) return false

  // Check that node ID part is not empty
  if (!parts[1]) return false

  // Basic UUID format check (8-4-4-4-12 hex characters)
  const uuidPattern =
    /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
  return uuidPattern.test(parts[0])
}

/**
 * Type guard to check if a value is a NodeExecutionId
 */
export function isNodeExecutionId(value: unknown): value is NodeExecutionId {
  if (typeof value !== 'string') return false
  // Must contain at least one colon to be an execution ID
  return value.includes(':')
}

/**
 * Parse a NodeLocatorId into its components
 * @param id The NodeLocatorId to parse
 * @returns The subgraph UUID and local node ID, or null if invalid
 */
export function parseNodeLocatorId(
  id: string
): { subgraphUuid: string | null; localNodeId: NodeId } | null {
  if (!isNodeLocatorId(id)) return null

  const parts = id.split(':')

  if (parts.length === 1) {
    // Simple node ID (root graph)
    return {
      subgraphUuid: null,
      localNodeId: isNaN(Number(id)) ? id : Number(id)
    }
  }

  const [subgraphUuid, localNodeId] = parts
  return {
    subgraphUuid,
    localNodeId: isNaN(Number(localNodeId)) ? localNodeId : Number(localNodeId)
  }
}

/**
 * Create a NodeLocatorId from components
 * @param subgraphUuid The UUID of the immediate containing subgraph
 * @param localNodeId The local node ID within that subgraph
 * @returns A properly formatted NodeLocatorId
 */
export function createNodeLocatorId(
  subgraphUuid: string,
  localNodeId: NodeId
): NodeLocatorId {
  return `${subgraphUuid}:${localNodeId}`
}

/**
 * Parse a NodeExecutionId into its component node IDs
 * @param id The NodeExecutionId to parse
 * @returns Array of node IDs from root to target, or null if not an execution ID
 */
export function parseNodeExecutionId(id: string): NodeId[] | null {
  if (!isNodeExecutionId(id)) return null

  return id
    .split(':')
    .map((part) => (isNaN(Number(part)) ? part : Number(part)))
}

/**
 * Create a NodeExecutionId from an array of node IDs
 * @param nodeIds Array of node IDs from root to target
 * @returns A properly formatted NodeExecutionId
 */
export function createNodeExecutionId(nodeIds: NodeId[]): NodeExecutionId {
  return nodeIds.join(':')
}

/**
 * Returns all ancestor execution IDs for a given execution ID, including itself.
 *
 * Example: "65:70:63" → ["65", "65:70", "65:70:63"]
 */
export function getAncestorExecutionIds(
  executionId: string | NodeExecutionId
): NodeExecutionId[] {
  const parts = executionId.split(':')
  return Array.from({ length: parts.length }, (_, i) =>
    parts.slice(0, i + 1).join(':')
  )
}

/**
 * Returns all ancestor execution IDs for a given execution ID, excluding itself.
 *
 * Example: "65:70:63" → ["65", "65:70"]
 */
export function getParentExecutionIds(
  executionId: string | NodeExecutionId
): NodeExecutionId[] {
  return getAncestorExecutionIds(executionId).slice(0, -1)
}
