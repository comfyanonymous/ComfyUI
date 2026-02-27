import type { LGraphNode } from '@/lib/litegraph/src/litegraph'

/**
 * Extracts the custom node registry ID (cnr_id or aux_id) from a raw
 * properties bag.
 *
 * @param properties - The properties object to inspect
 * @returns The cnrId string, or undefined if not found
 */
export function getCnrIdFromProperties(
  properties: Record<string, unknown> | undefined | null
): string | undefined {
  if (typeof properties?.cnr_id === 'string') return properties.cnr_id
  if (typeof properties?.aux_id === 'string') return properties.aux_id
  return undefined
}

/**
 * Extracts the custom node registry ID (cnr_id or aux_id) from a node's properties.
 * Returns undefined if neither property is present.
 *
 * @param node - The graph node to inspect
 * @returns The cnrId string, or undefined if not found
 */
export function getCnrIdFromNode(node: LGraphNode): string | undefined {
  return getCnrIdFromProperties(node.properties as Record<string, unknown>)
}
