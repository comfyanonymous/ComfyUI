import type { LGraphNode } from '@/lib/litegraph/src/litegraph'

/**
 * Checks if a node is an output node.
 * Output nodes are nodes that have the output_node flag set in their nodeData.
 *
 * @param node - The node to check
 * @returns True if the node is an output node, false otherwise
 */
export const isOutputNode = (node: LGraphNode) =>
  node.constructor.nodeData?.output_node

/**
 * Filters nodes to find only output nodes.
 * Output nodes are nodes that have the output_node flag set in their nodeData.
 *
 * @param nodes - Array of nodes to filter
 * @returns Array of output nodes only
 */
export const filterOutputNodes = (nodes: LGraphNode[]): LGraphNode[] =>
  nodes.filter(isOutputNode)
