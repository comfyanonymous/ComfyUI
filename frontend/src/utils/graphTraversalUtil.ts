import type {
  LGraph,
  LGraphNode,
  Subgraph
} from '@/lib/litegraph/src/litegraph'
import type { NodeExecutionId, NodeLocatorId } from '@/types/nodeIdentification'
import {
  createNodeLocatorId,
  parseNodeLocatorId
} from '@/types/nodeIdentification'

import { isSubgraphIoNode } from './typeGuardUtil'

interface NodeWithId {
  id: string | number
  subgraphId?: string | null
}

/**
 * Constructs a locator ID from node data with optional subgraph context.
 *
 * @param nodeData - Node data containing id and optional subgraphId
 * @returns The locator ID string
 */
export function getLocatorIdFromNodeData(nodeData: NodeWithId): string {
  return nodeData.subgraphId
    ? `${nodeData.subgraphId}:${String(nodeData.id)}`
    : String(nodeData.id)
}

/**
 * Parses an execution ID into its component parts.
 *
 * @param executionId - The execution ID (e.g., "123:456:789" or "789")
 * @returns Array of node IDs in the path, or null if invalid
 */
export function parseExecutionId(executionId: string): string[] | null {
  if (!executionId || typeof executionId !== 'string') return null
  return executionId.split(':').filter((part) => part.length > 0)
}

/**
 * Extracts the local node ID from an execution ID.
 *
 * @param executionId - The execution ID (e.g., "123:456:789" or "789")
 * @returns The local node ID or null if invalid
 */
export function getLocalNodeIdFromExecutionId(
  executionId: string
): string | null {
  const parts = parseExecutionId(executionId)
  return parts ? parts[parts.length - 1] : null
}

/**
 * Extracts the subgraph path from an execution ID.
 *
 * @param executionId - The execution ID (e.g., "123:456:789" or "789")
 * @returns Array of subgraph node IDs (excluding the final node ID), or empty array
 */
export function getSubgraphPathFromExecutionId(executionId: string): string[] {
  const parts = parseExecutionId(executionId)
  return parts ? parts.slice(0, -1) : []
}

/**
 * Visits each node in a graph (non-recursive, single level).
 *
 * @param graph - The graph to visit nodes from
 * @param visitor - Function called for each node
 */
export function visitGraphNodes(
  graph: LGraph | Subgraph,
  visitor: (node: LGraphNode) => void
): void {
  for (const node of graph.nodes) {
    visitor(node)
  }
}

/**
 * Traverses a path of subgraphs to reach a target graph.
 *
 * @param startGraph - The graph to start from
 * @param path - Array of subgraph node IDs to traverse
 * @returns The target graph or null if path is invalid
 */
export function traverseSubgraphPath(
  startGraph: LGraph | Subgraph,
  path: string[]
): LGraph | Subgraph | null {
  let currentGraph: LGraph | Subgraph = startGraph

  for (const nodeId of path) {
    const node = currentGraph.getNodeById(nodeId)
    if (!node?.isSubgraphNode?.() || !node.subgraph) return null
    currentGraph = node.subgraph
  }

  return currentGraph
}

/**
 * Traverses all nodes in a graph hierarchy (including subgraphs) and invokes
 * a callback on each node that has the specified property.
 *
 * @param graph - The root graph to start traversal from
 * @param callbackProperty - The name of the callback property to invoke on each node
 */
export function triggerCallbackOnAllNodes(
  graph: LGraph | Subgraph,
  callbackProperty: keyof LGraphNode
): void {
  forEachNode(graph, (node) => {
    const callback = node[callbackProperty]
    if (typeof callback === 'function') {
      callback.call(node)
    }
  })
}

/**
 * Maps a function over all nodes in a graph hierarchy (including subgraphs).
 * This is a pure functional traversal that doesn't mutate the graph.
 *
 * @param graph - The root graph to traverse
 * @param mapFn - Function to apply to each node
 * @returns Array of mapped results (excluding undefined values)
 */
export function mapAllNodes<T>(
  graph: LGraph | Subgraph,
  mapFn: (node: LGraphNode) => T | undefined
): T[] {
  const results: T[] = []

  visitGraphNodes(graph, (node) => {
    // Recursively map over subgraphs first
    if (node.isSubgraphNode?.() && node.subgraph) {
      results.push(...mapAllNodes(node.subgraph, mapFn))
    }

    // Apply map function to current node
    const result = mapFn(node)
    if (result !== undefined) {
      results.push(result)
    }
  })

  return results
}

/**
 * Executes a side-effect function on all nodes in a graph hierarchy.
 * This is for operations that modify nodes or perform side effects.
 *
 * @param graph - The root graph to traverse
 * @param fn - Function to execute on each node
 */
export function forEachNode(
  graph: LGraph | Subgraph,
  fn: (node: LGraphNode) => void
): void {
  visitGraphNodes(graph, (node) => {
    // Recursively process subgraphs first
    if (node.isSubgraphNode?.() && node.subgraph) {
      forEachNode(node.subgraph, fn)
    }

    // Execute function on current node
    fn(node)
  })
}

/**
 * Collects all nodes in a graph hierarchy (including subgraphs) into a flat array.
 *
 * @param graph - The root graph to collect nodes from
 * @param filter - Optional filter function to include only specific nodes
 * @returns Array of all nodes in the graph hierarchy
 */
export function collectAllNodes(
  graph: LGraph | Subgraph,
  filter?: (node: LGraphNode) => boolean
): LGraphNode[] {
  return mapAllNodes(graph, (node) => {
    if (!filter || filter(node)) {
      return node
    }
    return undefined
  })
}

/**
 * Finds a node by ID anywhere in the graph hierarchy.
 *
 * @param graph - The root graph to search
 * @param nodeId - The ID of the node to find
 * @returns The node if found, null otherwise
 */
export function findNodeInHierarchy(
  graph: LGraph | Subgraph,
  nodeId: string | number
): LGraphNode | null {
  // Check current graph
  const node = graph.getNodeById(nodeId)
  if (node) return node

  // Search in subgraphs
  for (const node of graph.nodes) {
    if (node.isSubgraphNode?.() && node.subgraph) {
      const found = findNodeInHierarchy(node.subgraph, nodeId)
      if (found) return found
    }
  }

  return null
}

/**
 * Find a subgraph by its UUID anywhere in the graph hierarchy.
 *
 * @param graph - The root graph to search
 * @param targetUuid - The UUID of the subgraph to find
 * @returns The subgraph if found, null otherwise
 */
export function findSubgraphByUuid(
  graph: LGraph | Subgraph,
  targetUuid: string
): Subgraph | null {
  // Check all nodes in the current graph
  for (const node of graph.nodes) {
    if (node.isSubgraphNode?.() && node.subgraph) {
      if (node.subgraph.id === targetUuid) {
        return node.subgraph
      }
      // Recursively search in nested subgraphs
      const found = findSubgraphByUuid(node.subgraph, targetUuid)
      if (found) return found
    }
  }
  return null
}

/**
 * Iteratively finds the path of subgraph IDs to a target subgraph.
 * @param rootGraph The graph to start searching from.
 * @param targetId The ID of the subgraph to find.
 * @returns An array of subgraph IDs representing the path, or `null` if not found.
 */
export function findSubgraphPathById(
  rootGraph: LGraph,
  targetId: string
): string[] | null {
  const stack: { graph: LGraph | Subgraph; path: string[] }[] = [
    { graph: rootGraph, path: [] }
  ]

  while (stack.length > 0) {
    const { graph, path } = stack.pop()!

    // Check if graph exists and has _nodes property
    if (!graph || !graph._nodes || !Array.isArray(graph._nodes)) {
      continue
    }

    for (const node of graph._nodes) {
      if (node.isSubgraphNode?.() && node.subgraph) {
        const newPath = [...path, String(node.subgraph.id)]
        if (node.subgraph.id === targetId) {
          return newPath
        }
        stack.push({ graph: node.subgraph, path: newPath })
      }
    }
  }

  return null
}

/**
 * Gets the root parent node associated with a hierarchical execution ID.
 * Both Group Nodes and Subgraph Nodes use hierarchical IDs (e.g. "rootId:childId:...").
 * The root parent is always located in the rootGraph.
 *
 * @param rootGraph - The root graph to search from
 * @param executionId - The execution ID (e.g., "123:456")
 * @returns The root parent node if found, null otherwise
 */
export function getRootParentNode(
  rootGraph: LGraph,
  executionId: string
): LGraphNode | null {
  const parts = parseExecutionId(executionId)
  if (!parts || parts.length < 2) return null

  const parentId = parts[0]
  if (!rootGraph) return null

  return rootGraph.getNodeById(Number(parentId)) || null
}

/**
 * Get a node by its execution ID from anywhere in the graph hierarchy.
 * Execution IDs use hierarchical format like "123:456:789" for nested nodes.
 *
 * @param rootGraph - The root graph to search from
 * @param executionId - The execution ID (e.g., "123:456:789" or "789")
 * @returns The node if found, null otherwise
 */
export function getNodeByExecutionId(
  rootGraph: LGraph,
  executionId: string
): LGraphNode | null {
  if (!rootGraph) return null

  const localNodeId = getLocalNodeIdFromExecutionId(executionId)
  if (!localNodeId) return null

  const subgraphPath = getSubgraphPathFromExecutionId(executionId)

  // If no subgraph path, it's in the root graph
  if (subgraphPath.length === 0) {
    return rootGraph.getNodeById(localNodeId) || null
  }

  // Traverse to the target subgraph
  const targetGraph = traverseSubgraphPath(rootGraph, subgraphPath)
  if (!targetGraph) return null

  // Get the node from the target graph
  return targetGraph.getNodeById(localNodeId) || null
}

/**
 * Returns the execution ID for a node relative to the root graph.
 *
 * Root-level nodes return their ID directly (e.g. "42").
 * Nodes inside subgraphs return a colon-separated chain (e.g. "65:70:63").
 *
 * @param rootGraph - The root graph to resolve from
 * @param node - The node whose execution ID to compute
 * @returns The execution ID string, or null if the node has no graph
 */
export function getExecutionIdByNode(
  rootGraph: LGraph,
  node: LGraphNode
): NodeExecutionId | null {
  if (!node.graph) return null

  if (node.graph === rootGraph || node.graph.isRootGraph) {
    return String(node.id)
  }

  const parentPath = findPartialExecutionPathToGraph(
    node.graph as LGraph,
    rootGraph
  )
  if (parentPath === undefined) return null

  return `${parentPath}:${node.id}`
}

/**
 * Get a node by its locator ID from anywhere in the graph hierarchy.
 * Locator IDs use UUID format like "uuid:nodeId" for subgraph nodes.
 *
 * @param rootGraph - The root graph to search from
 * @param locatorId - The locator ID (e.g., "uuid:123" or "123")
 * @returns The node if found, null otherwise
 */
export function getNodeByLocatorId(
  rootGraph: LGraph,
  locatorId: NodeLocatorId | string
): LGraphNode | null {
  if (!rootGraph) return null

  const parsedIds = parseNodeLocatorId(locatorId)
  if (!parsedIds) return null

  const { subgraphUuid, localNodeId } = parsedIds

  // If no subgraph UUID, it's in the root graph
  if (!subgraphUuid) {
    return rootGraph.getNodeById(localNodeId) || null
  }

  // Find the subgraph with the matching UUID
  const targetSubgraph = findSubgraphByUuid(rootGraph, subgraphUuid)
  if (!targetSubgraph) return null

  return targetSubgraph.getNodeById(localNodeId) || null
}

/**
 * Convert execution context node IDs to NodeLocatorIds.
 * Uses traverseSubgraphPath to resolve the subgraph chain.
 *
 * @param rootGraph - The root graph to resolve against
 * @param nodeId - The node ID from execution context (could be execution ID like "123:456:789")
 * @returns The NodeLocatorId, or undefined if resolution fails
 */
export function executionIdToNodeLocatorId(
  rootGraph: LGraph,
  nodeId: string | number
): NodeLocatorId | undefined {
  const nodeIdStr = String(nodeId)

  if (!nodeIdStr.includes(':')) {
    // It's a top-level node ID
    return nodeIdStr
  }

  // It's an execution node ID — resolve subgraph path
  const parts = nodeIdStr.split(':')
  const localNodeId = parts.at(-1)!
  const subgraphPath = parts.slice(0, -1)

  const targetGraph = traverseSubgraphPath(rootGraph, subgraphPath)
  if (!targetGraph) return undefined

  return createNodeLocatorId(targetGraph.id, localNodeId)
}

/**
 * Finds the root graph from any graph in the hierarchy.
 *
 * @param graph - Any graph or subgraph in the hierarchy
 * @returns The root graph
 */
export function getRootGraph(graph: LGraph | Subgraph): LGraph | Subgraph {
  let current: LGraph | Subgraph = graph
  while ('rootGraph' in current && current.rootGraph) {
    current = current.rootGraph
  }
  return current
}

/**
 * Applies a function to all nodes whose type matches a subgraph ID.
 * Operates on the entire graph hierarchy starting from the root.
 *
 * @param rootGraph - The root graph to search in
 * @param subgraphId - The ID/type of the subgraph to match nodes against
 * @param fn - Function to apply to each matching node
 */
export function forEachSubgraphNode(
  rootGraph: LGraph | Subgraph | null | undefined,
  subgraphId: string | null | undefined,
  fn: (node: LGraphNode) => void
): void {
  if (!rootGraph || !subgraphId) return

  forEachNode(rootGraph, (node) => {
    if (node.type === subgraphId) {
      fn(node)
    }
  })
}

/**
 * Maps a function over all nodes whose type matches a subgraph ID.
 * Operates on the entire graph hierarchy starting from the root.
 *
 * @param rootGraph - The root graph to search in
 * @param subgraphId - The ID/type of the subgraph to match nodes against
 * @param mapFn - Function to apply to each matching node
 * @returns Array of mapped results
 */
export function mapSubgraphNodes<T>(
  rootGraph: LGraph | Subgraph | null | undefined,
  subgraphId: string | null | undefined,
  mapFn: (node: LGraphNode) => T
): T[] {
  if (!rootGraph || !subgraphId) return []

  return mapAllNodes(rootGraph, (node) => {
    if (node.type === subgraphId) {
      return mapFn(node)
    }
    return undefined
  })
}

/**
 * Gets all non-IO nodes from a subgraph (excludes SubgraphInputNode and SubgraphOutputNode).
 * These are the user-created nodes that can be safely removed when clearing a subgraph.
 *
 * @param subgraph - The subgraph to get non-IO nodes from
 * @returns Array of non-IO nodes (user-created nodes)
 */
export function getAllNonIoNodesInSubgraph(subgraph: Subgraph): LGraphNode[] {
  return subgraph.nodes.filter((node) => !isSubgraphIoNode(node))
}

/**
 * Options for traverseNodesDepthFirst function
 */
interface TraverseNodesOptions<T> {
  /** Function called for each node during traversal */
  visitor?: (node: LGraphNode, context: T) => T
  /** Initial context value */
  initialContext?: T
  /** Whether to traverse into subgraph nodes (default: true) */
  expandSubgraphs?: boolean
}

/**
 * Performs depth-first traversal of nodes and their subgraphs.
 * Generic visitor pattern that can be used for various node processing tasks.
 *
 * @param nodes - Starting nodes for traversal
 * @param options - Optional traversal configuration
 */
export function traverseNodesDepthFirst<T = void>(
  nodes: LGraphNode[],
  options?: TraverseNodesOptions<T>
): void {
  const {
    visitor = () => undefined as T,
    initialContext = undefined as T,
    expandSubgraphs = true
  } = options || {}
  type StackItem = { node: LGraphNode; context: T }
  const stack: StackItem[] = []

  // Initialize stack with starting nodes
  for (const node of nodes) {
    stack.push({ node, context: initialContext })
  }

  // Process stack iteratively (DFS)
  while (stack.length > 0) {
    const { node, context } = stack.pop()!

    // Visit node and get updated context for children
    const childContext = visitor(node, context)

    // If it's a subgraph and we should expand, add children to stack
    if (expandSubgraphs && node.isSubgraphNode?.() && node.subgraph) {
      // Process children in reverse order to maintain left-to-right DFS processing
      // when popping from stack (LIFO). Iterate backwards to avoid array reversal.
      const children = node.subgraph.nodes
      for (let i = children.length - 1; i >= 0; i--) {
        stack.push({ node: children[i], context: childContext })
      }
    }
  }
}

/**
 * Reduces all nodes in a graph hierarchy to a single value using a reducer function.
 * Single-pass traversal for efficient aggregation.
 *
 * @param graph - The root graph to traverse
 * @param reducer - Function that reduces each node into the accumulator
 * @param initialValue - The initial accumulator value
 * @returns The final reduced value
 */
export function reduceAllNodes<T>(
  graph: LGraph | Subgraph,
  reducer: (accumulator: T, node: LGraphNode) => T,
  initialValue: T
): T {
  let result = initialValue
  forEachNode(graph, (node) => {
    result = reducer(result, node)
  })
  return result
}

/**
 * Options for collectFromNodes function
 */
interface CollectFromNodesOptions<T, C> {
  /** Function that returns data to collect for each node */
  collector?: (node: LGraphNode, context: C) => T | null
  /** Function that builds context for child nodes */
  contextBuilder?: (node: LGraphNode, parentContext: C) => C
  /** Initial context value */
  initialContext?: C
  /** Whether to traverse into subgraph nodes (default: true) */
  expandSubgraphs?: boolean
}

/**
 * Collects nodes with custom data during depth-first traversal.
 * Generic collector that can gather any type of data per node.
 *
 * @param nodes - Starting nodes for traversal
 * @param options - Optional collection configuration
 * @returns Array of collected data
 */
export function collectFromNodes<T = LGraphNode, C = void>(
  nodes: LGraphNode[],
  options?: CollectFromNodesOptions<T, C>
): T[] {
  const {
    collector = (node: LGraphNode) => node as unknown as T,
    contextBuilder = () => undefined as C,
    initialContext = undefined as C,
    expandSubgraphs = true
  } = options || {}
  const results: T[] = []

  traverseNodesDepthFirst(nodes, {
    visitor: (node, context) => {
      const data = collector(node, context)
      if (data !== null) {
        results.push(data)
      }
      return contextBuilder(node, context)
    },
    initialContext,
    expandSubgraphs
  })

  return results
}

/**
 * Collects execution IDs for selected nodes and all their descendants.
 * Uses the generic DFS traversal with optimized string building.
 *
 * @param selectedNodes - The selected nodes to process
 * @returns Array of execution IDs for selected nodes and all nodes within selected subgraphs
 */
export function getExecutionIdsForSelectedNodes(
  selectedNodes: LGraphNode[],
  startGraph = selectedNodes[0]?.graph
): NodeExecutionId[] {
  if (!startGraph) return []
  const rootGraph = startGraph.rootGraph
  const parentPath = startGraph.isRootGraph
    ? ''
    : findPartialExecutionPathToGraph(startGraph, rootGraph)
  if (parentPath === undefined) return []

  return collectFromNodes<NodeExecutionId, string>(selectedNodes, {
    collector: (node, parentExecutionId) => {
      const nodeId = String(node.id)
      return parentExecutionId ? `${parentExecutionId}:${nodeId}` : nodeId
    },
    contextBuilder: (node, parentExecutionId) => {
      const nodeId = String(node.id)
      return parentExecutionId ? `${parentExecutionId}:${nodeId}` : nodeId
    },
    initialContext: parentPath,
    expandSubgraphs: true
  })
}

function findPartialExecutionPathToGraph(
  target: LGraph,
  root: LGraph
): string | undefined {
  for (const node of root.nodes) {
    if (!node.isSubgraphNode()) continue

    if (node.subgraph === target) return `${node.id}`

    const subpath = findPartialExecutionPathToGraph(target, node.subgraph)
    if (subpath !== undefined) return node.id + ':' + subpath
  }
  return undefined
}
