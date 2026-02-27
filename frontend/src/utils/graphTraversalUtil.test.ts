// oxlint-disable no-misused-spread
import { describe, expect, it, vi } from 'vitest'

import type {
  LGraph,
  LGraphNode,
  Subgraph
} from '@/lib/litegraph/src/litegraph'
import {
  collectAllNodes,
  collectFromNodes,
  findNodeInHierarchy,
  findSubgraphByUuid,
  forEachNode,
  forEachSubgraphNode,
  getAllNonIoNodesInSubgraph,
  getExecutionIdsForSelectedNodes,
  getLocalNodeIdFromExecutionId,
  getNodeByExecutionId,
  getNodeByLocatorId,
  getRootGraph,
  getSubgraphPathFromExecutionId,
  mapAllNodes,
  mapSubgraphNodes,
  parseExecutionId,
  traverseNodesDepthFirst,
  traverseSubgraphPath,
  triggerCallbackOnAllNodes,
  visitGraphNodes
} from '@/utils/graphTraversalUtil'
import { createMockLGraphNode } from './__tests__/litegraphTestUtils'

// Mock node factory
function createMockNode(
  id: string | number,
  options: {
    isSubgraph?: boolean
    subgraph?: Subgraph
    callback?: () => void
    graph?: LGraph
  } = {}
): LGraphNode {
  const node = createMockLGraphNode({
    id,
    isSubgraphNode: options.isSubgraph ? () => true : undefined,
    subgraph: options.subgraph,
    onExecutionStart: options.callback,
    graph: options.graph
  }) satisfies Partial<LGraphNode> as LGraphNode
  options.graph?.nodes?.push(node)
  return node
}

// Mock graph factory
function createMockGraph(nodes: LGraphNode[]): LGraph {
  return {
    _nodes: nodes,
    nodes: nodes,
    isRootGraph: true,
    getNodeById: (id: string | number) =>
      nodes.find((n) => String(n.id) === String(id)) || null
  } satisfies Partial<LGraph> as LGraph
}

// Mock subgraph factory
function createMockSubgraph(
  id: string,
  nodes: LGraphNode[],
  rootGraph?: LGraph
): Subgraph {
  const graph = {
    id,
    _nodes: nodes,
    nodes: nodes,
    isRootGraph: false,
    rootGraph,
    getNodeById: (nodeId: string | number) =>
      nodes.find((n) => String(n.id) === String(nodeId)) || null
  } satisfies Partial<Subgraph> as Subgraph
  return graph
}

describe('graphTraversalUtil', () => {
  describe('Pure utility functions', () => {
    describe('parseExecutionId', () => {
      it('should parse simple execution ID', () => {
        expect(parseExecutionId('123')).toEqual(['123'])
      })

      it('should parse complex execution ID', () => {
        expect(parseExecutionId('123:456:789')).toEqual(['123', '456', '789'])
      })

      it('should handle empty parts', () => {
        expect(parseExecutionId('123::789')).toEqual(['123', '789'])
      })

      it('should return null for invalid input', () => {
        expect(parseExecutionId('')).toBeNull()
        expect(parseExecutionId(null!)).toBeNull()
        expect(parseExecutionId(undefined!)).toBeNull()
      })
    })

    describe('getLocalNodeIdFromExecutionId', () => {
      it('should extract local node ID from simple ID', () => {
        expect(getLocalNodeIdFromExecutionId('123')).toBe('123')
      })

      it('should extract local node ID from complex ID', () => {
        expect(getLocalNodeIdFromExecutionId('123:456:789')).toBe('789')
      })

      it('should return null for invalid input', () => {
        expect(getLocalNodeIdFromExecutionId('')).toBeNull()
      })
    })

    describe('getSubgraphPathFromExecutionId', () => {
      it('should return empty array for root node', () => {
        expect(getSubgraphPathFromExecutionId('123')).toEqual([])
      })

      it('should return subgraph path for nested node', () => {
        expect(getSubgraphPathFromExecutionId('123:456:789')).toEqual([
          '123',
          '456'
        ])
      })

      it('should return empty array for invalid input', () => {
        expect(getSubgraphPathFromExecutionId('')).toEqual([])
      })
    })

    describe('visitGraphNodes', () => {
      it('should visit all nodes in graph', () => {
        const visited: number[] = []
        const nodes = [createMockNode(1), createMockNode(2), createMockNode(3)]
        const graph = createMockGraph(nodes)

        visitGraphNodes(graph, (node) => {
          visited.push(node.id as number)
        })

        expect(visited).toEqual([1, 2, 3])
      })

      it('should handle empty graph', () => {
        const visited: number[] = []
        const graph = createMockGraph([])

        visitGraphNodes(graph, (node) => {
          visited.push(node.id as number)
        })

        expect(visited).toEqual([])
      })
    })

    describe('traverseSubgraphPath', () => {
      it('should return start graph for empty path', () => {
        const graph = createMockGraph([])
        const result = traverseSubgraphPath(graph, [])
        expect(result).toBe(graph)
      })

      it('should traverse single level', () => {
        const subgraph = createMockSubgraph('sub-uuid', [])
        const node = createMockNode('1', { isSubgraph: true, subgraph })
        const graph = createMockGraph([node])

        const result = traverseSubgraphPath(graph, ['1'])
        expect(result).toBe(subgraph)
      })

      it('should traverse multiple levels', () => {
        const deepSubgraph = createMockSubgraph('deep-uuid', [])
        const midNode = createMockNode('2', {
          isSubgraph: true,
          subgraph: deepSubgraph
        })
        const midSubgraph = createMockSubgraph('mid-uuid', [midNode])
        const topNode = createMockNode('1', {
          isSubgraph: true,
          subgraph: midSubgraph
        })
        const graph = createMockGraph([topNode])

        const result = traverseSubgraphPath(graph, ['1', '2'])
        expect(result).toBe(deepSubgraph)
      })

      it('should return null for invalid path', () => {
        const graph = createMockGraph([createMockNode('1')])
        const result = traverseSubgraphPath(graph, ['999'])
        expect(result).toBeNull()
      })
    })
  })

  describe('Main functions', () => {
    describe('triggerCallbackOnAllNodes', () => {
      it('should trigger callbacks on all nodes in a flat graph', () => {
        const callback1 = vi.fn()
        const callback2 = vi.fn()

        const node1 = createMockNode(1, { callback: callback1 })
        const node2 = createMockNode(2, { callback: callback2 })
        const node3 = createMockNode(3) // No callback

        const graph = createMockGraph([node1, node2, node3])

        triggerCallbackOnAllNodes(graph, 'onExecutionStart')

        expect(callback1).toHaveBeenCalledOnce()
        expect(callback2).toHaveBeenCalledOnce()
      })

      it('should trigger callbacks on nodes in subgraphs', () => {
        const callback1 = vi.fn()
        const callback2 = vi.fn()
        const callback3 = vi.fn()

        // Create a subgraph with one node
        const subNode = createMockNode(100, { callback: callback3 })
        const subgraph = createMockSubgraph('sub-uuid', [subNode])

        // Create main graph with two nodes, one being a subgraph
        const node1 = createMockNode(1, { callback: callback1 })
        const node2 = createMockNode(2, {
          isSubgraph: true,
          subgraph,
          callback: callback2
        })

        const graph = createMockGraph([node1, node2])

        triggerCallbackOnAllNodes(graph, 'onExecutionStart')

        expect(callback1).toHaveBeenCalledOnce()
        expect(callback2).toHaveBeenCalledOnce()
        expect(callback3).toHaveBeenCalledOnce()
      })

      it('should handle nested subgraphs', () => {
        const callbacks = [vi.fn(), vi.fn(), vi.fn(), vi.fn()]

        // Create deeply nested structure
        const deepNode = createMockNode(300, { callback: callbacks[3] })
        const deepSubgraph = createMockSubgraph('deep-uuid', [deepNode])

        const midNode1 = createMockNode(200, { callback: callbacks[2] })
        const midNode2 = createMockNode(201, {
          isSubgraph: true,
          subgraph: deepSubgraph
        })
        const midSubgraph = createMockSubgraph('mid-uuid', [midNode1, midNode2])

        const node1 = createMockNode(1, { callback: callbacks[0] })
        const node2 = createMockNode(2, {
          isSubgraph: true,
          subgraph: midSubgraph,
          callback: callbacks[1]
        })

        const graph = createMockGraph([node1, node2])

        triggerCallbackOnAllNodes(graph, 'onExecutionStart')

        callbacks.forEach((cb) => expect(cb).toHaveBeenCalledOnce())
      })
    })

    describe('collectAllNodes', () => {
      it('should collect all nodes from a flat graph', () => {
        const nodes = [createMockNode(1), createMockNode(2), createMockNode(3)]

        const graph = createMockGraph(nodes)
        const collected = collectAllNodes(graph)

        expect(collected).toHaveLength(3)
        expect(collected.map((n) => n.id)).toEqual([1, 2, 3])
      })

      it('should collect nodes from subgraphs', () => {
        const subNode = createMockNode(100)
        const subgraph = createMockSubgraph('sub-uuid', [subNode])

        const nodes = [
          createMockNode(1),
          createMockNode(2, { isSubgraph: true, subgraph })
        ]

        const graph = createMockGraph(nodes)
        const collected = collectAllNodes(graph)

        expect(collected).toHaveLength(3)
        expect(collected.map((n) => n.id)).toContain(100)
      })

      it('should filter nodes when filter function provided', () => {
        const nodes = [createMockNode(1), createMockNode(2), createMockNode(3)]

        const graph = createMockGraph(nodes)
        const collected = collectAllNodes(graph, (node) => Number(node.id) > 1)

        expect(collected).toHaveLength(2)
        expect(collected.map((n) => n.id)).toEqual([2, 3])
      })
    })

    describe('mapAllNodes', () => {
      it('should map over all nodes in a flat graph', () => {
        const nodes = [createMockNode(1), createMockNode(2), createMockNode(3)]
        const graph = createMockGraph(nodes)

        const results = mapAllNodes(graph, (node) => node.id)

        expect(results).toEqual([1, 2, 3])
      })

      it('should map over nodes in subgraphs', () => {
        const subNode = createMockNode(100)
        const subgraph = createMockSubgraph('sub-uuid', [subNode])

        const nodes = [
          createMockNode(1),
          createMockNode(2, { isSubgraph: true, subgraph })
        ]

        const graph = createMockGraph(nodes)
        const results = mapAllNodes(graph, (node) => node.id)

        expect(results).toHaveLength(3)
        expect(results).toContain(100)
      })

      it('should exclude undefined results', () => {
        const nodes = [createMockNode(1), createMockNode(2), createMockNode(3)]
        const graph = createMockGraph(nodes)

        const results = mapAllNodes(graph, (node) => {
          return Number(node.id) > 1 ? node.id : undefined
        })

        expect(results).toEqual([2, 3])
      })

      it('should handle deeply nested structures', () => {
        const deepNode = createMockNode(300)
        const deepSubgraph = createMockSubgraph('deep-uuid', [deepNode])

        const midNode = createMockNode(200)
        const midSubgraphNode = createMockNode(201, {
          isSubgraph: true,
          subgraph: deepSubgraph
        })
        const midSubgraph = createMockSubgraph('mid-uuid', [
          midNode,
          midSubgraphNode
        ])

        const nodes = [
          createMockNode(1),
          createMockNode(2, { isSubgraph: true, subgraph: midSubgraph })
        ]

        const graph = createMockGraph(nodes)
        const results = mapAllNodes(graph, (node) => `node-${node.id}`)

        expect(results).toHaveLength(5)
        expect(results).toContain('node-300')
      })
    })

    describe('forEachNode', () => {
      it('should execute function on all nodes in a flat graph', () => {
        const nodes = [createMockNode(1), createMockNode(2), createMockNode(3)]
        const graph = createMockGraph(nodes)

        const visited: number[] = []
        forEachNode(graph, (node) => {
          visited.push(node.id as number)
        })

        expect(visited).toHaveLength(3)
        expect(visited).toContain(1)
        expect(visited).toContain(2)
        expect(visited).toContain(3)
      })

      it('should execute function on nodes in subgraphs', () => {
        const subNode = createMockNode(100)
        const subgraph = createMockSubgraph('sub-uuid', [subNode])

        const nodes = [
          createMockNode(1),
          createMockNode(2, { isSubgraph: true, subgraph })
        ]

        const graph = createMockGraph(nodes)

        const visited: number[] = []
        forEachNode(graph, (node) => {
          visited.push(node.id as number)
        })

        expect(visited).toHaveLength(3)
        expect(visited).toContain(100)
      })

      it('should allow node mutations', () => {
        const nodes = [createMockNode(1), createMockNode(2), createMockNode(3)]
        const graph = createMockGraph(nodes)

        // Add a title property to each node
        forEachNode(graph, (node) => {
          node.title = `Node ${node.id}`
        })

        expect(nodes[0]).toHaveProperty('title', 'Node 1')
        expect(nodes[1]).toHaveProperty('title', 'Node 2')
        expect(nodes[2]).toHaveProperty('title', 'Node 3')
      })

      it('should handle node type matching for subgraph references', () => {
        const subgraphId = 'my-subgraph-123'
        const nodes = [
          createMockNode(1),
          { ...createMockNode(2), type: subgraphId } as LGraphNode,
          createMockNode(3),
          { ...createMockNode(4), type: subgraphId } as LGraphNode
        ]
        const graph = createMockGraph(nodes)

        const matchingNodes: number[] = []
        forEachNode(graph, (node) => {
          if (node.type === subgraphId) {
            matchingNodes.push(node.id as number)
          }
        })

        expect(matchingNodes).toEqual([2, 4])
      })
    })

    describe('findNodeInHierarchy', () => {
      it('should find node in root graph', () => {
        const nodes = [createMockNode(1), createMockNode(2), createMockNode(3)]

        const graph = createMockGraph(nodes)
        const found = findNodeInHierarchy(graph, 2)

        expect(found).toBeTruthy()
        expect(found?.id).toBe(2)
      })

      it('should find node in subgraph', () => {
        const subNode = createMockNode(100)
        const subgraph = createMockSubgraph('sub-uuid', [subNode])

        const nodes = [
          createMockNode(1),
          createMockNode(2, { isSubgraph: true, subgraph })
        ]

        const graph = createMockGraph(nodes)
        const found = findNodeInHierarchy(graph, 100)

        expect(found).toBeTruthy()
        expect(found?.id).toBe(100)
      })

      it('should return null for non-existent node', () => {
        const nodes = [createMockNode(1), createMockNode(2)]
        const graph = createMockGraph(nodes)

        const found = findNodeInHierarchy(graph, 999)
        expect(found).toBeNull()
      })
    })

    describe('findSubgraphByUuid', () => {
      it('should find subgraph by UUID', () => {
        const targetUuid = 'target-uuid'
        const subgraph = createMockSubgraph(targetUuid, [])

        const nodes = [
          createMockNode(1),
          createMockNode(2, { isSubgraph: true, subgraph })
        ]

        const graph = createMockGraph(nodes)
        const found = findSubgraphByUuid(graph, targetUuid)

        expect(found).toBe(subgraph)
        expect(found?.id).toBe(targetUuid)
      })

      it('should find nested subgraph', () => {
        const targetUuid = 'deep-uuid'
        const deepSubgraph = createMockSubgraph(targetUuid, [])

        const midSubgraph = createMockSubgraph('mid-uuid', [
          createMockNode(200, { isSubgraph: true, subgraph: deepSubgraph })
        ])

        const graph = createMockGraph([
          createMockNode(1, { isSubgraph: true, subgraph: midSubgraph })
        ])

        const found = findSubgraphByUuid(graph, targetUuid)

        expect(found).toBe(deepSubgraph)
        expect(found?.id).toBe(targetUuid)
      })

      it('should return null for non-existent UUID', () => {
        const subgraph = createMockSubgraph('some-uuid', [])
        const graph = createMockGraph([
          createMockNode(1, { isSubgraph: true, subgraph })
        ])

        const found = findSubgraphByUuid(graph, 'non-existent-uuid')
        expect(found).toBeNull()
      })
    })

    describe('getNodeByExecutionId', () => {
      it('should find node in root graph', () => {
        const nodes = [createMockNode('123'), createMockNode('456')]

        const graph = createMockGraph(nodes)
        const found = getNodeByExecutionId(graph, '123')

        expect(found).toBeTruthy()
        expect(found?.id).toBe('123')
      })

      it('should find node in subgraph using execution path', () => {
        const targetNode = createMockNode('789')
        const subgraph = createMockSubgraph('sub-uuid', [targetNode])

        const subgraphNode = createMockNode('456', {
          isSubgraph: true,
          subgraph
        })

        const graph = createMockGraph([createMockNode('123'), subgraphNode])

        const found = getNodeByExecutionId(graph, '456:789')

        expect(found).toBe(targetNode)
        expect(found?.id).toBe('789')
      })

      it('should handle deeply nested execution paths', () => {
        const targetNode = createMockNode('999')
        const deepSubgraph = createMockSubgraph('deep-uuid', [targetNode])

        const midNode = createMockNode('456', {
          isSubgraph: true,
          subgraph: deepSubgraph
        })
        const midSubgraph = createMockSubgraph('mid-uuid', [midNode])

        const topNode = createMockNode('123', {
          isSubgraph: true,
          subgraph: midSubgraph
        })

        const graph = createMockGraph([topNode])

        const found = getNodeByExecutionId(graph, '123:456:999')

        expect(found).toBe(targetNode)
        expect(found?.id).toBe('999')
      })

      it('should return null for invalid path', () => {
        const subgraph = createMockSubgraph('sub-uuid', [createMockNode('789')])
        const graph = createMockGraph([
          createMockNode('456', { isSubgraph: true, subgraph })
        ])

        // Wrong path - node 123 doesn't exist
        const found = getNodeByExecutionId(graph, '123:789')
        expect(found).toBeNull()
      })

      it('should return null for invalid execution ID', () => {
        const graph = createMockGraph([createMockNode('123')])
        const found = getNodeByExecutionId(graph, '')
        expect(found).toBeNull()
      })
    })

    describe('getNodeByLocatorId', () => {
      it('should find node in root graph', () => {
        const nodes = [createMockNode('123'), createMockNode('456')]

        const graph = createMockGraph(nodes)
        const found = getNodeByLocatorId(graph, '123')

        expect(found).toBeTruthy()
        expect(found?.id).toBe('123')
      })

      it('should find node in subgraph using UUID format', () => {
        const targetUuid = 'a1b2c3d4-e5f6-7890-abcd-ef1234567890'
        const targetNode = createMockNode('789')
        const subgraph = createMockSubgraph(targetUuid, [targetNode])

        const graph = createMockGraph([
          createMockNode('123'),
          createMockNode('456', { isSubgraph: true, subgraph })
        ])

        const locatorId = `${targetUuid}:789`
        const found = getNodeByLocatorId(graph, locatorId)

        expect(found).toBe(targetNode)
        expect(found?.id).toBe('789')
      })

      it('should return null for invalid locator ID', () => {
        const graph = createMockGraph([createMockNode('123')])

        const found = getNodeByLocatorId(graph, 'invalid:::format')
        expect(found).toBeNull()
      })

      it('should return null when subgraph UUID not found', () => {
        const subgraph = createMockSubgraph('some-uuid', [
          createMockNode('789')
        ])
        const graph = createMockGraph([
          createMockNode('456', { isSubgraph: true, subgraph })
        ])

        const locatorId = 'non-existent-uuid:789'
        const found = getNodeByLocatorId(graph, locatorId)
        expect(found).toBeNull()
      })
    })

    describe('getRootGraph', () => {
      it('should return the same graph if it is already root', () => {
        const graph = createMockGraph([])
        expect(getRootGraph(graph)).toBe(graph)
      })

      it('should return root graph from subgraph', () => {
        const rootGraph = createMockGraph([])
        const subgraph = createMockSubgraph('sub-uuid', [])
        ;(subgraph as Subgraph & { rootGraph: LGraph }).rootGraph = rootGraph

        expect(getRootGraph(subgraph)).toBe(rootGraph)
      })

      it('should return root graph from deeply nested subgraph', () => {
        const rootGraph = createMockGraph([])
        const midSubgraph = createMockSubgraph('mid-uuid', [])
        const deepSubgraph = createMockSubgraph('deep-uuid', [])
        ;(midSubgraph as Subgraph & { rootGraph: LGraph }).rootGraph = rootGraph
        ;(
          deepSubgraph as Subgraph & { rootGraph: LGraph | Subgraph }
        ).rootGraph = midSubgraph

        expect(getRootGraph(deepSubgraph)).toBe(rootGraph)
      })
    })

    describe('forEachSubgraphNode', () => {
      it('should apply function to all nodes matching subgraph type', () => {
        const subgraphId = 'my-subgraph-123'
        const nodes = [
          createMockNode(1),
          { ...createMockNode(2), type: subgraphId } as LGraphNode,
          createMockNode(3),
          { ...createMockNode(4), type: subgraphId } as LGraphNode
        ]
        const graph = createMockGraph(nodes)

        const matchingIds: number[] = []
        forEachSubgraphNode(graph, subgraphId, (node) => {
          matchingIds.push(node.id as number)
        })

        expect(matchingIds).toEqual([2, 4])
      })

      it('should work with root graph directly', () => {
        const subgraphId = 'target-subgraph'
        const rootNodes = [
          { ...createMockNode(1), type: subgraphId } as LGraphNode,
          createMockNode(2),
          { ...createMockNode(3), type: subgraphId } as LGraphNode
        ]
        const rootGraph = createMockGraph(rootNodes)

        const matchingIds: number[] = []
        forEachSubgraphNode(rootGraph, subgraphId, (node) => {
          matchingIds.push(node.id as number)
        })

        expect(matchingIds).toEqual([1, 3])
      })

      it('should handle null inputs gracefully', () => {
        const fn = vi.fn()

        forEachSubgraphNode(null, 'id', fn)
        forEachSubgraphNode(createMockGraph([]), null, fn)
        forEachSubgraphNode(null, null, fn)

        expect(fn).not.toHaveBeenCalled()
      })

      it('should allow node mutations like title updates', () => {
        const subgraphId = 'my-subgraph'
        const nodes = [
          { ...createMockNode(1), type: subgraphId } as LGraphNode,
          { ...createMockNode(2), type: subgraphId } as LGraphNode,
          createMockNode(3)
        ]
        const graph = createMockGraph(nodes)

        forEachSubgraphNode(graph, subgraphId, (node) => {
          node.title = 'Updated Title'
        })

        expect(nodes[0]).toHaveProperty('title', 'Updated Title')
        expect(nodes[1]).toHaveProperty('title', 'Updated Title')
        expect(nodes[2]).not.toHaveProperty('title', 'Updated Title')
      })
    })

    describe('mapSubgraphNodes', () => {
      it('should map over nodes matching subgraph type', () => {
        const subgraphId = 'my-subgraph-123'
        const nodes = [
          createMockNode(1),
          { ...createMockNode(2), type: subgraphId } as LGraphNode,
          createMockNode(3),
          { ...createMockNode(4), type: subgraphId } as LGraphNode
        ]
        const graph = createMockGraph(nodes)

        const results = mapSubgraphNodes(graph, subgraphId, (node) => node.id)

        expect(results).toEqual([2, 4])
      })

      it('should return empty array for null inputs', () => {
        expect(mapSubgraphNodes(null, 'id', (n) => n.id)).toEqual([])
        expect(
          mapSubgraphNodes(createMockGraph([]), null, (n) => n.id)
        ).toEqual([])
      })

      it('should work with complex transformations', () => {
        const subgraphId = 'target'
        const nodes = [
          { ...createMockNode(1), type: subgraphId } as LGraphNode,
          { ...createMockNode(2), type: 'other' } as LGraphNode,
          { ...createMockNode(3), type: subgraphId } as LGraphNode
        ]
        const graph = createMockGraph(nodes)

        const results = mapSubgraphNodes(graph, subgraphId, (node) => ({
          id: node.id,
          isTarget: true
        }))

        expect(results).toEqual([
          { id: 1, isTarget: true },
          { id: 3, isTarget: true }
        ])
      })
    })

    describe('getAllNonIoNodesInSubgraph', () => {
      it('should filter out SubgraphInputNode and SubgraphOutputNode', () => {
        const nodes = [
          { id: 'input', constructor: { comfyClass: 'SubgraphInputNode' } },
          { id: 'output', constructor: { comfyClass: 'SubgraphOutputNode' } },
          { id: 'user1', constructor: { comfyClass: 'CLIPTextEncode' } },
          { id: 'user2', constructor: { comfyClass: 'KSampler' } }
        ] as LGraphNode[]

        const subgraph = createMockSubgraph('sub-uuid', nodes)
        const nonIoNodes = getAllNonIoNodesInSubgraph(subgraph)

        expect(nonIoNodes).toHaveLength(2)
        expect(nonIoNodes.map((n) => n.id)).toEqual(['user1', 'user2'])
      })

      it('should handle subgraph with only IO nodes', () => {
        const nodes = [
          { id: 'input', constructor: { comfyClass: 'SubgraphInputNode' } },
          { id: 'output', constructor: { comfyClass: 'SubgraphOutputNode' } }
        ] as LGraphNode[]

        const subgraph = createMockSubgraph('sub-uuid', nodes)
        const nonIoNodes = getAllNonIoNodesInSubgraph(subgraph)

        expect(nonIoNodes).toHaveLength(0)
      })

      it('should handle subgraph with only user nodes', () => {
        const nodes = [
          { id: 'user1', constructor: { comfyClass: 'CLIPTextEncode' } },
          { id: 'user2', constructor: { comfyClass: 'KSampler' } }
        ] as LGraphNode[]

        const subgraph = createMockSubgraph('sub-uuid', nodes)
        const nonIoNodes = getAllNonIoNodesInSubgraph(subgraph)

        expect(nonIoNodes).toHaveLength(2)
        expect(nonIoNodes).toEqual(nodes)
      })

      it('should handle empty subgraph', () => {
        const subgraph = createMockSubgraph('sub-uuid', [])
        const nonIoNodes = getAllNonIoNodesInSubgraph(subgraph)

        expect(nonIoNodes).toHaveLength(0)
      })
    })

    describe('traverseNodesDepthFirst', () => {
      it('should traverse nodes in depth-first order', () => {
        const visited: string[] = []
        const nodes = [
          createMockNode('1'),
          createMockNode('2'),
          createMockNode('3')
        ]

        traverseNodesDepthFirst(nodes, {
          visitor: (node, context) => {
            visited.push(`${node.id}:${context}`)
            return `${context}-${node.id}`
          },
          initialContext: 'root'
        })

        expect(visited).toEqual(['3:root', '2:root', '1:root']) // DFS processes in LIFO order
      })

      it('should traverse into subgraphs when expandSubgraphs is true', () => {
        const visited: string[] = []
        const subNode = createMockNode('sub1')
        const subgraph = createMockSubgraph('sub-uuid', [subNode])
        const nodes = [
          createMockNode('1'),
          createMockNode('2', { isSubgraph: true, subgraph })
        ]

        traverseNodesDepthFirst(nodes, {
          visitor: (node, depth: number) => {
            visited.push(`${node.id}:${depth}`)
            return depth + 1
          },
          initialContext: 0
        })

        expect(visited).toEqual(['2:0', 'sub1:1', '1:0']) // DFS: last node first, then its children
      })

      it('should skip subgraphs when expandSubgraphs is false', () => {
        const visited: string[] = []
        const subNode = createMockNode('sub1')
        const subgraph = createMockSubgraph('sub-uuid', [subNode])
        const nodes = [
          createMockNode('1'),
          createMockNode('2', { isSubgraph: true, subgraph })
        ]

        traverseNodesDepthFirst(nodes, {
          visitor: (node, context) => {
            visited.push(String(node.id))
            return context
          },
          initialContext: null,
          expandSubgraphs: false
        })

        expect(visited).toEqual(['2', '1']) // DFS processes in LIFO order
        expect(visited).not.toContain('sub1')
      })

      it('should handle deeply nested subgraphs', () => {
        const visited: string[] = []

        const deepNode = createMockNode('300')
        const deepSubgraph = createMockSubgraph('deep-uuid', [deepNode])

        const midNode = createMockNode('200', {
          isSubgraph: true,
          subgraph: deepSubgraph
        })
        const midSubgraph = createMockSubgraph('mid-uuid', [midNode])

        const topNode = createMockNode('100', {
          isSubgraph: true,
          subgraph: midSubgraph
        })

        traverseNodesDepthFirst([topNode], {
          visitor: (node, path: string) => {
            visited.push(`${node.id}:${path}`)
            return path ? `${path}/${node.id}` : String(node.id)
          },
          initialContext: ''
        })

        expect(visited).toEqual(['100:', '200:100', '300:100/200'])
      })
    })

    describe('collectFromNodes', () => {
      it('should collect data from all nodes', () => {
        const nodes = [
          createMockNode('1'),
          createMockNode('2'),
          createMockNode('3')
        ]

        const results = collectFromNodes(nodes, {
          collector: (node) => `node-${node.id}`,
          contextBuilder: (_node, context) => context,
          initialContext: null
        })

        expect(results).toEqual(['node-3', 'node-2', 'node-1']) // DFS processes in LIFO order
      })

      it('should filter out null results', () => {
        const nodes = [
          createMockNode('1'),
          createMockNode('2'),
          createMockNode('3')
        ]

        const results = collectFromNodes(nodes, {
          collector: (node) => (Number(node.id) > 1 ? `node-${node.id}` : null),
          contextBuilder: (_node, context) => context,
          initialContext: null
        })

        expect(results).toEqual(['node-3', 'node-2']) // DFS processes in LIFO order, node-1 filtered out
      })

      it('should collect from subgraphs with context', () => {
        const subNodes = [createMockNode('10'), createMockNode('11')]
        const subgraph = createMockSubgraph('sub-uuid', subNodes)
        const nodes = [
          createMockNode('1'),
          createMockNode('2', { isSubgraph: true, subgraph })
        ]

        const results = collectFromNodes(nodes, {
          collector: (node, prefix: string) => `${prefix}${node.id}`,
          contextBuilder: (node, prefix: string) => `${prefix}${node.id}-`,
          initialContext: 'node-',
          expandSubgraphs: true
        })

        expect(results).toEqual([
          'node-2',
          'node-2-10', // Actually processes in original order within subgraph
          'node-2-11',
          'node-1'
        ])
      })

      it('should not expand subgraphs when expandSubgraphs is false', () => {
        const subNodes = [createMockNode('10'), createMockNode('11')]
        const subgraph = createMockSubgraph('sub-uuid', subNodes)
        const nodes = [
          createMockNode('1'),
          createMockNode('2', { isSubgraph: true, subgraph })
        ]

        const results = collectFromNodes(nodes, {
          collector: (node) => String(node.id),
          contextBuilder: (_node, context) => context,
          initialContext: null,
          expandSubgraphs: false
        })

        expect(results).toEqual(['2', '1']) // DFS processes in LIFO order
      })
    })

    describe('getExecutionIdsForSelectedNodes', () => {
      it('should return simple IDs for top-level nodes', () => {
        const graph = createMockGraph([])
        createMockNode('123', { graph })
        createMockNode('456', { graph })
        createMockNode('789', { graph })

        const executionIds = getExecutionIdsForSelectedNodes(graph.nodes)

        expect(executionIds).toEqual(['789', '456', '123']) // DFS processes in LIFO order
      })

      it('should expand subgraph nodes to include all children', () => {
        const graph = createMockGraph([])
        const subNodes = [createMockNode('10'), createMockNode('11')]
        const subgraph = createMockSubgraph('sub-uuid', subNodes)
        createMockNode('1', { graph })
        createMockNode('2', { isSubgraph: true, subgraph, graph })

        const executionIds = getExecutionIdsForSelectedNodes(graph.nodes)

        expect(executionIds).toEqual(['2', '2:10', '2:11', '1']) // DFS: node 2 first, then its children
      })

      it('should handle deeply nested subgraphs correctly', () => {
        const graph = createMockGraph([])
        const deepNodes = [createMockNode('30'), createMockNode('31')]
        const deepSubgraph = createMockSubgraph('deep-uuid', deepNodes)

        const midNode = createMockNode('20', {
          isSubgraph: true,
          subgraph: deepSubgraph
        })
        const midSubgraph = createMockSubgraph('mid-uuid', [midNode])

        const topNode = createMockNode('10', {
          isSubgraph: true,
          subgraph: midSubgraph,
          graph
        })

        const executionIds = getExecutionIdsForSelectedNodes([topNode])

        expect(executionIds).toEqual(['10', '10:20', '10:20:30', '10:20:31'])
      })

      it('should handle mixed selection of regular and subgraph nodes', () => {
        const graph = createMockGraph([])
        const subNodes = [createMockNode('100'), createMockNode('101')]
        const subgraph = createMockSubgraph('sub-uuid', subNodes)

        createMockNode('1', { graph })
        createMockNode('2', { isSubgraph: true, subgraph, graph })
        createMockNode('3', { graph })

        const executionIds = getExecutionIdsForSelectedNodes(graph.nodes)

        expect(executionIds).toEqual([
          '3',
          '2',
          '2:100', // Subgraph children in original order
          '2:101',
          '1'
        ])
      })

      it('should handle empty selection', () => {
        const executionIds = getExecutionIdsForSelectedNodes([])
        expect(executionIds).toEqual([])
      })

      it('should handle subgraph with no children', () => {
        const graph = createMockGraph([])
        const emptySubgraph = createMockSubgraph('empty-uuid', [])
        const node = createMockNode('1', {
          isSubgraph: true,
          subgraph: emptySubgraph,
          graph
        })

        const executionIds = getExecutionIdsForSelectedNodes([node])

        expect(executionIds).toEqual(['1'])
      })

      it('should handle nodes with very long execution paths', () => {
        // Create a chain of 10 nested subgraphs
        let currentSubgraph = createMockSubgraph('deep-10', [
          createMockNode('10')
        ])

        for (let i = 9; i >= 1; i--) {
          const node = createMockNode(`${i}0`, {
            isSubgraph: true,
            subgraph: currentSubgraph
          })
          currentSubgraph = createMockSubgraph(`deep-${i}`, [node])
        }

        const graph = createMockGraph([])
        const topNode = createMockNode('1', {
          isSubgraph: true,
          subgraph: currentSubgraph,
          graph
        })

        const executionIds = getExecutionIdsForSelectedNodes([topNode])

        expect(executionIds).toHaveLength(11)
        expect(executionIds[0]).toBe('1')
        expect(executionIds[10]).toBe('1:10:20:30:40:50:60:70:80:90:10')
      })

      it('should handle duplicate node IDs in different subgraphs', () => {
        // Create two subgraphs with nodes that have the same IDs
        const subgraph1 = createMockSubgraph('sub1-uuid', [
          createMockNode('100'),
          createMockNode('101')
        ])

        const subgraph2 = createMockSubgraph('sub2-uuid', [
          createMockNode('100'), // Same ID as in subgraph1
          createMockNode('101') // Same ID as in subgraph1
        ])

        const graph = createMockGraph([])
        createMockNode('1', { isSubgraph: true, subgraph: subgraph1, graph })
        createMockNode('2', { isSubgraph: true, subgraph: subgraph2, graph })

        const executionIds = getExecutionIdsForSelectedNodes(graph.nodes)

        expect(executionIds).toEqual([
          '2',
          '2:100',
          '2:101',
          '1',
          '1:100',
          '1:101'
        ])
      })

      it('should handle subgraphs with many children efficiently', () => {
        // Create a subgraph with 100 nodes
        const manyNodes = []
        for (let i = 0; i < 100; i++) {
          manyNodes.push(createMockNode(`child-${i}`))
        }
        const bigSubgraph = createMockSubgraph('big-uuid', manyNodes)

        const graph = createMockGraph([])
        const node = createMockNode('parent', {
          isSubgraph: true,
          subgraph: bigSubgraph,
          graph
        })

        const start = performance.now()
        const executionIds = getExecutionIdsForSelectedNodes([node])
        const duration = performance.now() - start

        expect(executionIds).toHaveLength(101)
        expect(executionIds[0]).toBe('parent')
        expect(executionIds[100]).toBe('parent:child-99') // Due to backward iteration optimization

        // Should complete quickly even with many nodes
        expect(duration).toBeLessThan(50)
      })

      it('should handle selection of nodes at different depths', () => {
        // Create a complex nested structure
        const deepNode = createMockNode('300')
        const deepSubgraph = createMockSubgraph('deep-uuid', [deepNode])

        const midNode1 = createMockNode('201')
        const midNode2 = createMockNode('202', {
          isSubgraph: true,
          subgraph: deepSubgraph
        })
        const midSubgraph = createMockSubgraph('mid-uuid', [midNode1, midNode2])

        const graph = createMockGraph([])
        // Select nodes at different nesting levels
        createMockNode('100', {
          isSubgraph: true,
          subgraph: midSubgraph,
          graph
        })
        createMockNode('1', { graph })
        createMockNode('2', { graph })

        const executionIds = getExecutionIdsForSelectedNodes(graph.nodes)

        expect(executionIds).toContain('1')
        expect(executionIds).toContain('2')
        expect(executionIds).toContain('100')
        expect(executionIds).toContain('100:201')
        expect(executionIds).toContain('100:202')
        expect(executionIds).toContain('100:202:300')
      })
      it('should resolve full execution path of a node inside a subgraph', () => {
        const graph = createMockGraph([])
        const subgraph = createMockSubgraph('sub-uuid', [], graph)
        createMockNode('11', { graph: subgraph })
        createMockNode('10', { graph: subgraph })
        createMockNode('2', { isSubgraph: true, subgraph, graph })

        const executionIds = getExecutionIdsForSelectedNodes(subgraph.nodes)

        expect(executionIds).toEqual(['2:10', '2:11'])
      })
    })
  })
})
