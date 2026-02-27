// TODO: Fix these tests after migration
import { describe, expect, it } from 'vitest'

import {
  LGraph,
  findUsedSubgraphIds,
  getDirectSubgraphIds
} from '@/lib/litegraph/src/litegraph'
import type { UUID } from '@/lib/litegraph/src/litegraph'

import {
  createTestSubgraph,
  createTestSubgraphNode
} from './__fixtures__/subgraphHelpers'

describe.skip('subgraphUtils', () => {
  describe.skip('getDirectSubgraphIds', () => {
    it('should return empty set for graph with no subgraph nodes', () => {
      const graph = new LGraph()
      const result = getDirectSubgraphIds(graph)
      expect(result.size).toBe(0)
    })

    it('should find single subgraph node', () => {
      const graph = new LGraph()
      const subgraph = createTestSubgraph()
      const subgraphNode = createTestSubgraphNode(subgraph)
      graph.add(subgraphNode)

      const result = getDirectSubgraphIds(graph)
      expect(result.size).toBe(1)
      expect(result.has(subgraph.id)).toBe(true)
    })

    it('should find multiple unique subgraph nodes', () => {
      const graph = new LGraph()
      const subgraph1 = createTestSubgraph({ name: 'Subgraph 1' })
      const subgraph2 = createTestSubgraph({ name: 'Subgraph 2' })

      const node1 = createTestSubgraphNode(subgraph1)
      const node2 = createTestSubgraphNode(subgraph2)

      graph.add(node1)
      graph.add(node2)

      const result = getDirectSubgraphIds(graph)
      expect(result.size).toBe(2)
      expect(result.has(subgraph1.id)).toBe(true)
      expect(result.has(subgraph2.id)).toBe(true)
    })

    it('should return unique IDs when same subgraph is used multiple times', () => {
      const graph = new LGraph()
      const subgraph = createTestSubgraph()

      const node1 = createTestSubgraphNode(subgraph, { id: 1 })
      const node2 = createTestSubgraphNode(subgraph, { id: 2 })

      graph.add(node1)
      graph.add(node2)

      const result = getDirectSubgraphIds(graph)
      expect(result.size).toBe(1)
      expect(result.has(subgraph.id)).toBe(true)
    })
  })

  describe.skip('findUsedSubgraphIds', () => {
    it('should handle graph with no subgraphs', () => {
      const graph = new LGraph()
      const registry = new Map<UUID, LGraph>()

      const result = findUsedSubgraphIds(graph, registry)
      expect(result.size).toBe(0)
    })

    it('should find nested subgraphs', () => {
      const rootGraph = new LGraph()
      const subgraph1 = createTestSubgraph({ name: 'Level 1' })
      const subgraph2 = createTestSubgraph({ name: 'Level 2' })

      // Add subgraph1 node to root
      const node1 = createTestSubgraphNode(subgraph1)
      rootGraph.add(node1)

      // Add subgraph2 node inside subgraph1
      const node2 = createTestSubgraphNode(subgraph2)
      subgraph1.add(node2)

      const registry = new Map<UUID, LGraph>([
        [subgraph1.id, subgraph1],
        [subgraph2.id, subgraph2]
      ])

      const result = findUsedSubgraphIds(rootGraph, registry)
      expect(result.size).toBe(2)
      expect(result.has(subgraph1.id)).toBe(true)
      expect(result.has(subgraph2.id)).toBe(true)
    })

    it('should handle circular references without infinite loop', () => {
      const rootGraph = new LGraph()
      const subgraph1 = createTestSubgraph({ name: 'Subgraph 1' })
      const subgraph2 = createTestSubgraph({ name: 'Subgraph 2' })

      // Add subgraph1 to root
      const node1 = createTestSubgraphNode(subgraph1)
      rootGraph.add(node1)

      // Add subgraph2 to subgraph1
      const node2 = createTestSubgraphNode(subgraph2)
      subgraph1.add(node2)

      // Add subgraph1 to subgraph2 (circular reference)
      const node3 = createTestSubgraphNode(subgraph1, { id: 3 })
      subgraph2.add(node3)

      const registry = new Map<UUID, LGraph>([
        [subgraph1.id, subgraph1],
        [subgraph2.id, subgraph2]
      ])

      const result = findUsedSubgraphIds(rootGraph, registry)
      expect(result.size).toBe(2)
      expect(result.has(subgraph1.id)).toBe(true)
      expect(result.has(subgraph2.id)).toBe(true)
    })

    it('should handle missing subgraphs in registry gracefully', () => {
      const rootGraph = new LGraph()
      const subgraph1 = createTestSubgraph({ name: 'Subgraph 1' })
      const subgraph2 = createTestSubgraph({ name: 'Subgraph 2' })

      // Add both subgraph nodes
      const node1 = createTestSubgraphNode(subgraph1)
      const node2 = createTestSubgraphNode(subgraph2)

      rootGraph.add(node1)
      rootGraph.add(node2)

      // Only register subgraph1
      const registry = new Map<UUID, LGraph>([[subgraph1.id, subgraph1]])

      const result = findUsedSubgraphIds(rootGraph, registry)
      expect(result.size).toBe(2)
      expect(result.has(subgraph1.id)).toBe(true)
      expect(result.has(subgraph2.id)).toBe(true) // Still found, just can't recurse into it
    })
  })
})
