import { describe, expect, it } from 'vitest'

import { LGraphNode } from '@/lib/litegraph/src/litegraph'
import { filterOutputNodes, isOutputNode } from '@/utils/nodeFilterUtil'

describe('nodeFilterUtil', () => {
  // Helper to create a mock node
  const createMockNode = (
    id: number,
    isOutputNode: boolean = false
  ): LGraphNode => {
    // Create a custom class with the nodeData static property
    class MockNode extends LGraphNode {
      static override nodeData = isOutputNode ? { output_node: true } : {}
    }

    const node = new MockNode('')
    node.id = id
    return node
  }

  describe('filterOutputNodes', () => {
    it('should return empty array when given empty array', () => {
      const result = filterOutputNodes([])
      expect(result).toEqual([])
    })

    it('should filter out non-output nodes', () => {
      const nodes = [
        createMockNode(1, false),
        createMockNode(2, true),
        createMockNode(3, false),
        createMockNode(4, true)
      ]

      const result = filterOutputNodes(nodes)
      expect(result).toHaveLength(2)
      expect(result.map((n) => n.id)).toEqual([2, 4])
    })

    it('should return all nodes if all are output nodes', () => {
      const nodes = [
        createMockNode(1, true),
        createMockNode(2, true),
        createMockNode(3, true)
      ]

      const result = filterOutputNodes(nodes)
      expect(result).toHaveLength(3)
      expect(result).toEqual(nodes)
    })

    it('should return empty array if no output nodes', () => {
      const nodes = [
        createMockNode(1, false),
        createMockNode(2, false),
        createMockNode(3, false)
      ]

      const result = filterOutputNodes(nodes)
      expect(result).toHaveLength(0)
    })

    it('should handle nodes without nodeData', () => {
      // Create a plain LGraphNode without custom constructor
      const node = new LGraphNode('')
      node.id = 1

      const result = filterOutputNodes([node])
      expect(result).toHaveLength(0)
    })

    it('should handle nodes with undefined output_node', () => {
      class MockNodeWithEmptyData extends LGraphNode {
        static override nodeData = {}
      }

      const node = new MockNodeWithEmptyData('')
      node.id = 1

      const result = filterOutputNodes([node])
      expect(result).toHaveLength(0)
    })
  })

  describe('isOutputNode', () => {
    it('should filter selected nodes to only output nodes', () => {
      const selectedNodes = [
        createMockNode(1, false),
        createMockNode(2, true),
        createMockNode(3, false),
        createMockNode(4, true),
        createMockNode(5, false)
      ]

      const result = selectedNodes.filter(isOutputNode)
      expect(result).toHaveLength(2)
      expect(result.map((n) => n.id)).toEqual([2, 4])
    })

    it('should handle empty selection', () => {
      const emptyNodes: LGraphNode[] = []
      const result = emptyNodes.filter(isOutputNode)
      expect(result).toEqual([])
    })

    it('should handle selection with no output nodes', () => {
      const selectedNodes = [createMockNode(1, false), createMockNode(2, false)]

      const result = selectedNodes.filter(isOutputNode)
      expect(result).toHaveLength(0)
    })
  })
})
