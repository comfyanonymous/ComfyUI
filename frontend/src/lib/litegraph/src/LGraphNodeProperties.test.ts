import { beforeEach, describe, expect, it } from 'vitest'

import { LGraphNodeProperties } from '@/lib/litegraph/src/LGraphNodeProperties'
import type { LGraph, LGraphNode } from '@/lib/litegraph/src/litegraph'
import {
  createMockLGraph,
  createMockLGraphNode
} from '@/utils/__tests__/litegraphTestUtils'

describe('LGraphNodeProperties', () => {
  let mockNode: LGraphNode
  let mockGraph: LGraph

  beforeEach(() => {
    mockGraph = createMockLGraph()

    mockNode = createMockLGraphNode({
      id: 123,
      title: 'Test Node',
      flags: {},
      graph: mockGraph
    })
  })

  describe('property tracking', () => {
    it('should track changes to existing properties', () => {
      new LGraphNodeProperties(mockNode)

      mockNode.title = 'New Title'

      expect(mockGraph.trigger).toHaveBeenCalledWith('node:property:changed', {
        nodeId: mockNode.id,
        property: 'title',
        oldValue: 'Test Node',
        newValue: 'New Title'
      })
    })

    it('should track changes to nested properties', () => {
      new LGraphNodeProperties(mockNode)

      mockNode.flags.collapsed = true

      expect(mockGraph.trigger).toHaveBeenCalledWith('node:property:changed', {
        nodeId: mockNode.id,
        property: 'flags.collapsed',
        oldValue: undefined,
        newValue: true
      })
    })

    it('should emit event when value is set to the same value', () => {
      new LGraphNodeProperties(mockNode)

      mockNode.title = 'Test Node' // Same value as original

      expect(mockGraph.trigger).toHaveBeenCalledTimes(1)
    })

    it('should not emit events when node has no graph', () => {
      mockNode.graph = null
      new LGraphNodeProperties(mockNode)

      // Should not throw
      expect(() => {
        mockNode.title = 'New Title'
      }).not.toThrow()
    })
  })

  describe('isTracked', () => {
    it('should correctly identify tracked properties', () => {
      const propManager = new LGraphNodeProperties(mockNode)

      expect(propManager.isTracked('title')).toBe(true)
      expect(propManager.isTracked('flags.collapsed')).toBe(true)
      expect(propManager.isTracked('untracked')).toBe(false)
    })
  })

  describe('serialization behavior', () => {
    it('should not make non-existent properties enumerable', () => {
      new LGraphNodeProperties(mockNode)

      // flags.collapsed doesn't exist initially
      const descriptor = Object.getOwnPropertyDescriptor(
        mockNode.flags,
        'collapsed'
      )
      expect(descriptor?.enumerable).toBe(false)
    })

    it('should make properties enumerable when set to non-default values', () => {
      new LGraphNodeProperties(mockNode)

      mockNode.flags.collapsed = true

      const descriptor = Object.getOwnPropertyDescriptor(
        mockNode.flags,
        'collapsed'
      )
      expect(descriptor?.enumerable).toBe(true)
    })

    it('should make properties non-enumerable when set back to undefined', () => {
      new LGraphNodeProperties(mockNode)

      mockNode.flags.collapsed = true
      mockNode.flags.collapsed = undefined

      const descriptor = Object.getOwnPropertyDescriptor(
        mockNode.flags,
        'collapsed'
      )
      expect(descriptor?.enumerable).toBe(false)
    })

    it('should keep existing properties enumerable', () => {
      // title exists initially
      const initialDescriptor = Object.getOwnPropertyDescriptor(
        mockNode,
        'title'
      )
      expect(initialDescriptor?.enumerable).toBe(true)

      new LGraphNodeProperties(mockNode)

      const afterDescriptor = Object.getOwnPropertyDescriptor(mockNode, 'title')
      expect(afterDescriptor?.enumerable).toBe(true)
    })

    it('should only include non-undefined values in JSON.stringify', () => {
      new LGraphNodeProperties(mockNode)

      // Initially, flags.collapsed shouldn't appear
      let json = JSON.parse(JSON.stringify(mockNode))
      expect(json.flags.collapsed).toBeUndefined()

      // After setting to true, it should appear
      mockNode.flags.collapsed = true
      json = JSON.parse(JSON.stringify(mockNode))
      expect(json.flags.collapsed).toBe(true)

      // After setting to false, it should still appear (false is not undefined)
      mockNode.flags.collapsed = false
      json = JSON.parse(JSON.stringify(mockNode))
      expect(json.flags.collapsed).toBe(false)

      // After setting back to undefined, it should disappear
      mockNode.flags.collapsed = undefined
      json = JSON.parse(JSON.stringify(mockNode))
      expect(json.flags.collapsed).toBeUndefined()
    })
  })
})
