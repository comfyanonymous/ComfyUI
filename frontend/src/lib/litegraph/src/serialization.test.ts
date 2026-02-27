import { describe, expect, it } from 'vitest'

import { LGraph, LGraphNode, LiteGraph } from './litegraph'
import type { NodeInputSlot } from './node/NodeInputSlot'
import type { NodeOutputSlot } from './node/NodeOutputSlot'

class TestNode extends LGraphNode {
  static override title = 'TestNode'
  constructor() {
    super('TestNode')
  }
}

LiteGraph.registerNodeType('test/TestNode', TestNode)

describe('Serialization - Circular Reference Prevention', () => {
  describe('LGraph.toJSON()', () => {
    it('should serialize without circular reference errors', () => {
      const graph = new LGraph()

      expect(() => JSON.stringify(graph)).not.toThrow()
    })

    it('should return serialize() output from toJSON()', () => {
      const graph = new LGraph()
      const serialized = graph.serialize()
      const jsonOutput = graph.toJSON()

      expect(jsonOutput).toEqual(serialized)
    })

    it('should not include list_of_graphcanvas in JSON output', () => {
      const graph = new LGraph()
      const json = JSON.stringify(graph)
      const parsed = JSON.parse(json)

      expect(parsed.list_of_graphcanvas).toBeUndefined()
    })
  })

  describe('NodeSlot.toJSON()', () => {
    it('NodeInputSlot should serialize without circular reference errors', () => {
      const graph = new LGraph()
      const node = LiteGraph.createNode('test/TestNode')!
      graph.add(node)
      node.addInput('test_input', 'TEST')

      const inputSlot = node.inputs[0]

      expect(() => JSON.stringify(inputSlot)).not.toThrow()
    })

    it('NodeOutputSlot should serialize without circular reference errors', () => {
      const graph = new LGraph()
      const node = LiteGraph.createNode('test/TestNode')!
      graph.add(node)
      node.addOutput('test_output', 'TEST')

      const outputSlot = node.outputs[0]

      expect(() => JSON.stringify(outputSlot)).not.toThrow()
    })

    it('NodeInputSlot.toJSON() should not include _node reference', () => {
      const graph = new LGraph()
      const node = LiteGraph.createNode('test/TestNode')!
      graph.add(node)
      node.addInput('test_input', 'TEST')

      const inputSlot = node.inputs[0] as NodeInputSlot
      const json = inputSlot.toJSON()

      expect('_node' in json).toBe(false)
      expect('node' in json).toBe(false)
      expect(json.name).toBe('test_input')
      expect(json.type).toBe('TEST')
    })

    it('NodeOutputSlot.toJSON() should not include _node reference', () => {
      const graph = new LGraph()
      const node = LiteGraph.createNode('test/TestNode')!
      graph.add(node)
      node.addOutput('test_output', 'TEST')

      const outputSlot = node.outputs[0] as NodeOutputSlot
      const json = outputSlot.toJSON()

      expect('_node' in json).toBe(false)
      expect('node' in json).toBe(false)
      expect(json.name).toBe('test_output')
      expect(json.type).toBe('TEST')
    })
  })

  describe('Full graph with nodes - no circular references', () => {
    it('should serialize a graph with connected nodes', () => {
      const graph = new LGraph()

      const node1 = LiteGraph.createNode('test/TestNode')!
      const node2 = LiteGraph.createNode('test/TestNode')!
      graph.add(node1)
      graph.add(node2)

      node1.addOutput('out', 'TEST')
      node2.addInput('in', 'TEST')

      node1.connect(0, node2, 0)

      expect(() => JSON.stringify(graph)).not.toThrow()
    })

    it('should serialize graph.serialize() output without errors', () => {
      const graph = new LGraph()

      const node1 = LiteGraph.createNode('test/TestNode')!
      const node2 = LiteGraph.createNode('test/TestNode')!
      graph.add(node1)
      graph.add(node2)

      node1.addOutput('out', 'TEST')
      node2.addInput('in', 'TEST')

      node1.connect(0, node2, 0)

      const serialized = graph.serialize()

      expect(() => JSON.stringify(serialized)).not.toThrow()
      expect(() => JSON.parse(JSON.stringify(serialized))).not.toThrow()
    })
  })
})
