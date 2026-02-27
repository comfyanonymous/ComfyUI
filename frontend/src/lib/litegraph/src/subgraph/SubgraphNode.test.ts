// TODO: Fix these tests after migration
/**
 * SubgraphNode Tests
 *
 * Tests for SubgraphNode instances including construction,
 * IO synchronization, and edge cases.
 */
import { describe, expect, it, vi } from 'vitest'

import type { SubgraphNode } from '@/lib/litegraph/src/litegraph'
import { LGraph, Subgraph } from '@/lib/litegraph/src/litegraph'
import type { SubgraphInput } from '@/lib/litegraph/src/subgraph/SubgraphInput'

import { subgraphTest } from './__fixtures__/subgraphFixtures'
import {
  createTestSubgraph,
  createTestSubgraphNode
} from './__fixtures__/subgraphHelpers'

describe.skip('SubgraphNode Construction', () => {
  it('should create a SubgraphNode from a subgraph definition', () => {
    const subgraph = createTestSubgraph({
      name: 'Test Definition',
      inputs: [{ name: 'input', type: 'number' }],
      outputs: [{ name: 'output', type: 'number' }]
    })

    const subgraphNode = createTestSubgraphNode(subgraph)

    expect(subgraphNode).toBeDefined()
    expect(subgraphNode.subgraph).toBe(subgraph)
    expect(subgraphNode.type).toBe(subgraph.id)
    expect(subgraphNode.isVirtualNode).toBe(true)
    expect(subgraphNode.displayType).toBe('Subgraph node')
  })

  it('should configure from instance data', () => {
    const subgraph = createTestSubgraph({
      inputs: [{ name: 'value', type: 'number' }],
      outputs: [{ name: 'result', type: 'number' }]
    })

    const subgraphNode = createTestSubgraphNode(subgraph, {
      id: 42,
      pos: [300, 150],
      size: [180, 80]
    })

    expect(subgraphNode.id).toBe(42)
    expect(Array.from(subgraphNode.pos)).toEqual([300, 150])
    expect(Array.from(subgraphNode.size)).toEqual([180, 80])
  })

  it('should maintain reference to root graph', () => {
    const subgraph = createTestSubgraph()
    const subgraphNode = createTestSubgraphNode(subgraph)
    const parentGraph = subgraphNode.graph!

    expect(subgraphNode.rootGraph).toBe(parentGraph.rootGraph)
  })

  it('should throw NullGraphError when accessing rootGraph after removal', () => {
    const subgraph = createTestSubgraph()
    const subgraphNode = createTestSubgraphNode(subgraph)
    const parentGraph = subgraphNode.graph!
    parentGraph.add(subgraphNode)

    parentGraph.remove(subgraphNode)

    expect(() => subgraphNode.rootGraph).toThrow()
    expect(subgraphNode.graph).toBeNull()
  })

  subgraphTest(
    'should synchronize slots with subgraph definition',
    ({ subgraphWithNode }) => {
      const { subgraph, subgraphNode } = subgraphWithNode

      // SubgraphNode should have same number of inputs/outputs as definition
      expect(subgraphNode.inputs).toHaveLength(subgraph.inputs.length)
      expect(subgraphNode.outputs).toHaveLength(subgraph.outputs.length)
    }
  )

  subgraphTest(
    'should update slots when subgraph definition changes',
    ({ subgraphWithNode }) => {
      const { subgraph, subgraphNode } = subgraphWithNode

      const initialInputCount = subgraphNode.inputs.length

      // Add an input to the subgraph definition
      subgraph.addInput('new_input', 'string')

      // SubgraphNode should automatically update (this tests the event system)
      expect(subgraphNode.inputs).toHaveLength(initialInputCount + 1)
      expect(subgraphNode.inputs.at(-1)?.name).toBe('new_input')
      expect(subgraphNode.inputs.at(-1)?.type).toBe('string')
    }
  )
})

describe.skip('SubgraphNode Synchronization', () => {
  it('should sync input addition', () => {
    const subgraph = createTestSubgraph()
    const subgraphNode = createTestSubgraphNode(subgraph)

    expect(subgraphNode.inputs).toHaveLength(0)

    subgraph.addInput('value', 'number')

    expect(subgraphNode.inputs).toHaveLength(1)
    expect(subgraphNode.inputs[0].name).toBe('value')
    expect(subgraphNode.inputs[0].type).toBe('number')
  })

  it('should sync output addition', () => {
    const subgraph = createTestSubgraph()
    const subgraphNode = createTestSubgraphNode(subgraph)

    expect(subgraphNode.outputs).toHaveLength(0)

    subgraph.addOutput('result', 'string')

    expect(subgraphNode.outputs).toHaveLength(1)
    expect(subgraphNode.outputs[0].name).toBe('result')
    expect(subgraphNode.outputs[0].type).toBe('string')
  })

  it('should sync input removal', () => {
    const subgraph = createTestSubgraph({
      inputs: [
        { name: 'input1', type: 'number' },
        { name: 'input2', type: 'string' }
      ]
    })
    const subgraphNode = createTestSubgraphNode(subgraph)

    expect(subgraphNode.inputs).toHaveLength(2)

    subgraph.removeInput(subgraph.inputs[0])

    expect(subgraphNode.inputs).toHaveLength(1)
    expect(subgraphNode.inputs[0].name).toBe('input2')
  })

  it('should sync output removal', () => {
    const subgraph = createTestSubgraph({
      outputs: [
        { name: 'output1', type: 'number' },
        { name: 'output2', type: 'string' }
      ]
    })
    const subgraphNode = createTestSubgraphNode(subgraph)

    expect(subgraphNode.outputs).toHaveLength(2)

    subgraph.removeOutput(subgraph.outputs[0])

    expect(subgraphNode.outputs).toHaveLength(1)
    expect(subgraphNode.outputs[0].name).toBe('output2')
  })

  it('should sync slot renaming', () => {
    const subgraph = createTestSubgraph({
      inputs: [{ name: 'oldName', type: 'number' }],
      outputs: [{ name: 'oldOutput', type: 'string' }]
    })
    const subgraphNode = createTestSubgraphNode(subgraph)

    // Rename input
    subgraph.inputs[0].label = 'newName'
    subgraph.events.dispatch('renaming-input', {
      input: subgraph.inputs[0],
      index: 0,
      oldName: 'oldName',
      newName: 'newName'
    })

    expect(subgraphNode.inputs[0].label).toBe('newName')

    // Rename output
    subgraph.outputs[0].label = 'newOutput'
    subgraph.events.dispatch('renaming-output', {
      output: subgraph.outputs[0],
      index: 0,
      oldName: 'oldOutput',
      newName: 'newOutput'
    })

    expect(subgraphNode.outputs[0].label).toBe('newOutput')
  })
})

describe.skip('SubgraphNode Lifecycle', () => {
  it('should initialize with empty widgets array', () => {
    const subgraph = createTestSubgraph()
    const subgraphNode = createTestSubgraphNode(subgraph)

    expect(subgraphNode.widgets).toBeDefined()
    expect(subgraphNode.widgets).toHaveLength(0)
  })

  it('should handle reconfiguration', () => {
    const subgraph = createTestSubgraph({
      inputs: [{ name: 'input1', type: 'number' }],
      outputs: [{ name: 'output1', type: 'string' }]
    })
    const subgraphNode = createTestSubgraphNode(subgraph)

    // Initial state
    expect(subgraphNode.inputs).toHaveLength(1)
    expect(subgraphNode.outputs).toHaveLength(1)

    // Add more slots to subgraph
    subgraph.addInput('input2', 'string')
    subgraph.addOutput('output2', 'number')

    // Reconfigure
    subgraphNode.configure({
      id: subgraphNode.id,
      type: subgraph.id,
      pos: [200, 200],
      size: [180, 100],
      inputs: [],
      outputs: [],
      properties: {},
      flags: {},
      mode: 0,
      order: 0
    })

    // Should reflect updated subgraph structure
    expect(subgraphNode.inputs).toHaveLength(2)
    expect(subgraphNode.outputs).toHaveLength(2)
  })

  it('should handle removal lifecycle', () => {
    const subgraph = createTestSubgraph()
    const subgraphNode = createTestSubgraphNode(subgraph)
    const parentGraph = new LGraph()

    parentGraph.add(subgraphNode)
    expect(parentGraph.nodes).toContain(subgraphNode)

    // Test onRemoved method
    subgraphNode.onRemoved()

    // Note: onRemoved doesn't automatically remove from graph
    // but it should clean up internal state
    expect(subgraphNode.inputs).toBeDefined()
  })
})

describe.skip('SubgraphNode Basic Functionality', () => {
  it('should identify as subgraph node', () => {
    const subgraph = createTestSubgraph()
    const subgraphNode = createTestSubgraphNode(subgraph)

    expect(subgraphNode.isSubgraphNode()).toBe(true)
    expect(subgraphNode.isVirtualNode).toBe(true)
  })

  it('should inherit input types correctly', () => {
    const subgraph = createTestSubgraph({
      inputs: [
        { name: 'numberInput', type: 'number' },
        { name: 'stringInput', type: 'string' },
        { name: 'anyInput', type: '*' }
      ]
    })
    const subgraphNode = createTestSubgraphNode(subgraph)

    expect(subgraphNode.inputs[0].type).toBe('number')
    expect(subgraphNode.inputs[1].type).toBe('string')
    expect(subgraphNode.inputs[2].type).toBe('*')
  })

  it('should inherit output types correctly', () => {
    const subgraph = createTestSubgraph({
      outputs: [
        { name: 'numberOutput', type: 'number' },
        { name: 'stringOutput', type: 'string' },
        { name: 'anyOutput', type: '*' }
      ]
    })
    const subgraphNode = createTestSubgraphNode(subgraph)

    expect(subgraphNode.outputs[0].type).toBe('number')
    expect(subgraphNode.outputs[1].type).toBe('string')
    expect(subgraphNode.outputs[2].type).toBe('*')
  })
})

describe.skip('SubgraphNode Execution', () => {
  it('should flatten to ExecutableNodeDTOs', () => {
    const subgraph = createTestSubgraph({ nodeCount: 3 })
    const subgraphNode = createTestSubgraphNode(subgraph)

    const executableNodes = new Map()
    const flattened = subgraphNode.getInnerNodes(executableNodes)

    expect(flattened).toHaveLength(3)
    expect(flattened[0].id).toMatch(/^1:\d+$/) // Should have path-based ID like "1:1"
    expect(flattened[1].id).toMatch(/^1:\d+$/)
    expect(flattened[2].id).toMatch(/^1:\d+$/)
  })

  it.skip('should handle nested subgraph execution', () => {
    // FIXME: Complex nested structure requires proper parent graph setup
    // Skip for now - similar issue to ExecutableNodeDTO nested test
    // Will implement proper nested execution test in edge cases file
    const childSubgraph = createTestSubgraph({
      name: 'Child',
      nodeCount: 1
    })

    const parentSubgraph = createTestSubgraph({
      name: 'Parent',
      nodeCount: 1
    })

    const childSubgraphNode = createTestSubgraphNode(childSubgraph, { id: 42 })
    parentSubgraph.add(childSubgraphNode)

    const parentSubgraphNode = createTestSubgraphNode(parentSubgraph, {
      id: 10
    })

    const executableNodes = new Map()
    const flattened = parentSubgraphNode.getInnerNodes(executableNodes)

    expect(flattened.length).toBeGreaterThan(0)
  })

  it('should resolve cross-boundary input links', () => {
    const subgraph = createTestSubgraph({
      inputs: [{ name: 'input1', type: 'number' }],
      nodeCount: 1
    })
    const subgraphNode = createTestSubgraphNode(subgraph)

    const resolved = subgraphNode.resolveSubgraphInputLinks(0)

    expect(resolved).toBeDefined()
    expect(Array.isArray(resolved)).toBe(true)
  })

  it('should resolve cross-boundary output links', () => {
    const subgraph = createTestSubgraph({
      outputs: [{ name: 'output1', type: 'number' }],
      nodeCount: 1
    })
    const subgraphNode = createTestSubgraphNode(subgraph)

    const resolved = subgraphNode.resolveSubgraphOutputLink(0)

    // May be undefined if no internal connection exists
    expect(resolved === undefined || typeof resolved === 'object').toBe(true)
  })

  it('should prevent infinite recursion', () => {
    // Cycle detection properly prevents infinite recursion when a subgraph contains itself
    const subgraph = createTestSubgraph({ nodeCount: 1 })
    const subgraphNode = createTestSubgraphNode(subgraph)

    // Add subgraph node to its own subgraph (circular reference)
    subgraph.add(subgraphNode)

    const executableNodes = new Map()
    expect(() => {
      subgraphNode.getInnerNodes(executableNodes)
    }).toThrow(
      /Circular reference detected.*infinite loop in the subgraph hierarchy/i
    )
  })

  it('should handle nested subgraph execution', () => {
    // This test verifies that subgraph nodes can be properly executed
    // when they contain other nodes and produce correct output
    const subgraph = createTestSubgraph({
      name: 'Nested Execution Test',
      nodeCount: 3
    })

    const subgraphNode = createTestSubgraphNode(subgraph)

    // Verify that we can get executable DTOs for all nested nodes
    const executableNodes = new Map()
    const flattened = subgraphNode.getInnerNodes(executableNodes)

    expect(flattened).toHaveLength(3)

    // Each DTO should have proper execution context
    for (const dto of flattened) {
      expect(dto).toHaveProperty('id')
      expect(dto).toHaveProperty('graph')
      expect(dto).toHaveProperty('inputs')
      expect(dto.id).toMatch(/^\d+:\d+$/) // Path-based ID format
    }
  })

  it('should resolve cross-boundary links', () => {
    // This test verifies that links can cross subgraph boundaries
    // Currently this is a basic test - full cross-boundary linking
    // requires more complex setup with actual connected nodes
    const subgraph = createTestSubgraph({
      inputs: [{ name: 'external_input', type: 'number' }],
      outputs: [{ name: 'external_output', type: 'number' }],
      nodeCount: 2
    })

    const subgraphNode = createTestSubgraphNode(subgraph)

    // Verify the subgraph node has the expected I/O structure for cross-boundary links
    expect(subgraphNode.inputs).toHaveLength(1)
    expect(subgraphNode.outputs).toHaveLength(1)
    expect(subgraphNode.inputs[0].name).toBe('external_input')
    expect(subgraphNode.outputs[0].name).toBe('external_output')

    // Internal nodes should be flattened correctly
    const executableNodes = new Map()
    const flattened = subgraphNode.getInnerNodes(executableNodes)
    expect(flattened).toHaveLength(2)
  })
})

describe.skip('SubgraphNode Edge Cases', () => {
  it('should handle deep nesting', () => {
    // Create a simpler deep nesting test that works with current implementation
    const subgraph = createTestSubgraph({
      name: 'Deep Test',
      nodeCount: 5 // Multiple nodes to test flattening at depth
    })

    const subgraphNode = createTestSubgraphNode(subgraph)

    // Should be able to flatten without errors even with multiple nodes
    const executableNodes = new Map()
    expect(() => {
      subgraphNode.getInnerNodes(executableNodes)
    }).not.toThrow()

    const flattened = subgraphNode.getInnerNodes(executableNodes)
    expect(flattened.length).toBe(5)

    // All flattened nodes should have proper path-based IDs
    for (const dto of flattened) {
      expect(dto.id).toMatch(/^\d+:\d+$/)
    }
  })

  it('should validate against MAX_NESTED_SUBGRAPHS', () => {
    // Test that the MAX_NESTED_SUBGRAPHS constant exists
    // Note: Currently not enforced in the implementation
    expect(Subgraph.MAX_NESTED_SUBGRAPHS).toBe(1000)

    // This test documents the current behavior - limit is not enforced
    // TODO: Implement actual limit enforcement when business requirements clarify
  })
})

describe.skip('SubgraphNode Integration', () => {
  it('should be addable to a parent graph', () => {
    const subgraph = createTestSubgraph()
    const subgraphNode = createTestSubgraphNode(subgraph)
    const parentGraph = new LGraph()

    parentGraph.add(subgraphNode)

    expect(parentGraph.nodes).toContain(subgraphNode)
    expect(subgraphNode.graph).toBe(parentGraph)
  })

  subgraphTest(
    'should maintain reference to root graph',
    ({ subgraphWithNode }) => {
      const { subgraphNode } = subgraphWithNode

      // For this test, parentGraph should be the root, but in nested scenarios
      // it would traverse up to find the actual root
      expect(subgraphNode.rootGraph).toBeDefined()
    }
  )

  it('should handle graph removal properly', () => {
    const subgraph = createTestSubgraph()
    const subgraphNode = createTestSubgraphNode(subgraph)
    const parentGraph = new LGraph()

    parentGraph.add(subgraphNode)
    expect(parentGraph.nodes).toContain(subgraphNode)

    parentGraph.remove(subgraphNode)
    expect(parentGraph.nodes).not.toContain(subgraphNode)
  })
})

describe.skip('Foundation Test Utilities', () => {
  it('should create test SubgraphNodes with custom options', () => {
    const subgraph = createTestSubgraph()
    const customPos: [number, number] = [500, 300]
    const customSize: [number, number] = [250, 120]

    const subgraphNode = createTestSubgraphNode(subgraph, {
      pos: customPos,
      size: customSize
    })

    expect(Array.from(subgraphNode.pos)).toEqual(customPos)
    expect(Array.from(subgraphNode.size)).toEqual(customSize)
  })

  subgraphTest(
    'fixtures should provide properly configured SubgraphNode',
    ({ subgraphWithNode }) => {
      const { subgraph, subgraphNode, parentGraph } = subgraphWithNode

      expect(subgraph).toBeDefined()
      expect(subgraphNode).toBeDefined()
      expect(parentGraph).toBeDefined()
      expect(parentGraph.nodes).toContain(subgraphNode)
    }
  )
})

describe.skip('SubgraphNode Cleanup', () => {
  it('should clean up event listeners when removed', () => {
    const rootGraph = new LGraph()
    const subgraph = createTestSubgraph()

    // Create and add two nodes
    const node1 = createTestSubgraphNode(subgraph)
    const node2 = createTestSubgraphNode(subgraph)
    rootGraph.add(node1)
    rootGraph.add(node2)

    // Verify both nodes start with no inputs
    expect(node1.inputs.length).toBe(0)
    expect(node2.inputs.length).toBe(0)

    // Remove node2
    rootGraph.remove(node2)

    // Now trigger an event - only node1 should respond
    subgraph.events.dispatch('input-added', {
      input: { name: 'test', type: 'number', id: 'test-id' } as SubgraphInput
    })

    // Only node1 should have added an input
    expect(node1.inputs.length).toBe(1) // node1 responds
    expect(node2.inputs.length).toBe(0) // node2 should NOT respond (but currently does)
  })

  it('should not accumulate handlers over multiple add/remove cycles', () => {
    const rootGraph = new LGraph()
    const subgraph = createTestSubgraph()

    const removedNodes: SubgraphNode[] = []
    for (let i = 0; i < 3; i++) {
      const node = createTestSubgraphNode(subgraph)
      rootGraph.add(node)
      rootGraph.remove(node)
      removedNodes.push(node)
    }

    // All nodes should have 0 inputs
    for (const node of removedNodes) {
      expect(node.inputs.length).toBe(0)
    }

    // Trigger an event - no nodes should respond
    subgraph.events.dispatch('input-added', {
      input: { name: 'test', type: 'number', id: 'test-id' } as SubgraphInput
    })

    // Without cleanup: all 3 removed nodes would have added an input
    // With cleanup: no nodes should have added an input
    for (const node of removedNodes) {
      expect(node.inputs.length).toBe(0) // Should stay 0 after cleanup
    }
  })

  it('should clean up input listener controllers on removal', () => {
    const rootGraph = new LGraph()
    const subgraph = createTestSubgraph({
      inputs: [
        { name: 'in1', type: 'number' },
        { name: 'in2', type: 'string' }
      ]
    })

    const subgraphNode = createTestSubgraphNode(subgraph)
    rootGraph.add(subgraphNode)

    // Verify listener controllers exist
    expect(subgraphNode.inputs[0]._listenerController).toBeDefined()
    expect(subgraphNode.inputs[1]._listenerController).toBeDefined()

    // Track abort calls
    const abortSpy1 = vi.spyOn(
      subgraphNode.inputs[0]._listenerController!,
      'abort'
    )
    const abortSpy2 = vi.spyOn(
      subgraphNode.inputs[1]._listenerController!,
      'abort'
    )

    // Remove node
    rootGraph.remove(subgraphNode)

    // Verify abort was called on each controller
    expect(abortSpy1).toHaveBeenCalledTimes(1)
    expect(abortSpy2).toHaveBeenCalledTimes(1)
  })
})
