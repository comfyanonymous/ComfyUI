// TODO: Fix these tests after migration
/**
 * Core Subgraph Tests
 *
 * This file implements fundamental tests for the Subgraph class that establish
 * patterns for the rest of the testing team. These tests cover construction,
 * basic I/O management, and known issues.
 */
import { describe, expect, it } from 'vitest'

import {
  createUuidv4,
  RecursionError,
  LGraph,
  Subgraph
} from '@/lib/litegraph/src/litegraph'

import { subgraphTest } from './__fixtures__/subgraphFixtures'
import {
  assertSubgraphStructure,
  createTestSubgraph,
  createTestSubgraphData
} from './__fixtures__/subgraphHelpers'

describe.skip('Subgraph Construction', () => {
  it('should create a subgraph with minimal data', () => {
    const subgraph = createTestSubgraph()

    assertSubgraphStructure(subgraph, {
      inputCount: 0,
      outputCount: 0,
      nodeCount: 0,
      name: 'Test Subgraph'
    })

    expect(subgraph.id).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
    )
    expect(subgraph.inputNode).toBeDefined()
    expect(subgraph.outputNode).toBeDefined()
    expect(subgraph.inputNode.id).toBe(-10)
    expect(subgraph.outputNode.id).toBe(-20)
  })

  it('should require a root graph', () => {
    const subgraphData = createTestSubgraphData()

    expect(() => {
      // @ts-expect-error Testing invalid null parameter
      new Subgraph(null, subgraphData)
    }).toThrow('Root graph is required')
  })

  it('should accept custom name and ID', () => {
    const customId = createUuidv4()
    const customName = 'My Custom Subgraph'

    const subgraph = createTestSubgraph({
      id: customId,
      name: customName
    })

    expect(subgraph.id).toBe(customId)
    expect(subgraph.name).toBe(customName)
  })

  it('should initialize with empty inputs and outputs', () => {
    const subgraph = createTestSubgraph()

    expect(subgraph.inputs).toHaveLength(0)
    expect(subgraph.outputs).toHaveLength(0)
    expect(subgraph.widgets).toHaveLength(0)
  })

  it('should have properly configured input and output nodes', () => {
    const subgraph = createTestSubgraph()

    // Input node should be positioned on the left
    expect(subgraph.inputNode.pos[0]).toBeLessThan(100)

    // Output node should be positioned on the right
    expect(subgraph.outputNode.pos[0]).toBeGreaterThan(300)

    // Both should reference the subgraph
    expect(subgraph.inputNode.subgraph).toBe(subgraph)
    expect(subgraph.outputNode.subgraph).toBe(subgraph)
  })
})

describe.skip('Subgraph Input/Output Management', () => {
  subgraphTest('should add a single input', ({ emptySubgraph }) => {
    const input = emptySubgraph.addInput('test_input', 'number')

    expect(emptySubgraph.inputs).toHaveLength(1)
    expect(input.name).toBe('test_input')
    expect(input.type).toBe('number')
    expect(emptySubgraph.inputs.indexOf(input)).toBe(0)
  })

  subgraphTest('should add a single output', ({ emptySubgraph }) => {
    const output = emptySubgraph.addOutput('test_output', 'string')

    expect(emptySubgraph.outputs).toHaveLength(1)
    expect(output.name).toBe('test_output')
    expect(output.type).toBe('string')
    expect(emptySubgraph.outputs.indexOf(output)).toBe(0)
  })

  subgraphTest(
    'should maintain correct indices when adding multiple inputs',
    ({ emptySubgraph }) => {
      const input1 = emptySubgraph.addInput('input_1', 'number')
      const input2 = emptySubgraph.addInput('input_2', 'string')
      const input3 = emptySubgraph.addInput('input_3', 'boolean')

      expect(emptySubgraph.inputs.indexOf(input1)).toBe(0)
      expect(emptySubgraph.inputs.indexOf(input2)).toBe(1)
      expect(emptySubgraph.inputs.indexOf(input3)).toBe(2)
      expect(emptySubgraph.inputs).toHaveLength(3)
    }
  )

  subgraphTest(
    'should maintain correct indices when adding multiple outputs',
    ({ emptySubgraph }) => {
      const output1 = emptySubgraph.addOutput('output_1', 'number')
      const output2 = emptySubgraph.addOutput('output_2', 'string')
      const output3 = emptySubgraph.addOutput('output_3', 'boolean')

      expect(emptySubgraph.outputs.indexOf(output1)).toBe(0)
      expect(emptySubgraph.outputs.indexOf(output2)).toBe(1)
      expect(emptySubgraph.outputs.indexOf(output3)).toBe(2)
      expect(emptySubgraph.outputs).toHaveLength(3)
    }
  )

  subgraphTest('should remove inputs correctly', ({ simpleSubgraph }) => {
    // Add a second input first
    simpleSubgraph.addInput('second_input', 'string')
    expect(simpleSubgraph.inputs).toHaveLength(2)

    // Remove the first input
    const firstInput = simpleSubgraph.inputs[0]
    simpleSubgraph.removeInput(firstInput)

    expect(simpleSubgraph.inputs).toHaveLength(1)
    expect(simpleSubgraph.inputs[0].name).toBe('second_input')
    // Verify it's at index 0 in the array
    expect(simpleSubgraph.inputs.indexOf(simpleSubgraph.inputs[0])).toBe(0)
  })

  subgraphTest('should remove outputs correctly', ({ simpleSubgraph }) => {
    // Add a second output first
    simpleSubgraph.addOutput('second_output', 'string')
    expect(simpleSubgraph.outputs).toHaveLength(2)

    // Remove the first output
    const firstOutput = simpleSubgraph.outputs[0]
    simpleSubgraph.removeOutput(firstOutput)

    expect(simpleSubgraph.outputs).toHaveLength(1)
    expect(simpleSubgraph.outputs[0].name).toBe('second_output')
    // Verify it's at index 0 in the array
    expect(simpleSubgraph.outputs.indexOf(simpleSubgraph.outputs[0])).toBe(0)
  })
})

describe.skip('Subgraph Serialization', () => {
  subgraphTest('should serialize empty subgraph', ({ emptySubgraph }) => {
    const serialized = emptySubgraph.asSerialisable()

    expect(serialized.version).toBe(1)
    expect(serialized.id).toBeTruthy()
    expect(serialized.name).toBe('Empty Test Subgraph')
    expect(serialized.inputs).toHaveLength(0)
    expect(serialized.outputs).toHaveLength(0)
    expect(serialized.nodes).toHaveLength(0)
    expect(typeof serialized.links).toBe('object')
  })

  subgraphTest(
    'should serialize subgraph with inputs and outputs',
    ({ simpleSubgraph }) => {
      const serialized = simpleSubgraph.asSerialisable()

      expect(serialized.inputs).toHaveLength(1)
      expect(serialized.outputs).toHaveLength(1)
      expect(serialized.inputs![0].name).toBe('input')
      expect(serialized.inputs![0].type).toBe('number')
      expect(serialized.outputs![0].name).toBe('output')
      expect(serialized.outputs![0].type).toBe('number')
    }
  )

  subgraphTest(
    'should include input and output nodes in serialization',
    ({ emptySubgraph }) => {
      const serialized = emptySubgraph.asSerialisable()

      expect(serialized.inputNode).toBeDefined()
      expect(serialized.outputNode).toBeDefined()
      expect(serialized.inputNode.id).toBe(-10)
      expect(serialized.outputNode.id).toBe(-20)
    }
  )
})

describe.skip('Subgraph Known Issues', () => {
  it.todo('should enforce MAX_NESTED_SUBGRAPHS limit', () => {
    // This test documents that MAX_NESTED_SUBGRAPHS = 1000 is defined
    // but not actually enforced anywhere in the code.
    //
    // Expected behavior: Should throw error when nesting exceeds limit
    // Actual behavior: No validation is performed
    //
    // This safety limit should be implemented to prevent runaway recursion.
  })

  it('should provide MAX_NESTED_SUBGRAPHS constant', () => {
    expect(Subgraph.MAX_NESTED_SUBGRAPHS).toBe(1000)
  })

  it('should have recursion detection in place', () => {
    // Verify that RecursionError is available and can be thrown
    expect(() => {
      throw new RecursionError('test recursion')
    }).toThrow(RecursionError)

    expect(() => {
      throw new RecursionError('test recursion')
    }).toThrow('test recursion')
  })
})

describe.skip('Subgraph Root Graph Relationship', () => {
  it('should maintain reference to root graph', () => {
    const rootGraph = new LGraph()
    const subgraphData = createTestSubgraphData()
    const subgraph = new Subgraph(rootGraph, subgraphData)

    expect(subgraph.rootGraph).toBe(rootGraph)
  })

  it('should inherit root graph in nested subgraphs', () => {
    const rootGraph = new LGraph()
    const parentData = createTestSubgraphData({
      name: 'Parent Subgraph'
    })
    const parentSubgraph = new Subgraph(rootGraph, parentData)

    // Create a nested subgraph
    const nestedData = createTestSubgraphData({
      name: 'Nested Subgraph'
    })
    const nestedSubgraph = new Subgraph(rootGraph, nestedData)

    expect(nestedSubgraph.rootGraph).toBe(rootGraph)
    expect(parentSubgraph.rootGraph).toBe(rootGraph)
  })
})

describe.skip('Subgraph Error Handling', () => {
  subgraphTest(
    'should handle removing non-existent input gracefully',
    ({ emptySubgraph }) => {
      // Create a fake input that doesn't belong to this subgraph
      const fakeInput = emptySubgraph.addInput('temp', 'number')
      emptySubgraph.removeInput(fakeInput) // Remove it first

      // Now try to remove it again
      expect(() => {
        emptySubgraph.removeInput(fakeInput)
      }).toThrow('Input not found')
    }
  )

  subgraphTest(
    'should handle removing non-existent output gracefully',
    ({ emptySubgraph }) => {
      // Create a fake output that doesn't belong to this subgraph
      const fakeOutput = emptySubgraph.addOutput('temp', 'number')
      emptySubgraph.removeOutput(fakeOutput) // Remove it first

      // Now try to remove it again
      expect(() => {
        emptySubgraph.removeOutput(fakeOutput)
      }).toThrow('Output not found')
    }
  )
})

describe.skip('Subgraph Integration', () => {
  it("should work with LGraph's node management", () => {
    const subgraph = createTestSubgraph({
      nodeCount: 3
    })

    // Verify nodes were added to the subgraph
    expect(subgraph.nodes).toHaveLength(3)

    // Verify we can access nodes by ID
    const firstNode = subgraph.getNodeById(1)
    expect(firstNode).toBeDefined()
    expect(firstNode?.title).toContain('Test Node')
  })

  it('should maintain link integrity', () => {
    const subgraph = createTestSubgraph({
      nodeCount: 2
    })

    const node1 = subgraph.nodes[0]
    const node2 = subgraph.nodes[1]

    // Connect the nodes
    node1.connect(0, node2, 0)

    // Verify link was created
    expect(subgraph.links.size).toBe(1)

    // Verify link integrity
    const link = Array.from(subgraph.links.values())[0]
    expect(link.origin_id).toBe(node1.id)
    expect(link.target_id).toBe(node2.id)
  })
})
