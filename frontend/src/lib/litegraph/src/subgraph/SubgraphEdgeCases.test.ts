// TODO: Fix these tests after migration
/**
 * SubgraphEdgeCases Tests
 *
 * Tests for edge cases, error handling, and boundary conditions in the subgraph system.
 * This covers unusual scenarios, invalid states, and stress testing.
 */
import { describe, expect, it } from 'vitest'

import { LGraph, LGraphNode, Subgraph } from '@/lib/litegraph/src/litegraph'

import {
  createNestedSubgraphs,
  createTestSubgraph,
  createTestSubgraphNode
} from './__fixtures__/subgraphHelpers'

describe.skip('SubgraphEdgeCases - Recursion Detection', () => {
  it('should handle circular subgraph references without crashing', () => {
    const sub1 = createTestSubgraph({ name: 'Sub1' })
    const sub2 = createTestSubgraph({ name: 'Sub2' })

    // Create circular reference
    const node1 = createTestSubgraphNode(sub1, { id: 1 })
    const node2 = createTestSubgraphNode(sub2, { id: 2 })

    sub1.add(node2)
    sub2.add(node1)

    // Should not crash or hang - currently throws path resolution error due to circular structure
    expect(() => {
      const executableNodes = new Map()
      node1.getInnerNodes(executableNodes)
    }).toThrow(/Node \[\d+\] not found/) // Current behavior: path resolution fails
  })

  it('should handle deep nesting scenarios', () => {
    // Test with reasonable depth to avoid timeout
    const nested = createNestedSubgraphs({ depth: 10, nodesPerLevel: 1 })

    // Should create nested structure without errors
    expect(nested.subgraphs).toHaveLength(10)
    expect(nested.subgraphNodes).toHaveLength(10)

    // First level should exist and be accessible
    const firstLevel = nested.rootGraph.nodes[0]
    expect(firstLevel).toBeDefined()
    expect(firstLevel.isSubgraphNode()).toBe(true)
  })

  it.todo('should use WeakSet for cycle detection', () => {
    // TODO: This test is currently skipped because cycle detection has a bug
    // The fix is to pass 'visited' directly instead of 'new Set(visited)' in SubgraphNode.ts:299
    const subgraph = createTestSubgraph({ nodeCount: 1 })
    const subgraphNode = createTestSubgraphNode(subgraph)

    // Add to own subgraph to create cycle
    subgraph.add(subgraphNode)

    // Should throw due to cycle detection
    const executableNodes = new Map()
    expect(() => {
      subgraphNode.getInnerNodes(executableNodes)
    }).toThrow(/while flattening subgraph/i)
  })

  it('should respect MAX_NESTED_SUBGRAPHS constant', () => {
    // Verify the constant exists and is a reasonable positive number
    expect(Subgraph.MAX_NESTED_SUBGRAPHS).toBeDefined()
    expect(typeof Subgraph.MAX_NESTED_SUBGRAPHS).toBe('number')
    expect(Subgraph.MAX_NESTED_SUBGRAPHS).toBeGreaterThan(0)
    expect(Subgraph.MAX_NESTED_SUBGRAPHS).toBeLessThanOrEqual(10_000) // Reasonable upper bound

    // Note: Currently not enforced in implementation
    // This test documents the intended behavior
  })
})

describe.skip('SubgraphEdgeCases - Invalid States', () => {
  it('should handle removing non-existent inputs gracefully', () => {
    const subgraph = createTestSubgraph()
    const fakeInput = {
      name: 'fake',
      type: 'number',
      disconnect: () => {}
    } as Partial<Parameters<typeof subgraph.removeInput>[0]> as Parameters<
      typeof subgraph.removeInput
    >[0]

    // Should throw appropriate error for non-existent input
    expect(() => {
      subgraph.removeInput(fakeInput)
    }).toThrow(/Input not found/) // Expected error
  })

  it('should handle removing non-existent outputs gracefully', () => {
    const subgraph = createTestSubgraph()
    const fakeOutput = {
      name: 'fake',
      type: 'number',
      disconnect: () => {}
    } as Partial<Parameters<typeof subgraph.removeOutput>[0]> as Parameters<
      typeof subgraph.removeOutput
    >[0]

    expect(() => {
      subgraph.removeOutput(fakeOutput)
    }).toThrow(/Output not found/) // Expected error
  })

  it('should throw error for null/undefined input names', () => {
    const subgraph = createTestSubgraph()

    const nullString: string = null!
    const undefinedString: string = undefined!

    expect(() => {
      subgraph.addInput(nullString, 'number')
    }).toThrow()

    expect(() => {
      subgraph.addInput(undefinedString, 'number')
    }).toThrow()
  })

  it('should handle null/undefined output names', () => {
    const subgraph = createTestSubgraph()

    const nullString: string = null!
    const undefinedString: string = undefined!

    expect(() => {
      subgraph.addOutput(nullString, 'number')
    }).toThrow()

    expect(() => {
      subgraph.addOutput(undefinedString, 'number')
    }).toThrow()
  })

  it('should handle empty string names', () => {
    const subgraph = createTestSubgraph()

    // Current implementation may allow empty strings
    // Document the actual behavior
    expect(() => {
      subgraph.addInput('', 'number')
    }).not.toThrow() // Current behavior: allows empty strings

    expect(() => {
      subgraph.addOutput('', 'number')
    }).not.toThrow() // Current behavior: allows empty strings
  })

  it('should handle undefined types gracefully', () => {
    const subgraph = createTestSubgraph()

    const undefinedString: string = undefined!

    // Undefined type should throw error
    expect(() => {
      subgraph.addInput('test', undefinedString)
    }).toThrow()

    expect(() => {
      subgraph.addOutput('test', undefinedString)
    }).toThrow()
  })

  it('should handle duplicate slot names', () => {
    const subgraph = createTestSubgraph()

    // Add first input
    subgraph.addInput('duplicate', 'number')

    // Adding duplicate should not crash (current behavior allows it)
    expect(() => {
      subgraph.addInput('duplicate', 'string')
    }).not.toThrow()

    // Should now have 2 inputs with same name
    expect(subgraph.inputs.length).toBe(2)
    expect(subgraph.inputs[0].name).toBe('duplicate')
    expect(subgraph.inputs[1].name).toBe('duplicate')
  })
})

describe.skip('SubgraphEdgeCases - Boundary Conditions', () => {
  it('should handle empty subgraphs (no nodes, no IO)', () => {
    const subgraph = createTestSubgraph({ nodeCount: 0 })
    const subgraphNode = createTestSubgraphNode(subgraph)

    // Should handle empty subgraph without errors
    const executableNodes = new Map()
    const flattened = subgraphNode.getInnerNodes(executableNodes)

    expect(flattened).toHaveLength(0)
    expect(subgraph.inputs).toHaveLength(0)
    expect(subgraph.outputs).toHaveLength(0)
  })

  it('should handle single input/output subgraphs', () => {
    const subgraph = createTestSubgraph({
      inputs: [{ name: 'single_in', type: 'number' }],
      outputs: [{ name: 'single_out', type: 'number' }],
      nodeCount: 1
    })

    const subgraphNode = createTestSubgraphNode(subgraph)

    expect(subgraphNode.inputs).toHaveLength(1)
    expect(subgraphNode.outputs).toHaveLength(1)
    expect(subgraphNode.inputs[0].name).toBe('single_in')
    expect(subgraphNode.outputs[0].name).toBe('single_out')
  })

  it('should handle subgraphs with many slots', () => {
    const subgraph = createTestSubgraph({ nodeCount: 1 })

    // Add many inputs (test with 20 to keep test fast)
    for (let i = 0; i < 20; i++) {
      subgraph.addInput(`input_${i}`, 'number')
    }

    // Add many outputs
    for (let i = 0; i < 20; i++) {
      subgraph.addOutput(`output_${i}`, 'number')
    }

    const subgraphNode = createTestSubgraphNode(subgraph)

    expect(subgraph.inputs).toHaveLength(20)
    expect(subgraph.outputs).toHaveLength(20)
    expect(subgraphNode.inputs).toHaveLength(20)
    expect(subgraphNode.outputs).toHaveLength(20)

    // Should still flatten correctly
    const executableNodes = new Map()
    const flattened = subgraphNode.getInnerNodes(executableNodes)
    expect(flattened).toHaveLength(1) // Original node count
  })

  it('should handle very long slot names', () => {
    const subgraph = createTestSubgraph()
    const longName = 'a'.repeat(1000) // 1000 character name

    expect(() => {
      subgraph.addInput(longName, 'number')
      subgraph.addOutput(longName, 'string')
    }).not.toThrow()

    expect(subgraph.inputs[0].name).toBe(longName)
    expect(subgraph.outputs[0].name).toBe(longName)
  })

  it('should handle Unicode characters in names', () => {
    const subgraph = createTestSubgraph()
    const unicodeName = '测试_🚀_تست_тест'

    expect(() => {
      subgraph.addInput(unicodeName, 'number')
      subgraph.addOutput(unicodeName, 'string')
    }).not.toThrow()

    expect(subgraph.inputs[0].name).toBe(unicodeName)
    expect(subgraph.outputs[0].name).toBe(unicodeName)
  })
})

describe.skip('SubgraphEdgeCases - Type Validation', () => {
  it('should allow connecting mismatched types (no validation currently)', () => {
    const rootGraph = new LGraph()
    const subgraph = createTestSubgraph()

    subgraph.addInput('num', 'number')
    subgraph.addOutput('str', 'string')

    // Create a basic node manually since createNode is not available
    const numberNode = new LGraphNode('basic/const')
    numberNode.addOutput('value', 'number')
    rootGraph.add(numberNode)

    const subgraphNode = createTestSubgraphNode(subgraph)
    rootGraph.add(subgraphNode)

    // Currently allows mismatched connections (no type validation)
    expect(() => {
      numberNode.connect(0, subgraphNode, 0)
    }).not.toThrow()
  })

  it('should handle invalid type strings', () => {
    const subgraph = createTestSubgraph()

    // These should not crash (current behavior)
    expect(() => {
      subgraph.addInput('test1', 'invalid_type')
      subgraph.addInput('test2', '')
      subgraph.addInput('test3', '123')
      subgraph.addInput('test4', 'special!@#$%')
    }).not.toThrow()
  })

  it('should handle complex type strings', () => {
    const subgraph = createTestSubgraph()

    expect(() => {
      subgraph.addInput('array', 'array<number>')
      subgraph.addInput('object', 'object<{x: number, y: string}>')
      subgraph.addInput('union', 'number|string')
    }).not.toThrow()

    expect(subgraph.inputs).toHaveLength(3)
    expect(subgraph.inputs[0].type).toBe('array<number>')
    expect(subgraph.inputs[1].type).toBe('object<{x: number, y: string}>')
    expect(subgraph.inputs[2].type).toBe('number|string')
  })
})

describe.skip('SubgraphEdgeCases - Performance and Scale', () => {
  it('should handle large numbers of nodes in subgraph', () => {
    // Create subgraph with many nodes (keep reasonable for test speed)
    const subgraph = createTestSubgraph({ nodeCount: 50 })
    const subgraphNode = createTestSubgraphNode(subgraph)

    const executableNodes = new Map()
    const flattened = subgraphNode.getInnerNodes(executableNodes)

    expect(flattened).toHaveLength(50)

    // Performance is acceptable for 50 nodes (typically < 1ms)
  })

  it('should handle rapid IO changes', () => {
    const subgraph = createTestSubgraph()

    // Rapidly add and remove inputs/outputs
    for (let i = 0; i < 10; i++) {
      const input = subgraph.addInput(`rapid_${i}`, 'number')
      const output = subgraph.addOutput(`rapid_${i}`, 'number')

      // Remove them immediately
      subgraph.removeInput(input)
      subgraph.removeOutput(output)
    }

    // Should end up with no inputs/outputs
    expect(subgraph.inputs).toHaveLength(0)
    expect(subgraph.outputs).toHaveLength(0)
  })

  it('should handle concurrent modifications safely', () => {
    // This test ensures the system doesn't crash under concurrent access
    // Note: JavaScript is single-threaded, so this tests rapid sequential access
    const subgraph = createTestSubgraph({ nodeCount: 5 })
    const subgraphNode = createTestSubgraphNode(subgraph)

    // Simulate concurrent operations
    const operations: Array<() => void> = []
    for (let i = 0; i < 20; i++) {
      operations.push(
        () => {
          const executableNodes = new Map()
          subgraphNode.getInnerNodes(executableNodes)
        },
        () => {
          subgraph.addInput(`concurrent_${i}`, 'number')
        },
        () => {
          if (subgraph.inputs.length > 0) {
            subgraph.removeInput(subgraph.inputs[0])
          }
        }
      )
    }

    // Execute all operations - should not crash
    expect(() => {
      for (const op of operations) op()
    }).not.toThrow()
  })
})
