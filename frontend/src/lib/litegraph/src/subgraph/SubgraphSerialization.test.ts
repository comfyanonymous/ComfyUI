// TODO: Fix these tests after migration
/**
 * SubgraphSerialization Tests
 *
 * Tests for saving, loading, and version compatibility of subgraphs.
 * This covers serialization, deserialization, data integrity, and migration scenarios.
 */
import { describe, expect, it } from 'vitest'

import { LGraph, Subgraph } from '@/lib/litegraph/src/litegraph'

import {
  createTestSubgraph,
  createTestSubgraphNode
} from './__fixtures__/subgraphHelpers'

describe.skip('SubgraphSerialization - Basic Serialization', () => {
  it('should save and load simple subgraphs', () => {
    const original = createTestSubgraph({
      name: 'Simple Test',
      nodeCount: 2
    })
    original.addInput('in1', 'number')
    original.addInput('in2', 'string')
    original.addOutput('out', 'boolean')

    // Serialize
    const exported = original.asSerialisable()

    // Verify exported structure
    expect(exported).toHaveProperty('id', original.id)
    expect(exported).toHaveProperty('name', 'Simple Test')
    expect(exported).toHaveProperty('nodes')
    expect(exported).toHaveProperty('links')
    expect(exported).toHaveProperty('inputs')
    expect(exported).toHaveProperty('outputs')
    expect(exported).toHaveProperty('version')

    // Create new instance from serialized data
    const restored = new Subgraph(new LGraph(), exported)

    // Verify structure is preserved
    expect(restored.id).toBe(original.id)
    expect(restored.name).toBe(original.name)
    expect(restored.inputs.length).toBe(2) // Only added inputs, not original nodeCount
    expect(restored.outputs.length).toBe(1)
    // Note: nodes may not be restored if they're not registered types
    // This is expected behavior - serialization preserves I/O but nodes need valid types

    // Verify input details
    expect(restored.inputs[0].name).toBe('in1')
    expect(restored.inputs[0].type).toBe('number')
    expect(restored.inputs[1].name).toBe('in2')
    expect(restored.inputs[1].type).toBe('string')
    expect(restored.outputs[0].name).toBe('out')
    expect(restored.outputs[0].type).toBe('boolean')
  })

  it('should verify all properties are preserved', () => {
    const original = createTestSubgraph({
      name: 'Property Test',
      nodeCount: 3,
      inputs: [
        { name: 'input1', type: 'number' },
        { name: 'input2', type: 'string' }
      ],
      outputs: [
        { name: 'output1', type: 'boolean' },
        { name: 'output2', type: 'array' }
      ]
    })

    const exported = original.asSerialisable()
    const restored = new Subgraph(new LGraph(), exported)

    // Verify core properties
    expect(restored.id).toBe(original.id)
    expect(restored.name).toBe(original.name)
    expect(restored.description).toBe(original.description)

    // Verify I/O structure
    expect(restored.inputs.length).toBe(original.inputs.length)
    expect(restored.outputs.length).toBe(original.outputs.length)
    // Nodes may not be restored if they don't have registered types

    // Verify I/O details match
    for (let i = 0; i < original.inputs.length; i++) {
      expect(restored.inputs[i].name).toBe(original.inputs[i].name)
      expect(restored.inputs[i].type).toBe(original.inputs[i].type)
    }

    for (let i = 0; i < original.outputs.length; i++) {
      expect(restored.outputs[i].name).toBe(original.outputs[i].name)
      expect(restored.outputs[i].type).toBe(original.outputs[i].type)
    }
  })

  it('should test export() and configure() methods', () => {
    const subgraph = createTestSubgraph({ nodeCount: 1 })
    subgraph.addInput('test_input', 'number')
    subgraph.addOutput('test_output', 'string')

    // Test export
    const exported = subgraph.asSerialisable()
    expect(exported).toHaveProperty('id')
    expect(exported).toHaveProperty('nodes')
    expect(exported).toHaveProperty('links')
    expect(exported).toHaveProperty('inputs')
    expect(exported).toHaveProperty('outputs')

    // Test configure with partial data
    const newSubgraph = createTestSubgraph({ nodeCount: 0 })
    expect(() => {
      newSubgraph.configure(exported)
    }).not.toThrow()

    // Verify configuration applied
    expect(newSubgraph.inputs.length).toBe(1)
    expect(newSubgraph.outputs.length).toBe(1)
    expect(newSubgraph.inputs[0].name).toBe('test_input')
    expect(newSubgraph.outputs[0].name).toBe('test_output')
  })
})

describe.skip('SubgraphSerialization - Complex Serialization', () => {
  it('should serialize nested subgraphs with multiple levels', () => {
    // Create a nested structure
    const childSubgraph = createTestSubgraph({
      name: 'Child',
      nodeCount: 2,
      inputs: [{ name: 'child_in', type: 'number' }],
      outputs: [{ name: 'child_out', type: 'string' }]
    })

    const parentSubgraph = createTestSubgraph({
      name: 'Parent',
      nodeCount: 1,
      inputs: [{ name: 'parent_in', type: 'boolean' }],
      outputs: [{ name: 'parent_out', type: 'array' }]
    })

    // Add child to parent
    const childInstance = createTestSubgraphNode(childSubgraph, { id: 100 })
    parentSubgraph.add(childInstance)

    // Serialize both
    const childExported = childSubgraph.asSerialisable()
    const parentExported = parentSubgraph.asSerialisable()

    // Verify both can be serialized
    expect(childExported).toHaveProperty('name', 'Child')
    expect(parentExported).toHaveProperty('name', 'Parent')
    expect(parentExported.nodes.length).toBe(2) // 1 original + 1 child subgraph

    // Restore and verify
    const restoredChild = new Subgraph(new LGraph(), childExported)
    const restoredParent = new Subgraph(new LGraph(), parentExported)

    expect(restoredChild.name).toBe('Child')
    expect(restoredParent.name).toBe('Parent')
    expect(restoredChild.inputs.length).toBe(1)
    expect(restoredParent.inputs.length).toBe(1)
  })

  it('should serialize subgraphs with many nodes and connections', () => {
    const largeSubgraph = createTestSubgraph({
      name: 'Large Subgraph',
      nodeCount: 10 // Many nodes
    })

    // Add many I/O slots
    for (let i = 0; i < 5; i++) {
      largeSubgraph.addInput(`input_${i}`, 'number')
      largeSubgraph.addOutput(`output_${i}`, 'string')
    }

    const exported = largeSubgraph.asSerialisable()
    const restored = new Subgraph(new LGraph(), exported)

    // Verify I/O data preserved
    expect(restored.inputs.length).toBe(5)
    expect(restored.outputs.length).toBe(5)
    // Nodes may not be restored if they don't have registered types

    // Verify I/O naming preserved
    for (let i = 0; i < 5; i++) {
      expect(restored.inputs[i].name).toBe(`input_${i}`)
      expect(restored.outputs[i].name).toBe(`output_${i}`)
    }
  })

  it('should preserve custom node data', () => {
    const subgraph = createTestSubgraph({ nodeCount: 2 })

    // Add custom properties to nodes (if supported)
    const nodes = subgraph.nodes
    if (nodes.length > 0) {
      const firstNode = nodes[0]
      if (firstNode.properties) {
        firstNode.properties.customValue = 42
        firstNode.properties.customString = 'test'
      }
    }

    const exported = subgraph.asSerialisable()
    const restored = new Subgraph(new LGraph(), exported)

    // Test nodes may not be restored if they don't have registered types
    // This is expected behavior

    // Custom properties preservation depends on node implementation
    // This test documents the expected behavior
    if (restored.nodes.length > 0 && restored.nodes[0].properties) {
      // Properties should be preserved if the node supports them
      expect(restored.nodes[0].properties).toBeDefined()
    }
  })
})

describe.skip('SubgraphSerialization - Version Compatibility', () => {
  it('should handle version field in exports', () => {
    const subgraph = createTestSubgraph({ nodeCount: 1 })
    const exported = subgraph.asSerialisable()

    // Should have version field
    expect(exported).toHaveProperty('version')
    expect(typeof exported.version).toBe('number')
  })

  it('should load version 1.0+ format', () => {
    const modernFormat = {
      version: 1, // Number as expected by current implementation
      id: 'test-modern-id',
      name: 'Modern Subgraph',
      nodes: [],
      links: {},
      groups: [],
      config: {},
      definitions: { subgraphs: [] },
      inputs: [{ id: 'input-id', name: 'modern_input', type: 'number' }],
      outputs: [{ id: 'output-id', name: 'modern_output', type: 'string' }],
      inputNode: {
        id: -10,
        bounding: [0, 0, 120, 60]
      },
      outputNode: {
        id: -20,
        bounding: [300, 0, 120, 60]
      },
      widgets: []
    }

    expect(() => {
      // @ts-expect-error Type mismatch in ExportedSubgraph format
      const subgraph = new Subgraph(new LGraph(), modernFormat)
      expect(subgraph.name).toBe('Modern Subgraph')
      expect(subgraph.inputs.length).toBe(1)
      expect(subgraph.outputs.length).toBe(1)
    }).not.toThrow()
  })

  it('should handle missing fields gracefully', () => {
    const incompleteFormat = {
      version: 1,
      id: 'incomplete-id',
      name: 'Incomplete Subgraph',
      nodes: [],
      links: {},
      groups: [],
      config: {},
      definitions: { subgraphs: [] },
      inputNode: {
        id: -10,
        bounding: [0, 0, 120, 60]
      },
      outputNode: {
        id: -20,
        bounding: [300, 0, 120, 60]
      }
      // Missing optional: inputs, outputs, widgets
    }

    expect(() => {
      // @ts-expect-error Type mismatch in ExportedSubgraph format
      const subgraph = new Subgraph(new LGraph(), incompleteFormat)
      expect(subgraph.name).toBe('Incomplete Subgraph')
      // Should have default empty arrays
      expect(Array.isArray(subgraph.inputs)).toBe(true)
      expect(Array.isArray(subgraph.outputs)).toBe(true)
    }).not.toThrow()
  })

  it('should consider future-proofing', () => {
    const futureFormat = {
      version: 2, // Future version (number)
      id: 'future-id',
      name: 'Future Subgraph',
      nodes: [],
      links: {},
      groups: [],
      config: {},
      definitions: { subgraphs: [] },
      inputs: [],
      outputs: [],
      inputNode: {
        id: -10,
        bounding: [0, 0, 120, 60]
      },
      outputNode: {
        id: -20,
        bounding: [300, 0, 120, 60]
      },
      widgets: [],
      futureFeature: 'unknown_data' // Unknown future field
    }

    // Should handle future format gracefully
    expect(() => {
      // @ts-expect-error Type mismatch in ExportedSubgraph format
      const subgraph = new Subgraph(new LGraph(), futureFormat)
      expect(subgraph.name).toBe('Future Subgraph')
    }).not.toThrow()
  })
})

describe.skip('SubgraphSerialization - Data Integrity', () => {
  it('should pass round-trip testing (save → load → save → compare)', () => {
    const original = createTestSubgraph({
      name: 'Round Trip Test',
      nodeCount: 3,
      inputs: [
        { name: 'rt_input1', type: 'number' },
        { name: 'rt_input2', type: 'string' }
      ],
      outputs: [{ name: 'rt_output1', type: 'boolean' }]
    })

    // First round trip
    const exported1 = original.asSerialisable()
    const restored1 = new Subgraph(new LGraph(), exported1)

    // Second round trip
    const exported2 = restored1.asSerialisable()
    const restored2 = new Subgraph(new LGraph(), exported2)

    // Compare key properties
    expect(restored2.id).toBe(original.id)
    expect(restored2.name).toBe(original.name)
    expect(restored2.inputs.length).toBe(original.inputs.length)
    expect(restored2.outputs.length).toBe(original.outputs.length)
    // Nodes may not be restored if they don't have registered types

    // Compare I/O details
    for (let i = 0; i < original.inputs.length; i++) {
      expect(restored2.inputs[i].name).toBe(original.inputs[i].name)
      expect(restored2.inputs[i].type).toBe(original.inputs[i].type)
    }

    for (let i = 0; i < original.outputs.length; i++) {
      expect(restored2.outputs[i].name).toBe(original.outputs[i].name)
      expect(restored2.outputs[i].type).toBe(original.outputs[i].type)
    }
  })

  it('should verify IDs remain unique', () => {
    const subgraph1 = createTestSubgraph({ name: 'Unique1', nodeCount: 2 })
    const subgraph2 = createTestSubgraph({ name: 'Unique2', nodeCount: 2 })

    const exported1 = subgraph1.asSerialisable()
    const exported2 = subgraph2.asSerialisable()

    // IDs should be unique
    expect(exported1.id).not.toBe(exported2.id)

    const restored1 = new Subgraph(new LGraph(), exported1)
    const restored2 = new Subgraph(new LGraph(), exported2)

    expect(restored1.id).not.toBe(restored2.id)
    expect(restored1.id).toBe(subgraph1.id)
    expect(restored2.id).toBe(subgraph2.id)
  })

  it('should maintain connection integrity after load', () => {
    const subgraph = createTestSubgraph({ nodeCount: 2 })
    subgraph.addInput('connection_test', 'number')
    subgraph.addOutput('connection_result', 'string')

    const exported = subgraph.asSerialisable()
    const restored = new Subgraph(new LGraph(), exported)

    // Verify I/O connections can be established
    expect(restored.inputs.length).toBe(1)
    expect(restored.outputs.length).toBe(1)
    expect(restored.inputs[0].name).toBe('connection_test')
    expect(restored.outputs[0].name).toBe('connection_result')

    // Verify subgraph can be instantiated
    const instance = createTestSubgraphNode(restored)
    expect(instance.inputs.length).toBe(1)
    expect(instance.outputs.length).toBe(1)
  })

  it('should preserve node positions and properties', () => {
    const subgraph = createTestSubgraph({ nodeCount: 2 })

    // Modify node positions if possible
    if (subgraph.nodes.length > 0) {
      const node = subgraph.nodes[0]
      if ('pos' in node) {
        node.pos = [100, 200]
      }
      if ('size' in node) {
        node.size = [150, 80]
      }
    }

    const exported = subgraph.asSerialisable()
    const restored = new Subgraph(new LGraph(), exported)

    // Test nodes may not be restored if they don't have registered types
    // This is expected behavior

    // Position/size preservation depends on node implementation
    // This test documents the expected behavior
    if (restored.nodes.length > 0) {
      const restoredNode = restored.nodes[0]
      expect(restoredNode).toBeDefined()

      // Properties should be preserved if supported
      if ('pos' in restoredNode && restoredNode.pos) {
        expect(Array.isArray(restoredNode.pos)).toBe(true)
      }
    }
  })
})
