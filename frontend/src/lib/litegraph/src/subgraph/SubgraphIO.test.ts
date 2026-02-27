// TODO: Fix these tests after migration
import { describe, expect, it } from 'vitest'

import { LGraphNode } from '@/lib/litegraph/src/litegraph'
import { ToInputFromIoNodeLink } from '@/lib/litegraph/src/canvas/ToInputFromIoNodeLink'
import { LinkDirection } from '@/lib/litegraph/src//types/globalEnums'

import { subgraphTest } from './__fixtures__/subgraphFixtures'
import {
  createTestSubgraph,
  createTestSubgraphNode
} from './__fixtures__/subgraphHelpers'

describe('SubgraphIO - Input Slot Dual-Nature Behavior', () => {
  subgraphTest(
    'input accepts external connections from parent graph',
    ({ subgraphWithNode }) => {
      const { subgraph, subgraphNode, parentGraph } = subgraphWithNode

      subgraph.addInput('test_input', 'number')

      const externalNode = new LGraphNode('External Source')
      externalNode.addOutput('out', 'number')
      parentGraph.add(externalNode)

      expect(() => {
        externalNode.connect(0, subgraphNode, 0)
      }).not.toThrow()

      expect(
        externalNode.outputs[0].links?.includes(subgraphNode.inputs[0].link!)
      ).toBe(true)
      expect(subgraphNode.inputs[0].link).not.toBe(null)
    }
  )

  subgraphTest(
    'empty input slot creation enables dynamic IO',
    ({ simpleSubgraph }) => {
      const initialInputCount = simpleSubgraph.inputs.length

      // Create empty input slot
      simpleSubgraph.addInput('', '*')

      // Should create new input
      expect(simpleSubgraph.inputs.length).toBe(initialInputCount + 1)

      // The empty slot should be configurable
      const emptyInput = simpleSubgraph.inputs.at(-1)!
      expect(emptyInput.name).toBe('')
      expect(emptyInput.type).toBe('*')
    }
  )

  subgraphTest(
    'handles slot removal with active connections',
    ({ subgraphWithNode }) => {
      const { subgraph, subgraphNode, parentGraph } = subgraphWithNode

      const externalNode = new LGraphNode('External Source')
      externalNode.addOutput('out', '*')
      parentGraph.add(externalNode)

      externalNode.connect(0, subgraphNode, 0)

      // Verify connection exists
      expect(subgraphNode.inputs[0].link).not.toBe(null)

      // Remove the existing input (fixture creates one input)
      const inputToRemove = subgraph.inputs[0]
      subgraph.removeInput(inputToRemove)

      // Connection should be cleaned up
      expect(subgraphNode.inputs.length).toBe(0)
      expect(externalNode.outputs[0].links).toHaveLength(0)
    }
  )

  subgraphTest('handles link disconnection', ({ subgraphWithNode }) => {
    const { subgraph } = subgraphWithNode
    const internalNode = new LGraphNode('External Source')
    internalNode.addInput('in', '*')
    internalNode.onConnectionsChange = vi.fn()
    subgraph.add(internalNode)

    const link = subgraph.inputNode.slots[0].connect(
      internalNode.inputs[0],
      internalNode
    )
    new ToInputFromIoNodeLink(
      subgraph,
      subgraph.inputNode,
      subgraph.inputNode.slots[0],
      undefined,
      LinkDirection.CENTER,
      link
    ).disconnect()

    expect(internalNode.inputs[0].link).toBeNull()
    expect(subgraph.inputNode.slots[0].linkIds.length).toBe(0)
    expect(internalNode.onConnectionsChange).toHaveBeenCalled()
  })

  subgraphTest(
    'handles slot renaming with active connections',
    ({ subgraphWithNode }) => {
      const { subgraph, subgraphNode, parentGraph } = subgraphWithNode

      const externalNode = new LGraphNode('External Source')
      externalNode.addOutput('out', '*')
      parentGraph.add(externalNode)

      externalNode.connect(0, subgraphNode, 0)

      // Verify connection exists
      expect(subgraphNode.inputs[0].link).not.toBe(null)

      // Rename the existing input (fixture creates input named "input")
      const inputToRename = subgraph.inputs[0]
      subgraph.renameInput(inputToRename, 'new_name')

      // Connection should persist and subgraph definition should be updated
      expect(subgraphNode.inputs[0].link).not.toBe(null)
      expect(subgraph.inputs[0].label).toBe('new_name')
      expect(subgraph.inputs[0].displayName).toBe('new_name')
    }
  )
})

describe('SubgraphIO - Output Slot Dual-Nature Behavior', () => {
  subgraphTest(
    'output provides connections to parent graph',
    ({ subgraphWithNode }) => {
      const { subgraph, subgraphNode, parentGraph } = subgraphWithNode

      // Add an output to the subgraph
      subgraph.addOutput('test_output', 'number')

      const externalNode = new LGraphNode('External Target')
      externalNode.addInput('in', 'number')
      parentGraph.add(externalNode)

      // External connection from subgraph output should work
      expect(() => {
        subgraphNode.connect(0, externalNode, 0)
      }).not.toThrow()

      expect(
        subgraphNode.outputs[0].links?.includes(externalNode.inputs[0].link!)
      ).toBe(true)
      expect(externalNode.inputs[0].link).not.toBe(null)
    }
  )

  subgraphTest(
    'empty output slot creation enables dynamic IO',
    ({ simpleSubgraph }) => {
      const initialOutputCount = simpleSubgraph.outputs.length

      // Create empty output slot
      simpleSubgraph.addOutput('', '*')

      // Should create new output
      expect(simpleSubgraph.outputs.length).toBe(initialOutputCount + 1)

      // The empty slot should be configurable
      const emptyOutput = simpleSubgraph.outputs.at(-1)!
      expect(emptyOutput.name).toBe('')
      expect(emptyOutput.type).toBe('*')
    }
  )

  subgraphTest(
    'handles slot removal with active connections',
    ({ subgraphWithNode }) => {
      const { subgraph, subgraphNode, parentGraph } = subgraphWithNode

      const externalNode = new LGraphNode('External Target')
      externalNode.addInput('in', '*')
      parentGraph.add(externalNode)

      subgraphNode.connect(0, externalNode, 0)

      // Verify connection exists
      expect(externalNode.inputs[0].link).not.toBe(null)

      // Remove the existing output (fixture creates one output)
      const outputToRemove = subgraph.outputs[0]
      subgraph.removeOutput(outputToRemove)

      // Connection should be cleaned up
      expect(subgraphNode.outputs.length).toBe(0)
      expect(externalNode.inputs[0].link).toBe(null)
    }
  )

  subgraphTest(
    'handles slot renaming updates all references',
    ({ subgraphWithNode }) => {
      const { subgraph, subgraphNode, parentGraph } = subgraphWithNode

      const externalNode = new LGraphNode('External Target')
      externalNode.addInput('in', '*')
      parentGraph.add(externalNode)

      subgraphNode.connect(0, externalNode, 0)

      // Verify connection exists
      expect(externalNode.inputs[0].link).not.toBe(null)

      // Rename the existing output (fixture creates output named "output")
      const outputToRename = subgraph.outputs[0]
      subgraph.renameOutput(outputToRename, 'new_name')

      // Connection should persist and subgraph definition should be updated
      expect(externalNode.inputs[0].link).not.toBe(null)
      expect(subgraph.outputs[0].label).toBe('new_name')
      expect(subgraph.outputs[0].displayName).toBe('new_name')
    }
  )
})

describe('SubgraphIO - Boundary Connection Management', () => {
  subgraphTest(
    'verifies cross-boundary link resolution',
    ({ complexSubgraph }) => {
      const subgraphNode = createTestSubgraphNode(complexSubgraph)
      const parentGraph = subgraphNode.graph!

      const externalSource = new LGraphNode('External Source')
      externalSource.addOutput('out', 'number')
      parentGraph.add(externalSource)

      const externalTarget = new LGraphNode('External Target')
      externalTarget.addInput('in', 'number')
      parentGraph.add(externalTarget)

      externalSource.connect(0, subgraphNode, 0)
      subgraphNode.connect(0, externalTarget, 0)

      expect(subgraphNode.inputs[0].link).not.toBe(null)
      expect(externalTarget.inputs[0].link).not.toBe(null)
    }
  )

  subgraphTest(
    'handles bypass nodes that pass through data',
    ({ simpleSubgraph }) => {
      const subgraphNode = createTestSubgraphNode(simpleSubgraph)
      const parentGraph = subgraphNode.graph!

      const externalSource = new LGraphNode('External Source')
      externalSource.addOutput('out', 'number')
      parentGraph.add(externalSource)

      const externalTarget = new LGraphNode('External Target')
      externalTarget.addInput('in', 'number')
      parentGraph.add(externalTarget)

      externalSource.connect(0, subgraphNode, 0)
      subgraphNode.connect(0, externalTarget, 0)

      expect(subgraphNode.inputs[0].link).not.toBe(null)
      expect(externalTarget.inputs[0].link).not.toBe(null)
    }
  )

  subgraphTest(
    'tests link integrity across subgraph boundaries',
    ({ subgraphWithNode }) => {
      const { subgraphNode, parentGraph } = subgraphWithNode

      const externalSource = new LGraphNode('External Source')
      externalSource.addOutput('out', '*')
      parentGraph.add(externalSource)

      const externalTarget = new LGraphNode('External Target')
      externalTarget.addInput('in', '*')
      parentGraph.add(externalTarget)

      externalSource.connect(0, subgraphNode, 0)
      subgraphNode.connect(0, externalTarget, 0)

      const inputBoundaryLink = subgraphNode.inputs[0].link
      const outputBoundaryLink = externalTarget.inputs[0].link

      expect(inputBoundaryLink).toBeTruthy()
      expect(outputBoundaryLink).toBeTruthy()

      // Links should exist in parent graph
      expect(inputBoundaryLink).toBeTruthy()
      expect(outputBoundaryLink).toBeTruthy()
    }
  )

  subgraphTest(
    'verifies proper link cleanup on slot removal',
    ({ complexSubgraph }) => {
      const subgraphNode = createTestSubgraphNode(complexSubgraph)
      const parentGraph = subgraphNode.graph!

      const externalSource = new LGraphNode('External Source')
      externalSource.addOutput('out', 'number')
      parentGraph.add(externalSource)

      const externalTarget = new LGraphNode('External Target')
      externalTarget.addInput('in', 'number')
      parentGraph.add(externalTarget)

      externalSource.connect(0, subgraphNode, 0)
      subgraphNode.connect(0, externalTarget, 0)

      expect(subgraphNode.inputs[0].link).not.toBe(null)
      expect(externalTarget.inputs[0].link).not.toBe(null)

      const inputToRemove = complexSubgraph.inputs[0]
      complexSubgraph.removeInput(inputToRemove)

      expect(subgraphNode.inputs.findIndex((i) => i.name === 'data')).toBe(-1)
      expect(externalSource.outputs[0].links).toHaveLength(0)

      const outputToRemove = complexSubgraph.outputs[0]
      complexSubgraph.removeOutput(outputToRemove)

      expect(subgraphNode.outputs.findIndex((o) => o.name === 'result')).toBe(
        -1
      )
      expect(externalTarget.inputs[0].link).toBe(null)
    }
  )
})

describe('SubgraphIO - Advanced Scenarios', () => {
  it('handles multiple inputs and outputs with complex connections', () => {
    const subgraph = createTestSubgraph({
      name: 'Complex IO Test',
      inputs: [
        { name: 'input1', type: 'number' },
        { name: 'input2', type: 'string' },
        { name: 'input3', type: 'boolean' }
      ],
      outputs: [
        { name: 'output1', type: 'number' },
        { name: 'output2', type: 'string' }
      ]
    })

    const subgraphNode = createTestSubgraphNode(subgraph)

    // Should have correct number of slots
    expect(subgraphNode.inputs.length).toBe(3)
    expect(subgraphNode.outputs.length).toBe(2)

    // Each slot should have correct type
    expect(subgraphNode.inputs[0].type).toBe('number')
    expect(subgraphNode.inputs[1].type).toBe('string')
    expect(subgraphNode.inputs[2].type).toBe('boolean')
    expect(subgraphNode.outputs[0].type).toBe('number')
    expect(subgraphNode.outputs[1].type).toBe('string')
  })

  it('handles dynamic slot creation and removal', () => {
    const subgraph = createTestSubgraph({
      name: 'Dynamic IO Test'
    })

    const subgraphNode = createTestSubgraphNode(subgraph)

    // Start with no slots
    expect(subgraphNode.inputs.length).toBe(0)
    expect(subgraphNode.outputs.length).toBe(0)

    // Add slots dynamically
    subgraph.addInput('dynamic_input', 'number')
    subgraph.addOutput('dynamic_output', 'string')

    // SubgraphNode should automatically update
    expect(subgraphNode.inputs.length).toBe(1)
    expect(subgraphNode.outputs.length).toBe(1)
    expect(subgraphNode.inputs[0].name).toBe('dynamic_input')
    expect(subgraphNode.outputs[0].name).toBe('dynamic_output')

    // Remove slots
    subgraph.removeInput(subgraph.inputs[0])
    subgraph.removeOutput(subgraph.outputs[0])

    // SubgraphNode should automatically update
    expect(subgraphNode.inputs.length).toBe(0)
    expect(subgraphNode.outputs.length).toBe(0)
  })

  it('maintains slot synchronization across multiple instances', () => {
    const subgraph = createTestSubgraph({
      name: 'Multi-Instance Test',
      inputs: [{ name: 'shared_input', type: 'number' }],
      outputs: [{ name: 'shared_output', type: 'number' }]
    })

    // Create multiple instances
    const instance1 = createTestSubgraphNode(subgraph)
    const instance2 = createTestSubgraphNode(subgraph)
    const instance3 = createTestSubgraphNode(subgraph)

    // All instances should have same slots
    expect(instance1.inputs.length).toBe(1)
    expect(instance2.inputs.length).toBe(1)
    expect(instance3.inputs.length).toBe(1)

    // Modify the subgraph definition
    subgraph.addInput('new_input', 'string')
    subgraph.addOutput('new_output', 'boolean')

    // All instances should automatically update
    expect(instance1.inputs.length).toBe(2)
    expect(instance2.inputs.length).toBe(2)
    expect(instance3.inputs.length).toBe(2)
    expect(instance1.outputs.length).toBe(2)
    expect(instance2.outputs.length).toBe(2)
    expect(instance3.outputs.length).toBe(2)
  })
})

describe('SubgraphIO - Empty Slot Connection', () => {
  subgraphTest(
    'creates new input and connects when dragging from empty slot inside subgraph',
    ({ subgraphWithNode }) => {
      const { subgraph, subgraphNode } = subgraphWithNode

      // Create a node inside the subgraph that will receive the connection
      const internalNode = new LGraphNode('Internal Node')
      internalNode.addInput('in', 'string')
      subgraph.add(internalNode)

      // Simulate the connection process from the empty slot to an internal node
      // The -1 indicates a connection from the "empty" slot
      subgraph.inputNode.connectByType(-1, internalNode, 'string')

      // 1. A new input should have been created on the subgraph
      expect(subgraph.inputs.length).toBe(2) // Fixture adds one input already
      const newInput = subgraph.inputs[1]
      expect(newInput.name).toBe('in')
      expect(newInput.type).toBe('string')

      // 2. The subgraph node should now have a corresponding real input slot
      expect(subgraphNode.inputs.length).toBe(2)
      const subgraphInputSlot = subgraphNode.inputs[1]
      expect(subgraphInputSlot.name).toBe('in')

      // 3. A link should be established inside the subgraph
      expect(internalNode.inputs[0].link).not.toBe(null)
      const link = subgraph.links.get(internalNode.inputs[0].link!)!
      expect(link).toBeDefined()
      expect(link.target_id).toBe(internalNode.id)
      expect(link.target_slot).toBe(0)
      expect(link.origin_id).toBe(subgraph.inputNode.id)
      expect(link.origin_slot).toBe(1) // Should be the second slot
    }
  )
})
