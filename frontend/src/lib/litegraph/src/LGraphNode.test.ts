import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, vi } from 'vitest'

import type {
  INodeInputSlot,
  Point,
  ISerialisedNode
} from '@/lib/litegraph/src/litegraph'
import {
  LGraphNode,
  LiteGraph,
  LGraph,
  NodeInputSlot,
  NodeOutputSlot
} from '@/lib/litegraph/src/litegraph'

import { test } from './__fixtures__/testExtensions'
import { createMockLGraphNodeWithArrayBoundingRect } from '@/utils/__tests__/litegraphTestUtils'

interface NodeConstructorWithSlotOffset {
  slot_start_y?: number
}

function getMockISerialisedNode(
  data: Partial<ISerialisedNode>
): ISerialisedNode {
  return Object.assign(
    {
      id: 0,
      flags: {},
      type: 'TestNode',
      pos: [100, 100],
      size: [100, 100],
      order: 0,
      mode: 0
    },
    data
  )
}

describe('LGraphNode', () => {
  let node: LGraphNode
  let origLiteGraph: typeof LiteGraph

  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    origLiteGraph = Object.assign({}, LiteGraph)
    // @ts-expect-error Intended: Force remove an otherwise readonly non-optional property
    delete origLiteGraph.Classes

    Object.assign(LiteGraph, {
      NODE_TITLE_HEIGHT: 20,
      NODE_SLOT_HEIGHT: 15,
      NODE_TEXT_SIZE: 14,
      DEFAULT_SHADOW_COLOR: 'rgba(0,0,0,0.5)',
      DEFAULT_GROUP_FONT_SIZE: 24,
      isValidConnection: vi.fn().mockReturnValue(true)
    })
    node = new LGraphNode('Test Node')
    node.pos = [100, 200]
    node.size = [150, 100] // Example size

    // Reset mocks if needed
    vi.clearAllMocks()
  })

  afterEach(() => {
    Object.assign(LiteGraph, origLiteGraph)
  })

  test('should serialize position/size correctly', () => {
    const node = new LGraphNode('TestNode')
    node.pos = [10, 20]
    node.size = [30, 40]
    const json = node.serialize()
    expect(json.pos).toEqual([10, 20])
    expect(json.size).toEqual([30, 40])

    const configureData: ISerialisedNode = {
      id: node.id,
      type: node.type,
      pos: [50, 60],
      size: [70, 80],
      flags: {},
      order: node.order,
      mode: node.mode,
      inputs: node.inputs?.map((i) => ({
        name: i.name,
        type: i.type,
        link: i.link
      })),
      outputs: node.outputs?.map((o) => ({
        name: o.name,
        type: o.type,
        links: o.links,
        slot_index: o.slot_index
      }))
    }
    node.configure(configureData)
    expect(node.pos).toEqual(new Float64Array([50, 60]))
    expect(node.size).toEqual(new Float64Array([70, 80]))
  })

  test('should configure inputs correctly', () => {
    const node = new LGraphNode('TestNode')
    node.configure(
      getMockISerialisedNode({
        id: 0,
        inputs: [{ name: 'TestInput', type: 'number', link: null }]
      })
    )
    expect(node.inputs.length).toEqual(1)
    expect(node.inputs[0].name).toEqual('TestInput')
    expect(node.inputs[0].link).toEqual(null)
    expect(node.inputs[0]).instanceOf(NodeInputSlot)

    // Should not override existing inputs
    node.configure(getMockISerialisedNode({ id: 1 }))
    expect(node.id).toEqual(1)
    expect(node.inputs.length).toEqual(1)
  })

  test('should configure outputs correctly', () => {
    const node = new LGraphNode('TestNode')
    node.configure(
      getMockISerialisedNode({
        id: 0,
        outputs: [{ name: 'TestOutput', type: 'number', links: [] }]
      })
    )
    expect(node.outputs.length).toEqual(1)
    expect(node.outputs[0].name).toEqual('TestOutput')
    expect(node.outputs[0].type).toEqual('number')
    expect(node.outputs[0].links).toEqual([])
    expect(node.outputs[0]).instanceOf(NodeOutputSlot)

    // Should not override existing outputs
    node.configure(getMockISerialisedNode({ id: 1 }))
    expect(node.id).toEqual(1)
    expect(node.outputs.length).toEqual(1)
  })
  test('should not allow configuring id to -1', () => {
    const graph = new LGraph()
    const node = new LGraphNode('TestNode')
    graph.add(node)
    node.configure(getMockISerialisedNode({ id: -1 }))
    expect(node.id).not.toBe(-1)
  })

  describe('Disconnect I/O Slots', () => {
    test('should disconnect input correctly', () => {
      const node1 = new LGraphNode('SourceNode')
      const node2 = new LGraphNode('TargetNode')

      // Configure nodes with input/output slots
      node1.configure(
        getMockISerialisedNode({
          id: 1,
          outputs: [{ name: 'Output1', type: 'number', links: [] }]
        })
      )
      node2.configure(
        getMockISerialisedNode({
          id: 2,
          inputs: [{ name: 'Input1', type: 'number', link: null }]
        })
      )

      // Create a graph and add nodes to it
      const graph = new LGraph()
      graph.add(node1)
      graph.add(node2)

      // Connect the nodes
      const link = node1.connect(0, node2, 0)
      expect(link).not.toBeNull()
      expect(node2.inputs[0].link).toBe(link?.id)
      expect(node1.outputs[0].links).toContain(link?.id)

      // Test disconnecting by slot number
      const disconnected = node2.disconnectInput(0)
      expect(disconnected).toBe(true)
      expect(node2.inputs[0].link).toBeNull()
      expect(node1.outputs[0].links?.length).toBe(0)
      expect(graph._links.has(link?.id ?? -1)).toBe(false)

      // Test disconnecting by slot name
      node1.connect(0, node2, 0)
      const disconnectedByName = node2.disconnectInput('Input1')
      expect(disconnectedByName).toBe(true)
      expect(node2.inputs[0].link).toBeNull()

      // Test disconnecting non-existent slot
      const invalidDisconnect = node2.disconnectInput(999)
      expect(invalidDisconnect).toBe(false)

      // Test disconnecting already disconnected input
      const alreadyDisconnected = node2.disconnectInput(0)
      expect(alreadyDisconnected).toBe(true)
    })

    test('should disconnect output correctly', () => {
      const sourceNode = new LGraphNode('SourceNode')
      const targetNode1 = new LGraphNode('TargetNode1')
      const targetNode2 = new LGraphNode('TargetNode2')

      // Configure nodes with input/output slots
      sourceNode.configure(
        getMockISerialisedNode({
          id: 1,
          outputs: [
            { name: 'Output1', type: 'number', links: [] },
            { name: 'Output2', type: 'number', links: [] }
          ]
        })
      )
      targetNode1.configure(
        getMockISerialisedNode({
          id: 2,
          inputs: [{ name: 'Input1', type: 'number', link: null }]
        })
      )
      targetNode2.configure(
        getMockISerialisedNode({
          id: 3,
          inputs: [{ name: 'Input1', type: 'number', link: null }]
        })
      )

      // Create a graph and add nodes to it
      const graph = new LGraph()
      graph.add(sourceNode)
      graph.add(targetNode1)
      graph.add(targetNode2)

      // Connect multiple nodes to the same output
      const link1 = sourceNode.connect(0, targetNode1, 0)
      const link2 = sourceNode.connect(0, targetNode2, 0)
      expect(link1).not.toBeNull()
      expect(link2).not.toBeNull()
      expect(sourceNode.outputs[0].links?.length).toBe(2)

      // Test disconnecting specific target node
      const disconnectedSpecific = sourceNode.disconnectOutput(0, targetNode1)
      expect(disconnectedSpecific).toBe(true)
      expect(targetNode1.inputs[0].link).toBeNull()
      expect(sourceNode.outputs[0].links?.length).toBe(1)
      expect(graph._links.has(link1?.id ?? -1)).toBe(false)
      expect(graph._links.has(link2?.id ?? -1)).toBe(true)

      // Test disconnecting by slot name
      const link3 = sourceNode.connect(1, targetNode1, 0)
      expect(link3).not.toBeNull()
      const disconnectedByName = sourceNode.disconnectOutput(
        'Output2',
        targetNode1
      )
      expect(disconnectedByName).toBe(true)
      expect(targetNode1.inputs[0].link).toBeNull()
      expect(sourceNode.outputs[1].links?.length).toBe(0)

      // Test disconnecting all connections from an output
      const link4 = sourceNode.connect(0, targetNode1, 0)
      expect(link4).not.toBeNull()
      expect(sourceNode.outputs[0].links?.length).toBe(2)
      const disconnectedAll = sourceNode.disconnectOutput(0)
      expect(disconnectedAll).toBe(true)
      expect(sourceNode.outputs[0].links).toBeNull()
      expect(targetNode1.inputs[0].link).toBeNull()
      expect(targetNode2.inputs[0].link).toBeNull()
      expect(graph._links.has(link2?.id ?? -1)).toBe(false)
      expect(graph._links.has(link4?.id ?? -1)).toBe(false)

      // Test disconnecting non-existent slot
      const invalidDisconnect = sourceNode.disconnectOutput(999)
      expect(invalidDisconnect).toBe(false)

      // Test disconnecting already disconnected output
      const alreadyDisconnected = sourceNode.disconnectOutput(0)
      expect(alreadyDisconnected).toBe(false)
    })
  })

  describe('Applies correct link type on connection', () => {
    it.for<[string, string, string]>([
      ['IMAGE', 'IMAGE', 'IMAGE'],
      ['*', 'IMAGE', 'IMAGE'],
      ['IMAGE', '*', 'IMAGE'],
      ['*', '*', '*'],
      ['IMAGE,MASK', 'IMAGE,LATENT', 'IMAGE'],
      //An invalid connection should use input type
      ['Mask', 'IMAGE', 'IMAGE']
    ])(
      'Link from %s to %s should have type %s',
      ([output, input, expected]) => {
        const target = new LGraphNode('target')
        const source = new LGraphNode('source')
        const graph = new LGraph()

        target.addInput('input', input)
        source.addOutput('output', output)

        graph.add(source)
        graph.add(target)

        const link = source.connect(0, target, 0)
        expect(link?.type).toBe(expected)
      }
    )
  })

  describe('getInputPos and getOutputPos', () => {
    test('should handle collapsed nodes correctly', () => {
      const node = createMockLGraphNodeWithArrayBoundingRect('TestNode')
      node.pos = [100, 100]
      node.size = [100, 100]
      node.updateArea()
      node.configure(
        getMockISerialisedNode({
          id: 1,
          inputs: [{ name: 'Input1', type: 'number', link: null }],
          outputs: [{ name: 'Output1', type: 'number', links: [] }]
        })
      )

      // Collapse the node
      node.flags.collapsed = true

      // Get positions in collapsed state
      const inputPos = node.getInputPos(0)
      const outputPos = node.getOutputPos(0)

      expect(inputPos).toEqual([100, 90])
      expect(outputPos).toEqual([180, 90])
    })

    test('should return correct positions for input and output slots', () => {
      const node = new LGraphNode('TestNode')
      node.pos = [100, 100]
      node.size = [100, 100]
      node.configure(
        getMockISerialisedNode({
          id: 1,
          inputs: [{ name: 'Input1', type: 'number', link: null }],
          outputs: [{ name: 'Output1', type: 'number', links: [] }]
        })
      )

      const inputPos = node.getInputPos(0)
      const outputPos = node.getOutputPos(0)

      expect(inputPos).toEqual([107.5, 110.5])
      expect(outputPos).toEqual([193.5, 110.5])
    })
  })

  describe('getSlotOnPos', () => {
    test('should return undefined when point is outside node bounds', () => {
      const node = new LGraphNode('TestNode')
      node.pos = [100, 100]
      node.size = [100, 100]
      node.configure(
        getMockISerialisedNode({
          id: 1,
          inputs: [{ name: 'Input1', type: 'number', link: null }],
          outputs: [{ name: 'Output1', type: 'number', links: [] }]
        })
      )

      // Test point far outside node bounds
      expect(node.getSlotOnPos([0, 0])).toBeUndefined()
      // Test point just outside node bounds
      expect(node.getSlotOnPos([99, 99])).toBeUndefined()
    })

    test('should detect input slots correctly', () => {
      const node = createMockLGraphNodeWithArrayBoundingRect('TestNode')
      node.pos = [100, 100]
      node.size = [100, 100]
      node.updateArea()
      node.configure(
        getMockISerialisedNode({
          id: 1,
          inputs: [
            { name: 'Input1', type: 'number', link: null },
            { name: 'Input2', type: 'string', link: null }
          ]
        })
      )

      // Get position of first input slot
      const inputPos = node.getInputPos(0)
      // Test point directly on input slot
      const slot = node.getSlotOnPos(inputPos)
      expect(slot).toBeDefined()
      expect(slot?.name).toBe('Input1')

      // Test point near but not on input slot
      expect(node.getSlotOnPos([inputPos[0] - 15, inputPos[1]])).toBeUndefined()
    })

    test('should detect output slots correctly', () => {
      const node = createMockLGraphNodeWithArrayBoundingRect('TestNode')
      node.pos = [100, 100]
      node.size = [100, 100]
      node.updateArea()
      node.configure(
        getMockISerialisedNode({
          id: 1,
          outputs: [
            { name: 'Output1', type: 'number', links: [] },
            { name: 'Output2', type: 'string', links: [] }
          ]
        })
      )

      // Get position of first output slot
      const outputPos = node.getOutputPos(0)
      // Test point directly on output slot
      const slot = node.getSlotOnPos(outputPos)
      expect(slot).toBeDefined()
      expect(slot?.name).toBe('Output1')

      // Test point near but not on output slot
      const gotslot = node.getSlotOnPos([outputPos[0] + 30, outputPos[1]])
      expect(gotslot).toBeUndefined()
    })

    test('should prioritize input slots over output slots', () => {
      const node = createMockLGraphNodeWithArrayBoundingRect('TestNode')
      node.pos = [100, 100]
      node.size = [100, 100]
      node.updateArea()
      node.configure(
        getMockISerialisedNode({
          id: 1,
          inputs: [{ name: 'Input1', type: 'number', link: null }],
          outputs: [{ name: 'Output1', type: 'number', links: [] }]
        })
      )

      // Get positions of first input and output slots
      const inputPos = node.getInputPos(0)

      // Test point that could theoretically hit both slots
      // Should return the input slot due to priority
      const slot = node.getSlotOnPos(inputPos)
      expect(slot).toBeDefined()
      expect(slot?.name).toBe('Input1')
    })
  })

  describe('LGraphNode slot positioning', () => {
    test('should correctly position slots with absolute coordinates', () => {
      // Setup
      const node = new LGraphNode('test')
      node.pos = [100, 100]

      // Add input/output with absolute positions
      node.addInput('abs-input', 'number')
      node.inputs[0].pos = [10, 20]

      node.addOutput('abs-output', 'number')
      node.outputs[0].pos = [50, 30]

      // Test
      const inputPos = node.getInputPos(0)
      const outputPos = node.getOutputPos(0)

      // Absolute positions should be relative to node position
      expect(inputPos).toEqual([110, 120]) // node.pos + slot.pos
      expect(outputPos).toEqual([150, 130]) // node.pos + slot.pos
    })

    test('should correctly position default vertical slots', () => {
      // Setup
      const node = new LGraphNode('test')
      node.pos = [100, 100]

      // Add multiple inputs/outputs without absolute positions
      node.addInput('input1', 'number')
      node.addInput('input2', 'number')
      node.addOutput('output1', 'number')
      node.addOutput('output2', 'number')

      // Calculate expected positions
      const slotOffset = LiteGraph.NODE_SLOT_HEIGHT * 0.5
      const slotSpacing = LiteGraph.NODE_SLOT_HEIGHT
      const nodeWidth = node.size[0]

      // Test input positions
      expect(node.getInputPos(0)).toEqual([
        100 + slotOffset,
        100 + (0 + 0.7) * slotSpacing
      ])
      expect(node.getInputPos(1)).toEqual([
        100 + slotOffset,
        100 + (1 + 0.7) * slotSpacing
      ])

      // Test output positions
      expect(node.getOutputPos(0)).toEqual([
        100 + nodeWidth + 1 - slotOffset,
        100 + (0 + 0.7) * slotSpacing
      ])
      expect(node.getOutputPos(1)).toEqual([
        100 + nodeWidth + 1 - slotOffset,
        100 + (1 + 0.7) * slotSpacing
      ])
    })

    test('should skip absolute positioned slots when calculating vertical positions', () => {
      // Setup
      const node = new LGraphNode('test')
      node.pos = [100, 100]

      // Add mix of absolute and default positioned slots
      node.addInput('abs-input', 'number')
      node.inputs[0].pos = [10, 20]
      node.addInput('default-input1', 'number')
      node.addInput('default-input2', 'number')

      const slotOffset = LiteGraph.NODE_SLOT_HEIGHT * 0.5
      const slotSpacing = LiteGraph.NODE_SLOT_HEIGHT

      // Test: default positioned slots should be consecutive, ignoring absolute positioned ones
      expect(node.getInputPos(1)).toEqual([
        100 + slotOffset,
        100 + (0 + 0.7) * slotSpacing // First default slot starts at index 0
      ])
      expect(node.getInputPos(2)).toEqual([
        100 + slotOffset,
        100 + (1 + 0.7) * slotSpacing // Second default slot at index 1
      ])
    })
  })

  describe('widget serialization', () => {
    test('should only serialize widgets with serialize flag not set to false', () => {
      const node = new LGraphNode('TestNode')
      node.serialize_widgets = true

      // Add widgets with different serialization settings
      node.addWidget('number', 'serializable1', 1, null)
      node.addWidget('number', 'serializable2', 2, null)
      node.addWidget('number', 'non-serializable', 3, null)
      expect(node.widgets?.length).toBe(3)

      // Set serialize flag to false for the last widget
      node.widgets![2].serialize = false

      // Set some widget values
      node.widgets![0].value = 10
      node.widgets![1].value = 20
      node.widgets![2].value = 30

      // Serialize the node
      const serialized = node.serialize()

      // Check that only serializable widgets' values are included
      expect(serialized.widgets_values).toEqual([10, 20])
      expect(serialized.widgets_values).toHaveLength(2)
    })

    test('should only configure widgets with serialize flag not set to false', () => {
      const node = new LGraphNode('TestNode')
      node.serialize_widgets = true

      node.addWidget('number', 'non-serializable', 1, null)
      node.addWidget('number', 'serializable1', 2, null)
      expect(node.widgets?.length).toBe(2)

      node.widgets![0].serialize = false
      node.configure(
        getMockISerialisedNode({
          id: 1,
          type: 'TestNode',
          pos: [100, 100],
          size: [100, 100],
          properties: {},
          widgets_values: [100]
        })
      )

      expect(node.widgets![0].value).toBe(1)
      expect(node.widgets![1].value).toBe(100)
    })
  })

  describe('getInputSlotPos', () => {
    let inputSlot: INodeInputSlot

    beforeEach(() => {
      inputSlot = {
        name: 'test_in',
        type: 'string',
        link: null,
        boundingRect: [0, 0, 0, 0]
      }
    })
    test('should return position based on title height when collapsed', () => {
      node.flags.collapsed = true
      const expectedPos: Point = [100, 200 - LiteGraph.NODE_TITLE_HEIGHT * 0.5]
      expect(node.getInputSlotPos(inputSlot)).toEqual(expectedPos)
    })

    test('should return position based on input.pos when defined and not collapsed', () => {
      node.flags.collapsed = false
      inputSlot.pos = [10, 50]
      node.inputs = [inputSlot]
      const expectedPos: Point = [100 + 10, 200 + 50]
      expect(node.getInputSlotPos(inputSlot)).toEqual(expectedPos)
    })

    test('should return default vertical position when input.pos is undefined and not collapsed', () => {
      node.flags.collapsed = false
      const inputSlot2: INodeInputSlot = {
        name: 'test_in_2',
        type: 'number',
        link: null,
        boundingRect: [0, 0, 0, 0]
      }
      node.inputs = [inputSlot, inputSlot2]
      const slotIndex = 0
      const nodeOffsetY =
        (node.constructor as NodeConstructorWithSlotOffset).slot_start_y || 0
      const expectedY =
        200 + (slotIndex + 0.7) * LiteGraph.NODE_SLOT_HEIGHT + nodeOffsetY
      const expectedX = 100 + LiteGraph.NODE_SLOT_HEIGHT * 0.5
      expect(node.getInputSlotPos(inputSlot)).toEqual([expectedX, expectedY])
      const slotIndex2 = 1
      const expectedY2 =
        200 + (slotIndex2 + 0.7) * LiteGraph.NODE_SLOT_HEIGHT + nodeOffsetY
      expect(node.getInputSlotPos(inputSlot2)).toEqual([expectedX, expectedY2])
    })

    test('should return default vertical position including slot_start_y when defined', () => {
      ;(node.constructor as NodeConstructorWithSlotOffset).slot_start_y = 25
      node.flags.collapsed = false
      node.inputs = [inputSlot]
      const slotIndex = 0
      const nodeOffsetY = 25
      const expectedY =
        200 + (slotIndex + 0.7) * LiteGraph.NODE_SLOT_HEIGHT + nodeOffsetY
      const expectedX = 100 + LiteGraph.NODE_SLOT_HEIGHT * 0.5
      expect(node.getInputSlotPos(inputSlot)).toEqual([expectedX, expectedY])
      delete (node.constructor as NodeConstructorWithSlotOffset).slot_start_y
    })
    test('should not overwrite onMouseDown prototype', () => {
      expect(Object.prototype.hasOwnProperty.call(node, 'onMouseDown')).toEqual(
        false
      )
    })
  })
})
