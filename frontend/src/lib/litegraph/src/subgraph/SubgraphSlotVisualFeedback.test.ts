// TODO: Fix these tests after migration
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { DefaultConnectionColors } from '@/lib/litegraph/src/interfaces'
import { LGraphNode } from '@/lib/litegraph/src/litegraph'

import { createTestSubgraph } from './__fixtures__/subgraphHelpers'

interface MockColorContext {
  defaultInputColor: string
  defaultOutputColor: string
  getConnectedColor: ReturnType<typeof vi.fn>
  getDisconnectedColor: ReturnType<typeof vi.fn>
}

describe.skip('SubgraphSlot visual feedback', () => {
  let mockCtx: CanvasRenderingContext2D
  let mockColorContext: MockColorContext
  let globalAlphaValues: number[]

  beforeEach(() => {
    // Clear the array before each test
    globalAlphaValues = []

    // Create a mock canvas context that tracks all globalAlpha values
    const mockContext = {
      _globalAlpha: 1,
      get globalAlpha() {
        return this._globalAlpha
      },
      set globalAlpha(value: number) {
        this._globalAlpha = value
        globalAlphaValues.push(value)
      },
      fillStyle: '',
      strokeStyle: '',
      lineWidth: 1,
      beginPath: vi.fn(),
      arc: vi.fn(),
      fill: vi.fn(),
      stroke: vi.fn(),
      rect: vi.fn(),
      fillText: vi.fn()
    }
    mockCtx =
      mockContext as Partial<CanvasRenderingContext2D> as CanvasRenderingContext2D

    // Create a mock color context
    mockColorContext = {
      defaultInputColor: '#FF0000',
      defaultOutputColor: '#00FF00',
      getConnectedColor: vi.fn().mockReturnValue('#0000FF'),
      getDisconnectedColor: vi.fn().mockReturnValue('#AAAAAA')
    } as Partial<MockColorContext> as MockColorContext
  })

  it('should render SubgraphInput slots with full opacity when dragging from compatible slot', () => {
    const subgraph = createTestSubgraph()
    const node = new LGraphNode('TestNode')
    node.addInput('in', 'number')
    subgraph.add(node)

    // Add a subgraph input
    const subgraphInput = subgraph.addInput('value', 'number')

    // Simulate dragging from the subgraph input (which acts as output inside subgraph)
    const nodeInput = node.inputs[0]

    // Draw the slot with a compatible fromSlot
    subgraphInput.draw({
      ctx: mockCtx,
      colorContext:
        mockColorContext as Partial<DefaultConnectionColors> as DefaultConnectionColors,
      fromSlot: nodeInput,
      editorAlpha: 1
    })

    // Should render with full opacity (not 0.4)
    // Check that 0.4 was NOT set during drawing
    expect(globalAlphaValues).not.toContain(0.4)
  })

  it('should render SubgraphInput slots with 40% opacity when dragging from another SubgraphInput', () => {
    const subgraph = createTestSubgraph()

    // Add two subgraph inputs
    const subgraphInput1 = subgraph.addInput('value1', 'number')
    const subgraphInput2 = subgraph.addInput('value2', 'number')

    // Draw subgraphInput2 while dragging from subgraphInput1 (incompatible - both are outputs inside subgraph)
    subgraphInput2.draw({
      ctx: mockCtx,
      colorContext:
        mockColorContext as Partial<DefaultConnectionColors> as DefaultConnectionColors,
      fromSlot: subgraphInput1,
      editorAlpha: 1
    })

    // Should render with 40% opacity
    // Check that 0.4 was set during drawing
    expect(globalAlphaValues).toContain(0.4)
  })

  it('should render SubgraphOutput slots with full opacity when dragging from compatible slot', () => {
    const subgraph = createTestSubgraph()
    const node = new LGraphNode('TestNode')
    node.addOutput('out', 'number')
    subgraph.add(node)

    // Add a subgraph output
    const subgraphOutput = subgraph.addOutput('result', 'number')

    // Simulate dragging from a node output
    const nodeOutput = node.outputs[0]

    // Draw the slot with a compatible fromSlot
    subgraphOutput.draw({
      ctx: mockCtx,
      colorContext:
        mockColorContext as Partial<DefaultConnectionColors> as DefaultConnectionColors,
      fromSlot: nodeOutput,
      editorAlpha: 1
    })

    // Should render with full opacity (not 0.4)
    // Check that 0.4 was NOT set during drawing
    expect(globalAlphaValues).not.toContain(0.4)
  })

  it('should render SubgraphOutput slots with 40% opacity when dragging from another SubgraphOutput', () => {
    const subgraph = createTestSubgraph()

    // Add two subgraph outputs
    const subgraphOutput1 = subgraph.addOutput('result1', 'number')
    const subgraphOutput2 = subgraph.addOutput('result2', 'number')

    // Draw subgraphOutput2 while dragging from subgraphOutput1 (incompatible - both are inputs inside subgraph)
    subgraphOutput2.draw({
      ctx: mockCtx,
      colorContext:
        mockColorContext as Partial<DefaultConnectionColors> as DefaultConnectionColors,
      fromSlot: subgraphOutput1,
      editorAlpha: 1
    })

    // Should render with 40% opacity
    // Check that 0.4 was set during drawing
    expect(globalAlphaValues).toContain(0.4)
  })

  // "not implemented yet"
  // it("should render slots with full opacity when dragging between compatible SubgraphInput and SubgraphOutput", () => {
  //   const subgraph = createTestSubgraph()

  //   // Add subgraph input and output with matching types
  //   const subgraphInput = subgraph.addInput("value", "number")
  //   const subgraphOutput = subgraph.addOutput("result", "number")

  //   // Draw SubgraphOutput slot while dragging from SubgraphInput
  //   subgraphOutput.draw({
  //     ctx: mockCtx,
  //     colorContext: mockColorContext,
  //     fromSlot: subgraphInput,
  //     editorAlpha: 1,
  //   })

  //   // Should render with full opacity
  //   expect(mockCtx.globalAlpha).toBe(1)
  // })

  it('should render slots with 40% opacity when dragging between incompatible types', () => {
    const subgraph = createTestSubgraph()
    const node = new LGraphNode('TestNode')
    node.addOutput('string_output', 'string')
    subgraph.add(node)

    // Add subgraph output with incompatible type
    const subgraphOutput = subgraph.addOutput('result', 'number')

    // Get the string output slot from the node
    const nodeStringOutput = node.outputs[0]

    // Draw the SubgraphOutput slot while dragging from a node output with incompatible type
    subgraphOutput.draw({
      ctx: mockCtx,
      colorContext:
        mockColorContext as Partial<DefaultConnectionColors> as DefaultConnectionColors,
      fromSlot: nodeStringOutput,
      editorAlpha: 1
    })

    // Should render with 40% opacity due to type mismatch
    // Check that 0.4 was set during drawing
    expect(globalAlphaValues).toContain(0.4)
  })
})
