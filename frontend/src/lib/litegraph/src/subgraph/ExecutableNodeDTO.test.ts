// TODO: Fix these tests after migration
import { describe, expect, it, vi } from 'vitest'

import {
  LGraph,
  LGraphNode,
  ExecutableNodeDTO
} from '@/lib/litegraph/src/litegraph'

import {
  createNestedSubgraphs,
  createTestSubgraph,
  createTestSubgraphNode
} from './__fixtures__/subgraphHelpers'

describe.skip('ExecutableNodeDTO Creation', () => {
  it('should create DTO from regular node', () => {
    const graph = new LGraph()
    const node = new LGraphNode('Test Node')
    node.addInput('in', 'number')
    node.addOutput('out', 'string')
    graph.add(node)

    const executableNodes = new Map()
    const dto = new ExecutableNodeDTO(node, [], executableNodes, undefined)

    expect(dto.node).toBe(node)
    expect(dto.subgraphNodePath).toEqual([])
    expect(dto.subgraphNode).toBeUndefined()
    expect(dto.id).toBe(node.id.toString())
  })

  it('should create DTO with subgraph path', () => {
    const graph = new LGraph()
    const node = new LGraphNode('Inner Node')
    node.id = 42
    graph.add(node)
    const subgraphPath = ['10', '20'] as const

    const dto = new ExecutableNodeDTO(node, subgraphPath, new Map(), undefined)

    expect(dto.subgraphNodePath).toBe(subgraphPath)
    expect(dto.id).toBe('10:20:42')
  })

  it('should clone input slot data', () => {
    const graph = new LGraph()
    const node = new LGraphNode('Test Node')
    node.addInput('input1', 'number')
    node.addInput('input2', 'string')
    node.inputs[0].link = 123 // Simulate connected input
    graph.add(node)

    const dto = new ExecutableNodeDTO(node, [], new Map(), undefined)

    expect(dto.inputs).toHaveLength(2)
    expect(dto.inputs[0].name).toBe('input1')
    expect(dto.inputs[0].type).toBe('number')
    expect(dto.inputs[0].linkId).toBe(123)
    expect(dto.inputs[1].name).toBe('input2')
    expect(dto.inputs[1].type).toBe('string')
    expect(dto.inputs[1].linkId).toBeNull()

    // Should be a copy, not reference
    expect(dto.inputs).not.toBe(node.inputs)
  })

  it('should inherit graph reference', () => {
    const graph = new LGraph()
    const node = new LGraphNode('Test Node')
    graph.add(node)

    const dto = new ExecutableNodeDTO(node, [], new Map(), undefined)

    expect(dto.graph).toBe(graph)
  })

  it('should wrap applyToGraph method if present', () => {
    const graph = new LGraph()
    const node = new LGraphNode('Test Node')
    const mockApplyToGraph = vi.fn()
    Object.assign(node, { applyToGraph: mockApplyToGraph })
    graph.add(node)

    const dto = new ExecutableNodeDTO(node, [], new Map(), undefined)

    expect(dto.applyToGraph).toBeDefined()

    const args = ['arg1', 'arg2']
    ;(dto.applyToGraph as (...args: unknown[]) => void)(args[0], args[1])

    expect(mockApplyToGraph).toHaveBeenCalledWith(args[0], args[1])
  })

  it("should not create applyToGraph wrapper if method doesn't exist", () => {
    const graph = new LGraph()
    const node = new LGraphNode('Test Node')
    graph.add(node)

    const dto = new ExecutableNodeDTO(node, [], new Map(), undefined)

    expect(dto.applyToGraph).toBeUndefined()
  })
})

describe.skip('ExecutableNodeDTO Path-Based IDs', () => {
  it('should generate simple ID for root node', () => {
    const graph = new LGraph()
    const node = new LGraphNode('Root Node')
    node.id = 5
    graph.add(node)

    const dto = new ExecutableNodeDTO(node, [], new Map(), undefined)

    expect(dto.id).toBe('5')
  })

  it('should generate path-based ID for nested node', () => {
    const graph = new LGraph()
    const node = new LGraphNode('Nested Node')
    node.id = 3
    graph.add(node)
    const path = ['1', '2'] as const

    const dto = new ExecutableNodeDTO(node, path, new Map(), undefined)

    expect(dto.id).toBe('1:2:3')
  })

  it('should handle deep nesting paths', () => {
    const graph = new LGraph()
    const node = new LGraphNode('Deep Node')
    node.id = 99
    graph.add(node)
    const path = ['1', '2', '3', '4', '5'] as const

    const dto = new ExecutableNodeDTO(node, path, new Map(), undefined)

    expect(dto.id).toBe('1:2:3:4:5:99')
  })

  it('should handle string and number IDs consistently', () => {
    const graph = new LGraph()
    const node1 = new LGraphNode('Node 1')
    node1.id = 10
    graph.add(node1)

    const node2 = new LGraphNode('Node 2')
    node2.id = 20
    graph.add(node2)

    const dto1 = new ExecutableNodeDTO(node1, ['5'], new Map(), undefined)
    const dto2 = new ExecutableNodeDTO(node2, ['5'], new Map(), undefined)

    expect(dto1.id).toBe('5:10')
    expect(dto2.id).toBe('5:20')
  })
})

describe.skip('ExecutableNodeDTO Input Resolution', () => {
  it('should return undefined for unconnected inputs', () => {
    const graph = new LGraph()
    const node = new LGraphNode('Test Node')
    node.addInput('in', 'number')
    graph.add(node)

    const dto = new ExecutableNodeDTO(node, [], new Map(), undefined)

    // Unconnected input should return undefined
    const resolved = dto.resolveInput(0)
    expect(resolved).toBeUndefined()
  })

  it('should throw for non-existent input slots', () => {
    const graph = new LGraph()
    const node = new LGraphNode('No Input Node')
    graph.add(node)

    const dto = new ExecutableNodeDTO(node, [], new Map(), undefined)

    // Should throw SlotIndexError for non-existent input
    expect(() => dto.resolveInput(0)).toThrow('No input found for flattened id')
  })

  it('should handle subgraph boundary inputs', () => {
    const subgraph = createTestSubgraph({
      inputs: [{ name: 'input1', type: 'number' }],
      nodeCount: 1
    })
    const subgraphNode = createTestSubgraphNode(subgraph)

    // Get the inner node and create DTO
    const innerNode = subgraph.nodes[0]
    const dto = new ExecutableNodeDTO(innerNode, ['1'], new Map(), subgraphNode)

    // Should return undefined for unconnected input
    const resolved = dto.resolveInput(0)
    expect(resolved).toBeUndefined()
  })
})

describe.skip('ExecutableNodeDTO Output Resolution', () => {
  it('should resolve outputs for simple nodes', () => {
    const graph = new LGraph()
    const node = new LGraphNode('Test Node')
    node.addOutput('out', 'string')
    graph.add(node)

    const dto = new ExecutableNodeDTO(node, [], new Map(), undefined)

    // resolveOutput requires type and visited parameters
    const resolved = dto.resolveOutput(0, 'string', new Set())

    expect(resolved).toBeDefined()
    expect(resolved?.node).toBe(dto)
    expect(resolved?.origin_id).toBe(dto.id)
    expect(resolved?.origin_slot).toBe(0)
  })

  it('should resolve cross-boundary outputs in subgraphs', () => {
    const subgraph = createTestSubgraph({
      outputs: [{ name: 'output1', type: 'string' }],
      nodeCount: 1
    })
    const subgraphNode = createTestSubgraphNode(subgraph)

    // Get the inner node and create DTO
    const innerNode = subgraph.nodes[0]
    const dto = new ExecutableNodeDTO(innerNode, ['1'], new Map(), subgraphNode)

    const resolved = dto.resolveOutput(0, 'string', new Set())

    expect(resolved).toBeDefined()
  })

  it('should handle nodes with no outputs', () => {
    const graph = new LGraph()
    const node = new LGraphNode('No Output Node')
    graph.add(node)

    const dto = new ExecutableNodeDTO(node, [], new Map(), undefined)

    // For regular nodes, resolveOutput returns the node itself even if no outputs
    // This tests the current implementation behavior
    const resolved = dto.resolveOutput(0, 'string', new Set())
    expect(resolved).toBeDefined()
    expect(resolved?.node).toBe(dto)
    expect(resolved?.origin_slot).toBe(0)
  })
})

describe.skip('ExecutableNodeDTO Properties', () => {
  it('should provide access to basic properties', () => {
    const graph = new LGraph()
    const node = new LGraphNode('Test Node')
    node.id = 42
    node.addInput('input', 'number')
    node.addOutput('output', 'string')
    graph.add(node)

    const dto = new ExecutableNodeDTO(node, ['1', '2'], new Map(), undefined)

    expect(dto.id).toBe('1:2:42')
    expect(dto.type).toBe(node.type)
    expect(dto.title).toBe(node.title)
    expect(dto.mode).toBe(node.mode)
    expect(dto.isVirtualNode).toBe(node.isVirtualNode)
  })

  it('should provide access to input information', () => {
    const graph = new LGraph()
    const node = new LGraphNode('Test Node')
    node.addInput('testInput', 'number')
    node.inputs[0].link = 999 // Simulate connection
    graph.add(node)

    const dto = new ExecutableNodeDTO(node, [], new Map(), undefined)

    expect(dto.inputs).toBeDefined()
    expect(dto.inputs).toHaveLength(1)
    expect(dto.inputs[0].name).toBe('testInput')
    expect(dto.inputs[0].type).toBe('number')
    expect(dto.inputs[0].linkId).toBe(999)
  })
})

describe.skip('ExecutableNodeDTO Memory Efficiency', () => {
  it('should create lightweight objects', () => {
    const graph = new LGraph()
    const node = new LGraphNode('Test Node')
    node.addInput('in1', 'number')
    node.addInput('in2', 'string')
    node.addOutput('out1', 'number')
    node.addOutput('out2', 'string')
    graph.add(node)

    const dto = new ExecutableNodeDTO(node, ['1'], new Map(), undefined)

    // DTO should be lightweight - only essential properties
    expect(dto.node).toBe(node) // Reference, not copy
    expect(dto.subgraphNodePath).toEqual(['1']) // Reference to path
    expect(dto.inputs).toHaveLength(2) // Copied input data only

    // Should not duplicate heavy node data
    // eslint-disable-next-line no-prototype-builtins
    expect(dto.hasOwnProperty('outputs')).toBe(false) // Outputs not copied
    // eslint-disable-next-line no-prototype-builtins
    expect(dto.hasOwnProperty('widgets')).toBe(false) // Widgets not copied
  })

  it('should handle disposal without memory leaks', () => {
    const graph = new LGraph()
    const nodes: ExecutableNodeDTO[] = []

    // Create DTOs
    for (let i = 0; i < 100; i++) {
      const node = new LGraphNode(`Node ${i}`)
      node.id = i
      graph.add(node)
      const dto = new ExecutableNodeDTO(node, ['parent'], new Map(), undefined)
      nodes.push(dto)
    }

    expect(nodes).toHaveLength(100)

    // Clear references
    nodes.length = 0

    // DTOs should be eligible for garbage collection
    // (No explicit disposal needed - they're lightweight wrappers)
    expect(nodes).toHaveLength(0)
  })

  it('should not retain unnecessary references', () => {
    const subgraph = createTestSubgraph({ nodeCount: 1 })
    const subgraphNode = createTestSubgraphNode(subgraph)
    const innerNode = subgraph.nodes[0]

    const dto = new ExecutableNodeDTO(innerNode, ['1'], new Map(), subgraphNode)

    // Should hold necessary references
    expect(dto.node).toBe(innerNode)
    expect(dto.subgraphNode).toBe(subgraphNode)
    expect(dto.graph).toBe(innerNode.graph)

    // Should not hold heavy references that prevent GC
    // eslint-disable-next-line no-prototype-builtins
    expect(dto.hasOwnProperty('parentGraph')).toBe(false)
    // eslint-disable-next-line no-prototype-builtins
    expect(dto.hasOwnProperty('rootGraph')).toBe(false)
  })
})

describe.skip('ExecutableNodeDTO Integration', () => {
  it('should work with SubgraphNode flattening', () => {
    const subgraph = createTestSubgraph({ nodeCount: 3 })
    const subgraphNode = createTestSubgraphNode(subgraph)

    const flattened = subgraphNode.getInnerNodes(new Map())

    expect(flattened).toHaveLength(3)
    expect(flattened[0]).toBeInstanceOf(ExecutableNodeDTO)
    expect(flattened[0].id).toMatch(/^1:\d+$/)
  })

  it.skip('should handle nested subgraph flattening', () => {
    // FIXME: Complex nested structure requires proper parent graph setup
    // This test needs investigation of how resolveSubgraphIdPath works
    // Skip for now - will implement in edge cases test file
    const nested = createNestedSubgraphs({
      depth: 2,
      nodesPerLevel: 1
    })

    const rootSubgraphNode = nested.subgraphNodes[0]
    const executableNodes = new Map()
    const flattened = rootSubgraphNode.getInnerNodes(executableNodes)

    expect(flattened.length).toBeGreaterThan(0)
    const hierarchicalIds = flattened.filter((dto) => dto.id.includes(':'))
    expect(hierarchicalIds.length).toBeGreaterThan(0)
  })

  it('should preserve original node properties through DTO', () => {
    const graph = new LGraph()
    const originalNode = new LGraphNode('Original')
    originalNode.id = 123
    originalNode.addInput('test', 'number')
    originalNode.properties = { value: 42 }
    graph.add(originalNode)

    const dto = new ExecutableNodeDTO(
      originalNode,
      ['parent'],
      new Map(),
      undefined
    )

    // DTO should provide access to original node properties
    expect(dto.node.id).toBe(123)
    expect(dto.node.inputs).toHaveLength(1)
    expect(dto.node.properties.value).toBe(42)

    // But DTO ID should be path-based
    expect(dto.id).toBe('parent:123')
  })

  it('should handle execution context correctly', () => {
    const subgraph = createTestSubgraph({ nodeCount: 1 })
    const subgraphNode = createTestSubgraphNode(subgraph, { id: 99 })
    const innerNode = subgraph.nodes[0]
    innerNode.id = 55

    const dto = new ExecutableNodeDTO(
      innerNode,
      ['99'],
      new Map(),
      subgraphNode
    )

    // DTO provides execution context
    expect(dto.id).toBe('99:55') // Path-based execution ID
    expect(dto.node.id).toBe(55) // Original node ID preserved
    expect(dto.subgraphNode?.id).toBe(99) // Subgraph context
  })
})

describe.skip('ExecutableNodeDTO Scale Testing', () => {
  it('should create DTOs at scale', () => {
    const graph = new LGraph()
    const startTime = performance.now()
    const dtos: ExecutableNodeDTO[] = []

    // Create DTOs to test performance
    for (let i = 0; i < 1000; i++) {
      const node = new LGraphNode(`Node ${i}`)
      node.id = i
      node.addInput('in', 'number')
      graph.add(node)

      const dto = new ExecutableNodeDTO(node, ['parent'], new Map(), undefined)
      dtos.push(dto)
    }

    const endTime = performance.now()
    const duration = endTime - startTime

    expect(dtos).toHaveLength(1000)
    // Test deterministic properties instead of flaky timing
    expect(dtos[0].id).toBe('parent:0')
    expect(dtos[999].id).toBe('parent:999')
    expect(dtos.every((dto, i) => dto.id === `parent:${i}`)).toBe(true)

    console.log(`Created 1000 DTOs in ${duration.toFixed(2)}ms`)
  })

  it('should handle complex path generation correctly', () => {
    const graph = new LGraph()
    const node = new LGraphNode('Deep Node')
    node.id = 999
    graph.add(node)

    // Test deterministic path generation behavior
    const testCases = [
      { depth: 1, expectedId: '1:999' },
      { depth: 3, expectedId: '1:2:3:999' },
      { depth: 5, expectedId: '1:2:3:4:5:999' },
      { depth: 10, expectedId: '1:2:3:4:5:6:7:8:9:10:999' }
    ]

    for (const testCase of testCases) {
      const path = Array.from({ length: testCase.depth }, (_, i) =>
        (i + 1).toString()
      )
      const dto = new ExecutableNodeDTO(node, path, new Map(), undefined)
      expect(dto.id).toBe(testCase.expectedId)
    }
  })
})
