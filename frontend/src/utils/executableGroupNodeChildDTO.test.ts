import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { GroupNodeHandler } from '@/extensions/core/groupNode'
import type {
  ExecutableLGraphNode,
  ExecutionId,
  LGraphNode
} from '@/lib/litegraph/src/litegraph'
import { ExecutableGroupNodeChildDTO } from '@/utils/executableGroupNodeChildDTO'
import { createMockLGraphNode } from './__tests__/litegraphTestUtils'

describe('ExecutableGroupNodeChildDTO', () => {
  let mockNode: LGraphNode
  let mockInputNode: LGraphNode
  let mockNodesByExecutionId: Map<ExecutionId, ExecutableLGraphNode>
  let mockGroupNodeHandler: GroupNodeHandler

  beforeEach(() => {
    // Create mock nodes
    mockNode = createMockLGraphNode({
      id: '3', // Simple node ID for most tests
      graph: {},
      getInputNode: vi.fn(),
      getInputLink: vi.fn(),
      inputs: []
    })

    mockInputNode = createMockLGraphNode({
      id: '1',
      graph: {}
    })

    // Create the nodesByExecutionId map
    mockNodesByExecutionId = new Map()

    mockGroupNodeHandler = {} as GroupNodeHandler
  })

  describe('resolveInput', () => {
    it('should resolve input from external node (node outside the group)', () => {
      // Setup: Group node child with ID '10:3'
      const groupNodeChild = createMockLGraphNode({
        id: '10:3',
        graph: {},
        getInputNode: vi.fn().mockReturnValue(mockInputNode),
        getInputLink: vi.fn().mockReturnValue({
          origin_slot: 0
        }),
        inputs: []
      })

      // External node with ID '1'
      const externalNodeDto = {
        id: '1',
        type: 'TestNode'
      } as ExecutableLGraphNode

      mockNodesByExecutionId.set('1', externalNodeDto)

      const dto = new ExecutableGroupNodeChildDTO(
        groupNodeChild,
        [], // No subgraph path - group is in root graph
        mockNodesByExecutionId,
        undefined,
        mockGroupNodeHandler
      )

      const result = dto.resolveInput(0)

      expect(result).toEqual({
        node: externalNodeDto,
        origin_id: '1',
        origin_slot: 0
      })
    })

    it('should resolve input from internal node (node inside the same group)', () => {
      // Setup: Group node child with ID '10:3'
      const groupNodeChild = createMockLGraphNode({
        id: '10:3',
        graph: {},
        getInputNode: vi.fn(),
        getInputLink: vi.fn(),
        inputs: []
      })

      // Internal node with ID '10:2'
      const internalInputNode = createMockLGraphNode({
        id: '10:2',
        graph: {}
      })

      const internalNodeDto = {
        id: '2',
        type: 'InternalNode'
      } as ExecutableLGraphNode

      // Internal nodes are stored with just their index
      mockNodesByExecutionId.set('2', internalNodeDto)

      vi.mocked(groupNodeChild.getInputNode).mockReturnValue(internalInputNode)
      vi.mocked(groupNodeChild.getInputLink).mockReturnValue({
        origin_slot: 1
      } as ReturnType<LGraphNode['getInputLink']>)

      const dto = new ExecutableGroupNodeChildDTO(
        groupNodeChild,
        [],
        mockNodesByExecutionId,
        undefined,
        mockGroupNodeHandler
      )

      const result = dto.resolveInput(0)

      expect(result).toEqual({
        node: internalNodeDto,
        origin_id: '10:2',
        origin_slot: 1
      })
    })

    it('should return undefined if no input node exists', () => {
      mockNode.getInputNode = vi.fn().mockReturnValue(null)

      const dto = new ExecutableGroupNodeChildDTO(
        mockNode,
        [],
        mockNodesByExecutionId,
        undefined,
        mockGroupNodeHandler
      )

      const result = dto.resolveInput(0)

      expect(result).toBeUndefined()
    })

    it('should throw error if input link is missing', () => {
      mockNode.getInputNode = vi.fn().mockReturnValue(mockInputNode)
      mockNode.getInputLink = vi.fn().mockReturnValue(null)

      const dto = new ExecutableGroupNodeChildDTO(
        mockNode,
        [],
        mockNodesByExecutionId,
        undefined,
        mockGroupNodeHandler
      )

      expect(() => dto.resolveInput(0)).toThrow('Failed to get input link')
    })

    it('should throw error if input node cannot be found in nodesByExecutionId', () => {
      // Node exists but is not in the map
      mockNode.getInputNode = vi.fn().mockReturnValue(mockInputNode)
      mockNode.getInputLink = vi.fn().mockReturnValue({
        origin_slot: 0
      })

      const dto = new ExecutableGroupNodeChildDTO(
        mockNode,
        [],
        mockNodesByExecutionId, // Empty map
        undefined,
        mockGroupNodeHandler
      )

      expect(() => dto.resolveInput(0)).toThrow(
        'Failed to get input node 1 for group node child 3 with slot 0'
      )
    })

    it('should throw error for group nodes inside subgraphs (unsupported)', () => {
      // Setup: Group node child inside a subgraph (execution ID has more than 2 segments)
      const nestedGroupNode = createMockLGraphNode({
        id: '1:2:3', // subgraph:groupnode:innernode
        graph: {},
        getInputNode: vi.fn().mockReturnValue(mockInputNode),
        getInputLink: vi.fn().mockReturnValue({
          origin_slot: 0
        }),
        inputs: []
      })

      // Create DTO with deeply nested path to simulate group node inside subgraph
      const dto = new ExecutableGroupNodeChildDTO(
        nestedGroupNode,
        ['1', '2'], // Path indicating it's inside a subgraph then group
        mockNodesByExecutionId,
        undefined,
        mockGroupNodeHandler
      )

      expect(() => dto.resolveInput(0)).toThrow(
        'Group nodes inside subgraphs are not supported. Please convert the group node to a subgraph instead.'
      )
    })
  })
})
