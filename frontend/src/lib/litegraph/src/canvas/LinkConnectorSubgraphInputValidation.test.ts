// TODO: Fix these tests after migration
import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  LinkConnector,
  MovingOutputLink,
  ToOutputRenderLink,
  LGraphNode,
  LLink
} from '@/lib/litegraph/src/litegraph'
import { ToInputFromIoNodeLink } from '@/lib/litegraph/src/canvas/ToInputFromIoNodeLink'
import type {
  CanvasPointerEvent,
  NodeInputSlot
} from '@/lib/litegraph/src/litegraph'
import { LinkDirection } from '@/lib/litegraph/src/types/globalEnums'

import { createTestSubgraph } from '../subgraph/__fixtures__/subgraphHelpers'
import {
  createMockCanvasPointerEvent,
  createMockNodeInputSlot
} from '@/utils/__tests__/litegraphTestUtils'

type MockPointerEvent = CanvasPointerEvent
type MockRenderLink = ToOutputRenderLink

describe('LinkConnector SubgraphInput connection validation', () => {
  let connector: LinkConnector
  const mockSetConnectingLinks = vi.fn()

  beforeEach(() => {
    connector = new LinkConnector(mockSetConnectingLinks)
    vi.clearAllMocks()
  })
  describe('Link disconnection validation', () => {
    it('should properly cleanup a moved input link', () => {
      const subgraph = createTestSubgraph({
        inputs: [{ name: 'number_input', type: 'number' }]
      })

      const fromTargetNode = new LGraphNode('TargetNode')
      fromTargetNode.addInput('number_in', 'number')
      subgraph.add(fromTargetNode)

      const toTargetNode = new LGraphNode('TargetNode')
      toTargetNode.addInput('number_in', 'number')
      subgraph.add(toTargetNode)

      const startLink = subgraph.inputNode.slots[0].connect(
        fromTargetNode.inputs[0],
        fromTargetNode
      )

      fromTargetNode.onConnectionsChange = vi.fn()
      toTargetNode.onConnectionsChange = vi.fn()

      const renderLink = new ToInputFromIoNodeLink(
        subgraph,
        subgraph.inputNode,
        subgraph.inputNode.slots[0],
        undefined,
        LinkDirection.CENTER,
        startLink
      )
      renderLink.connectToInput(
        toTargetNode,
        toTargetNode.inputs[0],
        connector.events
      )

      expect(fromTargetNode.inputs[0].link).toBeNull()
      expect(toTargetNode.inputs[0].link).not.toBeNull()
      expect(toTargetNode.onConnectionsChange).toHaveBeenCalledTimes(1)
      expect(fromTargetNode.onConnectionsChange).toHaveBeenCalledTimes(1)
    })
    it('should allow reconnection to same target', () => {
      const subgraph = createTestSubgraph({
        inputs: [{ name: 'number_input', type: 'number' }]
      })

      const node = new LGraphNode('TargetNode')
      node.addInput('number_in', 'number')
      subgraph.add(node)

      const link = subgraph.inputNode.slots[0].connect(node.inputs[0], node)

      const renderLink = new ToInputFromIoNodeLink(
        subgraph,
        subgraph.inputNode,
        subgraph.inputNode.slots[0],
        undefined,
        LinkDirection.CENTER,
        link
      )
      renderLink.connectToInput(node, node.inputs[0], connector.events)
      expect(node.inputs[0].link).not.toBeNull()
    })
  })

  describe('MovingOutputLink validation', () => {
    it('should implement canConnectToSubgraphInput method', () => {
      const subgraph = createTestSubgraph({
        inputs: [{ name: 'number_input', type: 'number' }]
      })

      const sourceNode = new LGraphNode('SourceNode')
      sourceNode.addOutput('number_out', 'number')
      subgraph.add(sourceNode)

      const targetNode = new LGraphNode('TargetNode')
      targetNode.addInput('number_in', 'number')
      subgraph.add(targetNode)

      const link = new LLink(1, 'number', sourceNode.id, 0, targetNode.id, 0)
      subgraph._links.set(link.id, link)

      const movingLink = new MovingOutputLink(subgraph, link)

      // Verify the method exists
      expect(typeof movingLink.canConnectToSubgraphInput).toBe('function')
    })

    it('should validate type compatibility correctly', () => {
      const subgraph = createTestSubgraph({
        inputs: [{ name: 'number_input', type: 'number' }]
      })

      const sourceNode = new LGraphNode('SourceNode')
      sourceNode.addOutput('number_out', 'number')
      sourceNode.addOutput('string_out', 'string')
      subgraph.add(sourceNode)

      const targetNode = new LGraphNode('TargetNode')
      targetNode.addInput('number_in', 'number')
      targetNode.addInput('string_in', 'string')
      subgraph.add(targetNode)

      // Create valid link (number -> number)
      const validLink = new LLink(
        1,
        'number',
        sourceNode.id,
        0,
        targetNode.id,
        0
      )
      subgraph._links.set(validLink.id, validLink)
      const validMovingLink = new MovingOutputLink(subgraph, validLink)

      // Create invalid link (string -> number)
      const invalidLink = new LLink(
        2,
        'string',
        sourceNode.id,
        1,
        targetNode.id,
        1
      )
      subgraph._links.set(invalidLink.id, invalidLink)
      const invalidMovingLink = new MovingOutputLink(subgraph, invalidLink)

      const numberInput = subgraph.inputs[0]

      // Test validation
      expect(validMovingLink.canConnectToSubgraphInput(numberInput)).toBe(true)
      expect(invalidMovingLink.canConnectToSubgraphInput(numberInput)).toBe(
        false
      )
    })

    it('should handle wildcard types', () => {
      const subgraph = createTestSubgraph({
        inputs: [{ name: 'wildcard_input', type: '*' }]
      })

      const sourceNode = new LGraphNode('SourceNode')
      sourceNode.addOutput('number_out', 'number')
      subgraph.add(sourceNode)

      const targetNode = new LGraphNode('TargetNode')
      targetNode.addInput('number_in', 'number')
      subgraph.add(targetNode)

      const link = new LLink(1, 'number', sourceNode.id, 0, targetNode.id, 0)
      subgraph._links.set(link.id, link)
      const movingLink = new MovingOutputLink(subgraph, link)

      const wildcardInput = subgraph.inputs[0]

      // Wildcard should accept any type
      expect(movingLink.canConnectToSubgraphInput(wildcardInput)).toBe(true)
    })
  })

  describe('ToOutputRenderLink validation', () => {
    it('should implement canConnectToSubgraphInput method', () => {
      // Create a minimal valid setup
      const subgraph = createTestSubgraph()
      const node = new LGraphNode('TestNode')
      node.id = 1
      node.addInput('test_in', 'number')
      subgraph.add(node)

      const slot = node.inputs[0] as NodeInputSlot
      const renderLink = new ToOutputRenderLink(subgraph, node, slot)

      // Verify the method exists
      expect(typeof renderLink.canConnectToSubgraphInput).toBe('function')
    })
  })

  describe('dropOnIoNode validation', () => {
    it('should prevent invalid connections when dropping on SubgraphInputNode', () => {
      const subgraph = createTestSubgraph({
        inputs: [{ name: 'number_input', type: 'number' }]
      })

      const sourceNode = new LGraphNode('SourceNode')
      sourceNode.addOutput('string_out', 'string')
      subgraph.add(sourceNode)

      const targetNode = new LGraphNode('TargetNode')
      targetNode.addInput('string_in', 'string')
      subgraph.add(targetNode)

      // Create an invalid link (string output -> string input, but subgraph expects number)
      const link = new LLink(1, 'string', sourceNode.id, 0, targetNode.id, 0)
      subgraph._links.set(link.id, link)
      const movingLink = new MovingOutputLink(subgraph, link)

      // Mock console.warn to verify it's called
      const consoleWarnSpy = vi
        .spyOn(console, 'warn')
        .mockImplementation(() => {})

      // Add the link to the connector
      connector.renderLinks.push(movingLink)
      connector.state.connectingTo = 'output'

      // Create mock event
      const mockEvent: MockPointerEvent = createMockCanvasPointerEvent(100, 100)

      // Mock the getSlotInPosition to return the subgraph input
      const mockGetSlotInPosition = vi.fn().mockReturnValue(subgraph.inputs[0])
      subgraph.inputNode.getSlotInPosition = mockGetSlotInPosition

      // Spy on connectToSubgraphInput to ensure it's NOT called
      const connectSpy = vi.spyOn(movingLink, 'connectToSubgraphInput')

      // Drop on the SubgraphInputNode
      connector.dropOnIoNode(subgraph.inputNode, mockEvent)

      // Verify that the invalid connection was skipped
      expect(consoleWarnSpy).toHaveBeenCalledWith(
        'Invalid connection type',
        'string',
        '->',
        'number'
      )
      expect(connectSpy).not.toHaveBeenCalled()

      consoleWarnSpy.mockRestore()
    })

    it('should allow valid connections when dropping on SubgraphInputNode', () => {
      const subgraph = createTestSubgraph({
        inputs: [{ name: 'number_input', type: 'number' }]
      })

      const sourceNode = new LGraphNode('SourceNode')
      sourceNode.addOutput('number_out', 'number')
      subgraph.add(sourceNode)

      const targetNode = new LGraphNode('TargetNode')
      targetNode.addInput('number_in', 'number')
      subgraph.add(targetNode)

      // Create a valid link (number -> number)
      const link = new LLink(1, 'number', sourceNode.id, 0, targetNode.id, 0)
      subgraph._links.set(link.id, link)
      const movingLink = new MovingOutputLink(subgraph, link)

      // Add the link to the connector
      connector.renderLinks.push(movingLink)
      connector.state.connectingTo = 'output'

      // Create mock event
      const mockEvent: MockPointerEvent = createMockCanvasPointerEvent(100, 100)

      // Mock the getSlotInPosition to return the subgraph input
      const mockGetSlotInPosition = vi.fn().mockReturnValue(subgraph.inputs[0])
      subgraph.inputNode.getSlotInPosition = mockGetSlotInPosition

      // Spy on connectToSubgraphInput to ensure it IS called
      const connectSpy = vi.spyOn(movingLink, 'connectToSubgraphInput')

      // Drop on the SubgraphInputNode
      connector.dropOnIoNode(subgraph.inputNode, mockEvent)

      // Verify that the valid connection was made
      expect(connectSpy).toHaveBeenCalledWith(
        subgraph.inputs[0],
        connector.events
      )
    })
  })

  describe('isSubgraphInputValidDrop', () => {
    it('should check if render links can connect to SubgraphInput', () => {
      const subgraph = createTestSubgraph({
        inputs: [{ name: 'number_input', type: 'number' }]
      })

      const sourceNode = new LGraphNode('SourceNode')
      sourceNode.addOutput('number_out', 'number')
      sourceNode.addOutput('string_out', 'string')
      subgraph.add(sourceNode)

      const targetNode = new LGraphNode('TargetNode')
      targetNode.addInput('number_in', 'number')
      targetNode.addInput('string_in', 'string')
      subgraph.add(targetNode)

      // Create valid and invalid links
      const validLink = new LLink(
        1,
        'number',
        sourceNode.id,
        0,
        targetNode.id,
        0
      )
      const invalidLink = new LLink(
        2,
        'string',
        sourceNode.id,
        1,
        targetNode.id,
        1
      )
      subgraph._links.set(validLink.id, validLink)
      subgraph._links.set(invalidLink.id, invalidLink)

      const validMovingLink = new MovingOutputLink(subgraph, validLink)
      const invalidMovingLink = new MovingOutputLink(subgraph, invalidLink)

      const subgraphInput = subgraph.inputs[0]

      // Test with only invalid link
      connector.renderLinks.length = 0
      connector.renderLinks.push(invalidMovingLink)
      expect(connector.isSubgraphInputValidDrop(subgraphInput)).toBe(false)

      // Test with valid link
      connector.renderLinks.length = 0
      connector.renderLinks.push(validMovingLink)
      expect(connector.isSubgraphInputValidDrop(subgraphInput)).toBe(true)

      // Test with mixed links
      connector.renderLinks.length = 0
      connector.renderLinks.push(invalidMovingLink, validMovingLink)
      expect(connector.isSubgraphInputValidDrop(subgraphInput)).toBe(true)
    })

    it('should handle render links without canConnectToSubgraphInput method', () => {
      const subgraph = createTestSubgraph({
        inputs: [{ name: 'number_input', type: 'number' }]
      })

      // Create a mock render link without the method
      const mockLink: Partial<MockRenderLink> = {
        fromSlot: createMockNodeInputSlot({ type: 'number' })
        // No canConnectToSubgraphInput method
      }

      connector.renderLinks.push(mockLink as MockRenderLink)

      const subgraphInput = subgraph.inputs[0]

      // Should return false as the link doesn't have the method
      expect(connector.isSubgraphInputValidDrop(subgraphInput)).toBe(false)
    })
  })
})
