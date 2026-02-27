import { describe, expect, it, vi } from 'vitest'

import {
  LinkDirection,
  ToOutputRenderLink
} from '@/lib/litegraph/src/litegraph'
import type { CustomEventTarget } from '@/lib/litegraph/src/infrastructure/CustomEventTarget'
import type { LinkConnectorEventMap } from '@/lib/litegraph/src/infrastructure/LinkConnectorEventMap'
import {
  createMockLGraphNode,
  createMockLinkNetwork,
  createMockNodeInputSlot,
  createMockNodeOutputSlot
} from '@/utils/__tests__/litegraphTestUtils'

describe('ToOutputRenderLink', () => {
  describe('connectToOutput', () => {
    it('should return early if inputNode is null', () => {
      // Setup
      const mockNetwork = createMockLinkNetwork()
      const mockFromSlot = createMockNodeInputSlot()
      const mockNode = createMockLGraphNode({
        inputs: [mockFromSlot],
        getInputPos: vi.fn().mockReturnValue([0, 0])
      })

      const renderLink = new ToOutputRenderLink(
        mockNetwork,
        mockNode,
        mockFromSlot,
        undefined,
        LinkDirection.CENTER
      )

      // Override the node property to simulate null case
      Object.defineProperty(renderLink, 'node', {
        value: null
      })

      const mockTargetNode = createMockLGraphNode({
        connectSlots: vi.fn()
      })
      const mockEvents: Partial<CustomEventTarget<LinkConnectorEventMap>> = {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
        dispatch: vi.fn()
      }

      // Act
      renderLink.connectToOutput(
        mockTargetNode,
        createMockNodeOutputSlot(),
        mockEvents as CustomEventTarget<LinkConnectorEventMap>
      )

      // Assert
      expect(mockTargetNode.connectSlots).not.toHaveBeenCalled()
      expect(mockEvents.dispatch).not.toHaveBeenCalled()
    })

    it('should create connection and dispatch event when inputNode exists', () => {
      // Setup
      const mockNetwork = createMockLinkNetwork()
      const mockFromSlot = createMockNodeInputSlot()
      const mockNode = createMockLGraphNode({
        inputs: [mockFromSlot],
        getInputPos: vi.fn().mockReturnValue([0, 0])
      })

      const renderLink = new ToOutputRenderLink(
        mockNetwork,
        mockNode,
        mockFromSlot,
        undefined,
        LinkDirection.CENTER
      )

      const mockNewLink = { id: 'new-link' }
      const mockTargetNode = createMockLGraphNode({
        connectSlots: vi.fn().mockReturnValue(mockNewLink)
      })
      const mockEvents: Partial<CustomEventTarget<LinkConnectorEventMap>> = {
        addEventListener: vi.fn(),
        removeEventListener: vi.fn(),
        dispatchEvent: vi.fn(),
        dispatch: vi.fn()
      }

      // Act
      renderLink.connectToOutput(
        mockTargetNode,
        createMockNodeOutputSlot(),
        mockEvents as CustomEventTarget<LinkConnectorEventMap>
      )

      // Assert
      expect(mockTargetNode.connectSlots).toHaveBeenCalledWith(
        expect.anything(),
        mockNode,
        mockFromSlot,
        undefined
      )
      expect(mockEvents.dispatch).toHaveBeenCalledWith(
        'link-created',
        mockNewLink
      )
    })
  })
})
