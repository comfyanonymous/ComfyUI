// TODO: Fix these tests after migration
import { beforeEach, describe, expect, test, vi } from 'vitest'

import { LinkConnector } from '@/lib/litegraph/src/litegraph'
import {
  createMockCanvasPointerEvent,
  createMockLGraphNode,
  createMockLinkNetwork,
  createMockNodeInputSlot,
  createMockNodeOutputSlot
} from '@/utils/__tests__/litegraphTestUtils'

// Mocks
const mockSetConnectingLinks = vi.fn()

type RenderLinkItem = LinkConnector['renderLinks'][number]

// Mock a structure that has the needed method
function mockRenderLinkImpl(canConnect: boolean): RenderLinkItem {
  const partial: Partial<RenderLinkItem> = {
    toType: 'output',
    fromPos: [0, 0],
    fromSlotIndex: 0,
    fromDirection: 0,
    network: createMockLinkNetwork(),
    node: createMockLGraphNode(),
    fromSlot: createMockNodeOutputSlot(),
    dragDirection: 0,
    canConnectToInput: vi.fn().mockReturnValue(canConnect),
    canConnectToOutput: vi.fn().mockReturnValue(false),
    canConnectToReroute: vi.fn().mockReturnValue(false),
    connectToInput: vi.fn(),
    connectToOutput: vi.fn(),
    connectToSubgraphInput: vi.fn(),
    connectToRerouteOutput: vi.fn(),
    connectToSubgraphOutput: vi.fn(),
    connectToRerouteInput: vi.fn()
  }
  return partial as RenderLinkItem
}

const mockNode = createMockLGraphNode()
const mockInput = createMockNodeInputSlot()

describe.skip('LinkConnector', () => {
  let connector: LinkConnector

  beforeEach(() => {
    connector = new LinkConnector(mockSetConnectingLinks)
    // Clear the array directly before each test
    connector.renderLinks.length = 0
    vi.clearAllMocks()
  })

  describe.skip('isInputValidDrop', () => {
    test('should return false if there are no render links', () => {
      expect(connector.isInputValidDrop(mockNode, mockInput)).toBe(false)
    })

    test('should return true if at least one render link can connect', () => {
      const link1 = mockRenderLinkImpl(false)
      const link2 = mockRenderLinkImpl(true)
      connector.renderLinks.push(link1, link2)
      expect(connector.isInputValidDrop(mockNode, mockInput)).toBe(true)
      expect(link1.canConnectToInput).toHaveBeenCalledWith(mockNode, mockInput)
      expect(link2.canConnectToInput).toHaveBeenCalledWith(mockNode, mockInput)
    })

    test('should return false if no render links can connect', () => {
      const link1 = mockRenderLinkImpl(false)
      const link2 = mockRenderLinkImpl(false)
      connector.renderLinks.push(link1, link2)
      expect(connector.isInputValidDrop(mockNode, mockInput)).toBe(false)
      expect(link1.canConnectToInput).toHaveBeenCalledWith(mockNode, mockInput)
      expect(link2.canConnectToInput).toHaveBeenCalledWith(mockNode, mockInput)
    })

    test('should call canConnectToInput on each render link until one returns true', () => {
      const link1 = mockRenderLinkImpl(false)
      const link2 = mockRenderLinkImpl(true) // This one can connect
      const link3 = mockRenderLinkImpl(false)
      connector.renderLinks.push(link1, link2, link3)

      expect(connector.isInputValidDrop(mockNode, mockInput)).toBe(true)

      expect(link1.canConnectToInput).toHaveBeenCalledTimes(1)
      expect(link2.canConnectToInput).toHaveBeenCalledTimes(1) // Stops here
      expect(link3.canConnectToInput).not.toHaveBeenCalled() // Should not be called
    })
  })

  describe.skip('listenUntilReset', () => {
    test('should add listener for the specified event and for reset', () => {
      const listener = vi.fn()
      const addEventListenerSpy = vi.spyOn(connector.events, 'addEventListener')

      connector.listenUntilReset('before-drop-links', listener)

      expect(addEventListenerSpy).toHaveBeenCalledWith(
        'before-drop-links',
        listener,
        undefined
      )
      expect(addEventListenerSpy).toHaveBeenCalledWith(
        'reset',
        expect.any(Function),
        { once: true }
      )
    })

    test('should call the listener when the event is dispatched before reset', () => {
      const listener = vi.fn()
      const eventData = {
        renderLinks: [],
        event: createMockCanvasPointerEvent(0, 0)
      }
      connector.listenUntilReset('before-drop-links', listener)

      connector.events.dispatch('before-drop-links', eventData)

      expect(listener).toHaveBeenCalledTimes(1)
      expect(listener).toHaveBeenCalledWith(
        new CustomEvent('before-drop-links')
      )
    })

    test('should remove the listener when reset is dispatched', () => {
      const listener = vi.fn()
      const removeEventListenerSpy = vi.spyOn(
        connector.events,
        'removeEventListener'
      )

      connector.listenUntilReset('before-drop-links', listener)

      // Simulate the reset event being dispatched
      connector.events.dispatch('reset', false)

      // Check if removeEventListener was called correctly for the original listener
      expect(removeEventListenerSpy).toHaveBeenCalledWith(
        'before-drop-links',
        listener
      )
    })

    test('should not call the listener after reset is dispatched', () => {
      const listener = vi.fn()
      const eventData = {
        renderLinks: [],
        event: createMockCanvasPointerEvent(0, 0)
      }
      connector.listenUntilReset('before-drop-links', listener)

      // Dispatch reset first
      connector.events.dispatch('reset', false)

      // Then dispatch the original event
      connector.events.dispatch('before-drop-links', eventData)

      expect(listener).not.toHaveBeenCalled()
    })

    test('should pass options to addEventListener', () => {
      const listener = vi.fn()
      const options = { once: true }
      const addEventListenerSpy = vi.spyOn(connector.events, 'addEventListener')

      connector.listenUntilReset('after-drop-links', listener, options)

      expect(addEventListenerSpy).toHaveBeenCalledWith(
        'after-drop-links',
        listener,
        options
      )
      // Still adds the reset listener
      expect(addEventListenerSpy).toHaveBeenCalledWith(
        'reset',
        expect.any(Function),
        { once: true }
      )
    })
  })
})
