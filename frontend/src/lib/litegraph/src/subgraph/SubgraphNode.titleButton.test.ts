// TODO: Fix these tests after migration
import { describe, expect, it, vi } from 'vitest'

import { LGraphButton } from '@/lib/litegraph/src/litegraph'
import type { LGraphCanvas } from '@/lib/litegraph/src/litegraph'
import type { CanvasPointerEvent } from '@/lib/litegraph/src/types/events'

import {
  createTestSubgraph,
  createTestSubgraphNode
} from './__fixtures__/subgraphHelpers'

interface MockPointerEvent {
  canvasX: number
  canvasY: number
}

describe.skip('SubgraphNode Title Button', () => {
  describe.skip('Constructor', () => {
    it('should automatically add enter_subgraph button', () => {
      const subgraph = createTestSubgraph({
        name: 'Test Subgraph',
        inputs: [{ name: 'input', type: 'number' }]
      })

      const subgraphNode = createTestSubgraphNode(subgraph)

      expect(subgraphNode.title_buttons).toHaveLength(1)

      const button = subgraphNode.title_buttons[0]
      expect(button).toBeInstanceOf(LGraphButton)
      expect(button.name).toBe('enter_subgraph')
      expect(button.text).toBe('\uE93B') // pi-window-maximize
      expect(button.xOffset).toBe(-10)
      expect(button.yOffset).toBe(0)
      expect(button.fontSize).toBe(16)
    })

    it('should preserve enter_subgraph button when adding more buttons', () => {
      const subgraph = createTestSubgraph()
      const subgraphNode = createTestSubgraphNode(subgraph)

      // Add another button
      const customButton = subgraphNode.addTitleButton({
        name: 'custom_button',
        text: 'C'
      })

      expect(subgraphNode.title_buttons).toHaveLength(2)
      expect(subgraphNode.title_buttons[0].name).toBe('enter_subgraph')
      expect(subgraphNode.title_buttons[1]).toBe(customButton)
    })
  })

  describe.skip('onTitleButtonClick', () => {
    it('should open subgraph when enter_subgraph button is clicked', () => {
      const subgraph = createTestSubgraph({
        name: 'Test Subgraph'
      })

      const subgraphNode = createTestSubgraphNode(subgraph)
      const enterButton = subgraphNode.title_buttons[0]

      const canvas = {
        openSubgraph: vi.fn(),
        dispatch: vi.fn()
      } as Partial<LGraphCanvas> as LGraphCanvas

      subgraphNode.onTitleButtonClick(enterButton, canvas)

      expect(canvas.openSubgraph).toHaveBeenCalledWith(subgraph)
      expect(canvas.dispatch).not.toHaveBeenCalled() // Should not call parent implementation
    })

    it('should call parent implementation for other buttons', () => {
      const subgraph = createTestSubgraph()
      const subgraphNode = createTestSubgraphNode(subgraph)

      const customButton = subgraphNode.addTitleButton({
        name: 'custom_button',
        text: 'X'
      })

      const canvas = {
        openSubgraph: vi.fn(),
        dispatch: vi.fn()
      } as Partial<LGraphCanvas> as LGraphCanvas

      subgraphNode.onTitleButtonClick(customButton, canvas)

      expect(canvas.openSubgraph).not.toHaveBeenCalled()
      expect(canvas.dispatch).toHaveBeenCalledWith(
        'litegraph:node-title-button-clicked',
        {
          node: subgraphNode,
          button: customButton
        }
      )
    })
  })

  describe.skip('Integration with node click handling', () => {
    it('should handle clicks on enter_subgraph button', () => {
      const subgraph = createTestSubgraph({
        name: 'Nested Subgraph',
        nodeCount: 3
      })

      const subgraphNode = createTestSubgraphNode(subgraph)
      subgraphNode.pos = [100, 100]
      subgraphNode.size = [200, 100]

      const enterButton = subgraphNode.title_buttons[0]
      enterButton.getWidth = vi.fn().mockReturnValue(25)
      enterButton.height = 20

      // Simulate button being drawn at node-relative coordinates
      // Button x: 200 - 5 - 25 = 170
      // Button y: -30 (title height)
      enterButton._last_area[0] = 170
      enterButton._last_area[1] = -30
      enterButton._last_area[2] = 25
      enterButton._last_area[3] = 20

      const canvas = {
        ctx: {
          measureText: vi.fn().mockReturnValue({ width: 25 })
        } as Partial<CanvasRenderingContext2D> as CanvasRenderingContext2D,
        openSubgraph: vi.fn(),
        dispatch: vi.fn()
      } as Partial<LGraphCanvas> as LGraphCanvas

      // Simulate click on the enter button
      const event: MockPointerEvent = {
        canvasX: 275, // Near right edge where button should be
        canvasY: 80 // In title area
      }

      // Calculate node-relative position
      const clickPosRelativeToNode: [number, number] = [
        275 - subgraphNode.pos[0], // 275 - 100 = 175
        80 - subgraphNode.pos[1] // 80 - 100 = -20
      ]

      // @ts-expect-error onMouseDown possibly undefined
      const handled = subgraphNode.onMouseDown(
        event as Partial<CanvasPointerEvent> as CanvasPointerEvent,
        clickPosRelativeToNode,
        canvas
      )

      expect(handled).toBe(true)
      expect(canvas.openSubgraph).toHaveBeenCalledWith(subgraph)
    })

    it('should not interfere with normal node operations', () => {
      const subgraph = createTestSubgraph()
      const subgraphNode = createTestSubgraphNode(subgraph)
      subgraphNode.pos = [100, 100]
      subgraphNode.size = [200, 100]

      const canvas = {
        ctx: {
          measureText: vi.fn().mockReturnValue({ width: 25 })
        } as Partial<CanvasRenderingContext2D> as CanvasRenderingContext2D,
        openSubgraph: vi.fn(),
        dispatch: vi.fn()
      } as Partial<LGraphCanvas> as LGraphCanvas

      // Click in the body of the node, not on button
      const event: MockPointerEvent = {
        canvasX: 200, // Middle of node
        canvasY: 150 // Body area
      }

      // Calculate node-relative position
      const clickPosRelativeToNode: [number, number] = [
        200 - subgraphNode.pos[0], // 200 - 100 = 100
        150 - subgraphNode.pos[1] // 150 - 100 = 50
      ]

      const handled = subgraphNode.onMouseDown!(
        event as Partial<CanvasPointerEvent> as CanvasPointerEvent,
        clickPosRelativeToNode,
        canvas
      )

      expect(handled).toBe(false)
      expect(canvas.openSubgraph).not.toHaveBeenCalled()
    })

    it('should not process button clicks when node is collapsed', () => {
      const subgraph = createTestSubgraph()
      const subgraphNode = createTestSubgraphNode(subgraph)
      subgraphNode.pos = [100, 100]
      subgraphNode.size = [200, 100]
      subgraphNode.flags.collapsed = true

      const enterButton = subgraphNode.title_buttons[0]
      enterButton.getWidth = vi.fn().mockReturnValue(25)
      enterButton.height = 20

      // Set button area as if it was drawn
      enterButton._last_area[0] = 170
      enterButton._last_area[1] = -30
      enterButton._last_area[2] = 25
      enterButton._last_area[3] = 20

      const canvas = {
        ctx: {
          measureText: vi.fn().mockReturnValue({ width: 25 })
        } as Partial<CanvasRenderingContext2D> as CanvasRenderingContext2D,
        openSubgraph: vi.fn(),
        dispatch: vi.fn()
      } as Partial<LGraphCanvas> as LGraphCanvas

      // Try to click on where the button would be
      const event: MockPointerEvent = {
        canvasX: 275,
        canvasY: 80
      }

      const clickPosRelativeToNode: [number, number] = [
        275 - subgraphNode.pos[0], // 175
        80 - subgraphNode.pos[1] // -20
      ]

      const handled = subgraphNode.onMouseDown!(
        event as Partial<CanvasPointerEvent> as CanvasPointerEvent,
        clickPosRelativeToNode,
        canvas
      )

      // Should not handle the click when collapsed
      expect(handled).toBe(false)
      expect(canvas.openSubgraph).not.toHaveBeenCalled()
    })
  })

  describe.skip('Visual properties', () => {
    it('should have appropriate visual properties for enter button', () => {
      const subgraph = createTestSubgraph()
      const subgraphNode = createTestSubgraphNode(subgraph)

      const enterButton = subgraphNode.title_buttons[0]

      // Check visual properties
      expect(enterButton.text).toBe('\uE93B') // pi-window-maximize
      expect(enterButton.fontSize).toBe(16) // Icon size
      expect(enterButton.xOffset).toBe(-10) // Positioned from right edge
      expect(enterButton.yOffset).toBe(0) // Centered vertically

      // Should be visible by default
      expect(enterButton.visible).toBe(true)
    })
  })
})
