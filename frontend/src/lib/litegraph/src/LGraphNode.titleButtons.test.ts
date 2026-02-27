import { describe, expect, it, vi } from 'vitest'

import type { LGraphCanvas } from '@/lib/litegraph/src/litegraph'
import { LGraphButton, LGraphNode } from '@/lib/litegraph/src/litegraph'

describe('LGraphNode Title Buttons', () => {
  describe('addTitleButton', () => {
    it('should add a title button to the node', () => {
      const node = new LGraphNode('Test Node')

      const button = node.addTitleButton({
        name: 'test_button',
        text: 'X',
        fgColor: '#FF0000'
      })

      expect(button).toBeInstanceOf(LGraphButton)
      expect(button.name).toBe('test_button')
      expect(button.text).toBe('X')
      expect(button.fgColor).toBe('#FF0000')
      expect(node.title_buttons).toHaveLength(1)
      expect(node.title_buttons[0]).toBe(button)
    })

    it('should add multiple title buttons', () => {
      const node = new LGraphNode('Test Node')

      const button1 = node.addTitleButton({ name: 'button1', text: 'A' })
      const button2 = node.addTitleButton({ name: 'button2', text: 'B' })
      const button3 = node.addTitleButton({ name: 'button3', text: 'C' })

      expect(node.title_buttons).toHaveLength(3)
      expect(node.title_buttons[0]).toBe(button1)
      expect(node.title_buttons[1]).toBe(button2)
      expect(node.title_buttons[2]).toBe(button3)
    })

    it('should create buttons with minimal options', () => {
      const node = new LGraphNode('Test Node')

      const button = node.addTitleButton({ text: '' })

      expect(button).toBeInstanceOf(LGraphButton)
      expect(button.name).toBeUndefined()
      expect(node.title_buttons).toHaveLength(1)
    })
  })

  describe('title button handling via canvas', () => {
    it('should handle click on title button through canvas processClick', () => {
      const node = new LGraphNode('Test Node')
      node.pos = [100, 200]
      node.size = [180, 60]

      const button = node.addTitleButton({
        name: 'close_button',
        text: 'X'
      })

      // Mock button methods
      button.getWidth = vi.fn().mockReturnValue(20)
      button.height = 16
      button.isPointInside = vi.fn().mockReturnValue(true)

      const canvas = {
        ctx: {} as CanvasRenderingContext2D,
        dispatch: vi.fn()
      } as Partial<LGraphCanvas> as LGraphCanvas

      // Mock the node's onTitleButtonClick method to verify it gets called
      const onTitleButtonClickSpy = vi.spyOn(node, 'onTitleButtonClick')

      // Calculate node-relative position for the click
      const clickPosRelativeToNode: [number, number] = [
        265 - node.pos[0], // 165
        178 - node.pos[1] // -22
      ]

      // Test the title button logic that's now in the canvas
      // This simulates what happens in LGraphCanvas.processMouseDown
      if (node.title_buttons?.length && !node.flags.collapsed) {
        const nodeRelativeX = clickPosRelativeToNode[0]
        const nodeRelativeY = clickPosRelativeToNode[1]

        for (let i = 0; i < node.title_buttons.length; i++) {
          const btn = node.title_buttons[i]
          if (btn.visible && btn.isPointInside(nodeRelativeX, nodeRelativeY)) {
            node.onTitleButtonClick(btn, canvas)
            break
          }
        }
      }

      expect(button.isPointInside).toHaveBeenCalledWith(165, -22)
      expect(onTitleButtonClickSpy).toHaveBeenCalledWith(button, canvas)
      expect(canvas.dispatch).toHaveBeenCalledWith(
        'litegraph:node-title-button-clicked',
        {
          node: node,
          button: button
        }
      )
    })

    it('should not handle click outside title buttons', () => {
      const node = new LGraphNode('Test Node')
      node.pos = [100, 200]
      node.size = [180, 60]

      const button = node.addTitleButton({
        name: 'test_button',
        text: 'T'
      })

      button.getWidth = vi.fn().mockReturnValue(20)
      button.height = 16
      button.isPointInside = vi.fn().mockReturnValue(false)

      const canvas = {
        ctx: {} as CanvasRenderingContext2D,
        dispatch: vi.fn()
      } as Partial<LGraphCanvas> as LGraphCanvas

      // Calculate node-relative position
      const clickPosRelativeToNode: [number, number] = [
        150 - node.pos[0], // 50
        180 - node.pos[1] // -20
      ]

      // Mock the node's onTitleButtonClick method to ensure it doesn't get called
      const onTitleButtonClickSpy = vi.spyOn(node, 'onTitleButtonClick')

      // Test the title button logic that's now in the canvas
      let buttonClicked = false
      if (node.title_buttons?.length && !node.flags.collapsed) {
        const nodeRelativeX = clickPosRelativeToNode[0]
        const nodeRelativeY = clickPosRelativeToNode[1]

        for (let i = 0; i < node.title_buttons.length; i++) {
          const btn = node.title_buttons[i]
          if (btn.visible && btn.isPointInside(nodeRelativeX, nodeRelativeY)) {
            node.onTitleButtonClick(btn, canvas)
            buttonClicked = true
            break
          }
        }
      }

      expect(button.isPointInside).toHaveBeenCalledWith(50, -20)
      expect(onTitleButtonClickSpy).not.toHaveBeenCalled()
      expect(canvas.dispatch).not.toHaveBeenCalled()
      expect(buttonClicked).toBe(false)
    })

    it('should handle multiple buttons correctly', () => {
      const node = new LGraphNode('Test Node')
      node.pos = [100, 200]
      node.size = [200, 60]

      const button1 = node.addTitleButton({
        name: 'button1',
        text: 'A'
      })

      const button2 = node.addTitleButton({
        name: 'button2',
        text: 'B'
      })

      // Mock button methods
      button1.getWidth = vi.fn().mockReturnValue(20)
      button2.getWidth = vi.fn().mockReturnValue(20)
      button1.height = button2.height = 16
      button1.isPointInside = vi.fn().mockReturnValue(false)
      button2.isPointInside = vi.fn().mockReturnValue(true)

      const canvas = {
        ctx: {} as CanvasRenderingContext2D,
        dispatch: vi.fn()
      } as Partial<LGraphCanvas> as LGraphCanvas

      // Mock the node's onTitleButtonClick method
      const onTitleButtonClickSpy = vi.spyOn(node, 'onTitleButtonClick')

      // Click on second button
      const titleY = 178 // node.pos[1] - NODE_TITLE_HEIGHT + 8 = 200 - 30 + 8 = 178
      // Calculate node-relative position
      const clickPosRelativeToNode: [number, number] = [
        255 - node.pos[0], // 155
        titleY - node.pos[1] // -22
      ]

      // Test the title button logic that's now in the canvas
      if (node.title_buttons?.length && !node.flags.collapsed) {
        const nodeRelativeX = clickPosRelativeToNode[0]
        const nodeRelativeY = clickPosRelativeToNode[1]

        for (let i = 0; i < node.title_buttons.length; i++) {
          const btn = node.title_buttons[i]
          if (btn.visible && btn.isPointInside(nodeRelativeX, nodeRelativeY)) {
            node.onTitleButtonClick(btn, canvas)
            break
          }
        }
      }

      expect(button1.isPointInside).toHaveBeenCalledWith(155, -22)
      expect(button2.isPointInside).toHaveBeenCalledWith(155, -22)
      expect(onTitleButtonClickSpy).toHaveBeenCalledWith(button2, canvas)
      expect(canvas.dispatch).toHaveBeenCalledWith(
        'litegraph:node-title-button-clicked',
        {
          node: node,
          button: button2
        }
      )
    })

    it('should skip invisible buttons', () => {
      const node = new LGraphNode('Test Node')
      node.pos = [100, 200]
      node.size = [180, 60]

      const button1 = node.addTitleButton({
        name: 'invisible_button',
        text: '' // Empty text makes it invisible
      })

      const button2 = node.addTitleButton({
        name: 'visible_button',
        text: 'V'
      })

      button1.getWidth = vi.fn().mockReturnValue(20)
      button2.getWidth = vi.fn().mockReturnValue(20)
      button1.height = button2.height = 16

      // Set visibility - button1 is invisible (empty text), button2 is visible
      button1.isPointInside = vi.fn().mockReturnValue(true) // Would be clicked if visible
      button2.isPointInside = vi.fn().mockReturnValue(true)

      const canvas = {
        ctx: {} as CanvasRenderingContext2D,
        dispatch: vi.fn()
      } as Partial<LGraphCanvas> as LGraphCanvas

      // Mock the node's onTitleButtonClick method
      const onTitleButtonClickSpy = vi.spyOn(node, 'onTitleButtonClick')

      // Click where both buttons would be positioned
      const titleY = 178 // node.pos[1] - NODE_TITLE_HEIGHT + 8 = 200 - 30 + 8 = 178
      // Calculate node-relative position
      const clickPosRelativeToNode: [number, number] = [
        265 - node.pos[0], // 165
        titleY - node.pos[1] // -22
      ]

      // Test the title button logic that's now in the canvas
      if (node.title_buttons?.length && !node.flags.collapsed) {
        const nodeRelativeX = clickPosRelativeToNode[0]
        const nodeRelativeY = clickPosRelativeToNode[1]

        for (let i = 0; i < node.title_buttons.length; i++) {
          const btn = node.title_buttons[i]
          // Only visible buttons are processed
          if (btn.visible && btn.isPointInside(nodeRelativeX, nodeRelativeY)) {
            node.onTitleButtonClick(btn, canvas)
            break
          }
        }
      }

      // button1 should not be checked because it's not visible
      expect(button1.isPointInside).not.toHaveBeenCalled()
      // button2 should be checked and clicked because it's visible
      expect(button2.isPointInside).toHaveBeenCalledWith(165, -22)
      expect(onTitleButtonClickSpy).toHaveBeenCalledWith(button2, canvas)
      expect(canvas.dispatch).toHaveBeenCalledWith(
        'litegraph:node-title-button-clicked',
        {
          node: node,
          button: button2 // Should click visible button, not invisible
        }
      )
    })
  })

  describe('onTitleButtonClick', () => {
    it('should dispatch litegraph:node-title-button-clicked event', () => {
      const node = new LGraphNode('Test Node')
      const button = new LGraphButton({ name: 'test_button', text: 'X' })

      const canvas = {
        dispatch: vi.fn()
      } as Partial<LGraphCanvas> as LGraphCanvas

      node.onTitleButtonClick(button, canvas)

      expect(canvas.dispatch).toHaveBeenCalledWith(
        'litegraph:node-title-button-clicked',
        {
          node: node,
          button: button
        }
      )
    })
  })
})
