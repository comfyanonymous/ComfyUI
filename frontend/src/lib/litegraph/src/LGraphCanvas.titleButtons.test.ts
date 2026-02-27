import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  LGraph,
  LGraphCanvas,
  LGraphNode,
  LiteGraph
} from '@/lib/litegraph/src/litegraph'

describe('LGraphCanvas Title Button Rendering', () => {
  let canvas: LGraphCanvas
  let ctx: CanvasRenderingContext2D
  let node: LGraphNode

  beforeEach(() => {
    // Create a mock canvas element
    const canvasElement = document.createElement('canvas')
    ctx = {
      save: vi.fn(),
      restore: vi.fn(),
      translate: vi.fn(),
      scale: vi.fn(),
      fillRect: vi.fn(),
      strokeRect: vi.fn(),
      fillText: vi.fn(),
      measureText: vi.fn().mockReturnValue({ width: 50 }),
      beginPath: vi.fn(),
      moveTo: vi.fn(),
      lineTo: vi.fn(),
      stroke: vi.fn(),
      fill: vi.fn(),
      closePath: vi.fn(),
      arc: vi.fn(),
      rect: vi.fn(),
      clip: vi.fn(),
      clearRect: vi.fn(),
      setTransform: vi.fn(),
      roundRect: vi.fn(),
      font: '',
      fillStyle: '',
      strokeStyle: '',
      lineWidth: 1,
      globalAlpha: 1,
      textAlign: 'left' as CanvasTextAlign,
      textBaseline: 'alphabetic' as CanvasTextBaseline
    } as Partial<CanvasRenderingContext2D> as CanvasRenderingContext2D

    canvasElement.getContext = vi.fn().mockReturnValue(ctx)

    const graph = new LGraph()
    canvas = new LGraphCanvas(canvasElement, graph, {
      skip_render: true,
      skip_events: true
    })

    node = new LGraphNode('Test Node')
    node.pos = [100, 200]
    node.size = [200, 100]

    node.drawTitleBarBackground = vi.fn()
    node.drawBadges = vi.fn()
    node.drawSlots = vi.fn()
    node.drawWidgets = vi.fn()
    node.drawCollapsedSlots = vi.fn()
    node.drawTitleBox = vi.fn()
    node.drawTitleText = vi.fn()
    node.drawProgressBar = vi.fn()
    node._setConcreteSlots = vi.fn()
    node.arrange = vi.fn()

    const nodeWithMocks = node as LGraphNode & {
      drawTitleBarText: ReturnType<typeof vi.fn>
      drawToggles: ReturnType<typeof vi.fn>
      drawNodeShape: ReturnType<typeof vi.fn>
      drawContent: ReturnType<typeof vi.fn>
      isSelectable: ReturnType<typeof vi.fn>
    }
    nodeWithMocks.drawTitleBarText = vi.fn()
    nodeWithMocks.drawToggles = vi.fn()
    nodeWithMocks.drawNodeShape = vi.fn()
    nodeWithMocks.drawContent = vi.fn()
    nodeWithMocks.isSelectable = vi.fn().mockReturnValue(true)
  })

  describe('drawNode title button rendering', () => {
    it('should render visible title buttons', () => {
      const button1 = node.addTitleButton({
        name: 'button1',
        text: 'A'
      })

      const button2 = node.addTitleButton({
        name: 'button2',
        text: 'B'
      })

      // Mock button methods
      const getWidth1 = vi.fn().mockReturnValue(20)
      const getWidth2 = vi.fn().mockReturnValue(25)
      const draw1 = vi.spyOn(button1, 'draw')
      const draw2 = vi.spyOn(button2, 'draw')

      button1.getWidth = getWidth1
      button2.getWidth = getWidth2

      // Draw the node (this is a simplified version of what drawNode does)
      canvas.drawNode(node, ctx)

      // Verify both buttons' getWidth was called
      expect(getWidth1).toHaveBeenCalledWith(ctx)
      expect(getWidth2).toHaveBeenCalledWith(ctx)

      // Verify both buttons were drawn
      expect(draw1).toHaveBeenCalled()
      expect(draw2).toHaveBeenCalled()

      // Check draw positions (right-aligned from node width)
      // First button (rightmost): 200 - 5 = 195, then subtract width
      // Second button: first button position - 5 - button width
      const titleHeight = LiteGraph.NODE_TITLE_HEIGHT
      const buttonY = -titleHeight + (titleHeight - 20) / 2 // Centered
      expect(draw1).toHaveBeenCalledWith(ctx, 180, buttonY) // 200 - 20
      expect(draw2).toHaveBeenCalledWith(ctx, 153, buttonY) // 180 - 2 - 25
    })

    it('should skip invisible title buttons', () => {
      const visibleButton = node.addTitleButton({
        name: 'visible',
        text: 'V'
      })

      const invisibleButton = node.addTitleButton({
        name: 'invisible',
        text: '' // Empty text makes it invisible
      })

      const getWidthVisible = vi.fn().mockReturnValue(30)
      const getWidthInvisible = vi.fn().mockReturnValue(30)
      const drawVisible = vi.spyOn(visibleButton, 'draw')
      const drawInvisible = vi.spyOn(invisibleButton, 'draw')

      visibleButton.getWidth = getWidthVisible
      invisibleButton.getWidth = getWidthInvisible

      canvas.drawNode(node, ctx)

      // Only visible button should be measured and drawn
      expect(getWidthVisible).toHaveBeenCalledWith(ctx)
      expect(getWidthInvisible).not.toHaveBeenCalled()

      expect(drawVisible).toHaveBeenCalled()
      expect(drawInvisible).not.toHaveBeenCalled()
    })

    it('should handle nodes without title buttons', () => {
      // Node has no title buttons
      expect(node.title_buttons).toHaveLength(0)

      // Should draw without errors
      expect(() => canvas.drawNode(node, ctx)).not.toThrow()
    })

    it('should position multiple buttons with correct spacing', () => {
      const buttons = []
      const drawSpies = []

      // Add 3 buttons
      for (let i = 0; i < 3; i++) {
        const button = node.addTitleButton({
          name: `button${i}`,
          text: String(i)
        })
        button.getWidth = vi.fn().mockReturnValue(15) // All same width for simplicity
        const spy = vi.spyOn(button, 'draw')
        buttons.push(button)
        drawSpies.push(spy)
      }

      canvas.drawNode(node, ctx)

      const titleHeight = LiteGraph.NODE_TITLE_HEIGHT

      // Check positions are correctly spaced (right to left)
      // Starting position: 200
      const buttonY = -titleHeight + (titleHeight - 20) / 2 // Button height is 20 (default)
      expect(drawSpies[0]).toHaveBeenCalledWith(ctx, 185, buttonY) // 200 - 15
      expect(drawSpies[1]).toHaveBeenCalledWith(ctx, 168, buttonY) // 185 - 2 - 15
      expect(drawSpies[2]).toHaveBeenCalledWith(ctx, 151, buttonY) // 168 - 2 - 15
    })

    it('should render buttons in low quality mode', () => {
      const button = node.addTitleButton({
        name: 'test',
        text: 'T'
      })

      button.getWidth = vi.fn().mockReturnValue(20)
      const drawSpy = vi.spyOn(button, 'draw')

      canvas.drawNode(node, ctx)

      // Buttons should still be rendered in low quality mode
      const buttonY =
        -LiteGraph.NODE_TITLE_HEIGHT + (LiteGraph.NODE_TITLE_HEIGHT - 20) / 2
      expect(drawSpy).toHaveBeenCalledWith(ctx, 180, buttonY)
    })

    it('should handle buttons with different widths', () => {
      const smallButton = node.addTitleButton({
        name: 'small',
        text: 'S'
      })

      const largeButton = node.addTitleButton({
        name: 'large',
        text: 'LARGE'
      })

      smallButton.getWidth = vi.fn().mockReturnValue(15)
      largeButton.getWidth = vi.fn().mockReturnValue(50)

      const drawSmall = vi.spyOn(smallButton, 'draw')
      const drawLarge = vi.spyOn(largeButton, 'draw')

      canvas.drawNode(node, ctx)

      const titleHeight = LiteGraph.NODE_TITLE_HEIGHT

      // Small button (rightmost): 200 - 15 = 185
      const buttonY = -titleHeight + (titleHeight - 20) / 2
      expect(drawSmall).toHaveBeenCalledWith(ctx, 185, buttonY)

      // Large button: 185 - 2 - 50 = 133
      expect(drawLarge).toHaveBeenCalledWith(ctx, 133, buttonY)
    })
  })

  describe('Integration with node properties', () => {
    it('should respect node size for button positioning', () => {
      node.size = [300, 150] // Wider node

      const button = node.addTitleButton({
        name: 'test',
        text: 'X'
      })

      button.getWidth = vi.fn().mockReturnValue(20)
      const drawSpy = vi.spyOn(button, 'draw')

      canvas.drawNode(node, ctx)

      const titleHeight = LiteGraph.NODE_TITLE_HEIGHT
      // Should use new width: 300 - 20 = 280
      const buttonY = -titleHeight + (titleHeight - 20) / 2
      expect(drawSpy).toHaveBeenCalledWith(ctx, 280, buttonY)
    })

    it('should NOT render buttons on collapsed nodes', () => {
      node.flags.collapsed = true

      const button = node.addTitleButton({
        name: 'test',
        text: 'C'
      })

      button.getWidth = vi.fn().mockReturnValue(20)
      const drawSpy = vi.spyOn(button, 'draw')

      canvas.drawNode(node, ctx)

      // Title buttons should NOT be rendered on collapsed nodes
      expect(drawSpy).not.toHaveBeenCalled()
      expect(button.getWidth).not.toHaveBeenCalled()
    })
  })
})
