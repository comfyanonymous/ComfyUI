import { describe, expect, it, vi } from 'vitest'

import { LGraphButton, Rectangle } from '@/lib/litegraph/src/litegraph'

describe('LGraphButton', () => {
  describe('Constructor', () => {
    it('should create a button with default options', () => {
      const button = new LGraphButton({ text: '' })
      expect(button).toBeInstanceOf(LGraphButton)
      expect(button.name).toBeUndefined()
      expect(button._last_area).toBeInstanceOf(Rectangle)
    })

    it('should create a button with custom name', () => {
      const button = new LGraphButton({ text: '', name: 'test_button' })
      expect(button.name).toBe('test_button')
    })

    it('should inherit badge properties', () => {
      const button = new LGraphButton({
        text: 'Test',
        fgColor: '#FF0000',
        bgColor: '#0000FF',
        fontSize: 16
      })
      expect(button.text).toBe('Test')
      expect(button.fgColor).toBe('#FF0000')
      expect(button.bgColor).toBe('#0000FF')
      expect(button.fontSize).toBe(16)
      expect(button.visible).toBe(true) // visible is computed based on text length
    })
  })

  describe('draw', () => {
    it('should not draw if not visible', () => {
      const button = new LGraphButton({ text: '' }) // Empty text makes it invisible
      const ctx = {
        measureText: vi.fn().mockReturnValue({ width: 100 })
      } as Partial<CanvasRenderingContext2D> as CanvasRenderingContext2D

      const superDrawSpy = vi.spyOn(
        Object.getPrototypeOf(Object.getPrototypeOf(button)),
        'draw'
      )

      button.draw(ctx, 50, 100)

      expect(superDrawSpy).not.toHaveBeenCalled()
      expect(button._last_area.width).toBe(0) // Rectangle default width
    })

    it('should draw and update last area when visible', () => {
      const button = new LGraphButton({
        text: 'Click',
        xOffset: 5,
        yOffset: 10
      })

      const ctx = {
        measureText: vi.fn().mockReturnValue({ width: 60 }),
        fillRect: vi.fn(),
        fillText: vi.fn(),
        beginPath: vi.fn(),
        roundRect: vi.fn(),
        fill: vi.fn(),
        font: '',
        fillStyle: '',
        globalAlpha: 1
      } as Partial<CanvasRenderingContext2D> as CanvasRenderingContext2D

      const mockGetWidth = vi.fn().mockReturnValue(80)
      button.getWidth = mockGetWidth

      const x = 100
      const y = 50

      button.draw(ctx, x, y)

      // Check that last area was updated correctly
      expect(button._last_area[0]).toBe(x + button.xOffset) // 100 + 5 = 105
      expect(button._last_area[1]).toBe(y + button.yOffset) // 50 + 10 = 60
      expect(button._last_area[2]).toBe(80)
      expect(button._last_area[3]).toBe(button.height)
    })

    it('should calculate last area without offsets', () => {
      const button = new LGraphButton({
        text: 'Test'
      })

      const ctx = {
        measureText: vi.fn().mockReturnValue({ width: 40 }),
        fillRect: vi.fn(),
        fillText: vi.fn(),
        beginPath: vi.fn(),
        roundRect: vi.fn(),
        fill: vi.fn(),
        font: '',
        fillStyle: '',
        globalAlpha: 1
      } as Partial<CanvasRenderingContext2D> as CanvasRenderingContext2D

      const mockGetWidth = vi.fn().mockReturnValue(50)
      button.getWidth = mockGetWidth

      button.draw(ctx, 200, 100)

      expect(button._last_area[0]).toBe(200)
      expect(button._last_area[1]).toBe(100)
      expect(button._last_area[2]).toBe(50)
    })
  })

  describe('isPointInside', () => {
    it('should return true when point is inside button area', () => {
      const button = new LGraphButton({ text: 'Test' })
      // Set the last area manually
      button._last_area[0] = 100
      button._last_area[1] = 50
      button._last_area[2] = 80
      button._last_area[3] = 20

      // Test various points inside
      expect(button.isPointInside(100, 50)).toBe(true) // Top-left corner
      expect(button.isPointInside(179, 69)).toBe(true) // Bottom-right corner
      expect(button.isPointInside(140, 60)).toBe(true) // Center
    })

    it('should return false when point is outside button area', () => {
      const button = new LGraphButton({ text: 'Test' })
      // Set the last area manually
      button._last_area[0] = 100
      button._last_area[1] = 50
      button._last_area[2] = 80
      button._last_area[3] = 20

      // Test various points outside
      expect(button.isPointInside(99, 50)).toBe(false) // Just left
      expect(button.isPointInside(181, 50)).toBe(false) // Just right
      expect(button.isPointInside(100, 49)).toBe(false) // Just above
      expect(button.isPointInside(100, 71)).toBe(false) // Just below
      expect(button.isPointInside(0, 0)).toBe(false) // Far away
    })

    it('should work with buttons that have not been drawn yet', () => {
      const button = new LGraphButton({ text: 'Test' })
      // _last_area has default values (0, 0, 0, 0)

      expect(button.isPointInside(10, 10)).toBe(false)
      expect(button.isPointInside(0, 0)).toBe(false)
    })
  })

  describe('Integration with LGraphBadge', () => {
    it('should properly inherit and use badge functionality', () => {
      const button = new LGraphButton({
        text: '→',
        fontSize: 20,
        fgColor: '#FFFFFF',
        bgColor: '#333333',
        xOffset: -10,
        yOffset: 5
      })

      const ctx = {
        measureText: vi.fn().mockReturnValue({ width: 20 }),
        fillRect: vi.fn(),
        fillText: vi.fn(),
        beginPath: vi.fn(),
        roundRect: vi.fn(),
        fill: vi.fn(),
        font: '',
        fillStyle: '',
        globalAlpha: 1
      } as Partial<CanvasRenderingContext2D> as CanvasRenderingContext2D

      // Draw the button
      button.draw(ctx, 100, 50)

      // Verify button draws text without background
      expect(ctx.beginPath).not.toHaveBeenCalled() // No background
      expect(ctx.roundRect).not.toHaveBeenCalled() // No background
      expect(ctx.fill).not.toHaveBeenCalled() // No background
      expect(ctx.fillText).toHaveBeenCalledWith(
        '→',
        expect.any(Number),
        expect.any(Number)
      ) // Just text
    })
  })
})
