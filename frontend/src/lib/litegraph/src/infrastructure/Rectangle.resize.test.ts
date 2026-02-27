import { beforeEach, describe, expect, test } from 'vitest'

import { Rectangle } from '@/lib/litegraph/src/litegraph'

describe('Rectangle resize functionality', () => {
  let rect: Rectangle

  beforeEach(() => {
    rect = new Rectangle(100, 200, 300, 400) // x, y, width, height
    // So: left=100, top=200, right=400, bottom=600
  })

  describe('findContainingCorner', () => {
    const cornerSize = 15

    test('should detect NW (top-left) corner', () => {
      expect(rect.findContainingCorner(100, 200, cornerSize)).toBe('NW')
      expect(rect.findContainingCorner(110, 210, cornerSize)).toBe('NW')
      expect(rect.findContainingCorner(114, 214, cornerSize)).toBe('NW')
    })

    test('should detect NE (top-right) corner', () => {
      // Top-right corner starts at (right - cornerSize, top) = (385, 200)
      expect(rect.findContainingCorner(385, 200, cornerSize)).toBe('NE')
      expect(rect.findContainingCorner(390, 210, cornerSize)).toBe('NE')
      expect(rect.findContainingCorner(399, 214, cornerSize)).toBe('NE')
    })

    test('should detect SW (bottom-left) corner', () => {
      // Bottom-left corner starts at (left, bottom - cornerSize) = (100, 585)
      expect(rect.findContainingCorner(100, 585, cornerSize)).toBe('SW')
      expect(rect.findContainingCorner(110, 590, cornerSize)).toBe('SW')
      expect(rect.findContainingCorner(114, 599, cornerSize)).toBe('SW')
    })

    test('should detect SE (bottom-right) corner', () => {
      // Bottom-right corner starts at (right - cornerSize, bottom - cornerSize) = (385, 585)
      expect(rect.findContainingCorner(385, 585, cornerSize)).toBe('SE')
      expect(rect.findContainingCorner(390, 590, cornerSize)).toBe('SE')
      expect(rect.findContainingCorner(399, 599, cornerSize)).toBe('SE')
    })

    test('should return undefined when not in any corner', () => {
      // Middle of rectangle
      expect(rect.findContainingCorner(250, 400, cornerSize)).toBeUndefined()
      // On edge but not in corner
      expect(rect.findContainingCorner(200, 200, cornerSize)).toBeUndefined()
      expect(rect.findContainingCorner(100, 400, cornerSize)).toBeUndefined()
      // Outside rectangle
      expect(rect.findContainingCorner(50, 150, cornerSize)).toBeUndefined()
    })
  })

  describe('corner detection methods', () => {
    const cornerSize = 20

    describe('isInTopLeftCorner', () => {
      test('should return true when point is in top-left corner', () => {
        expect(rect.isInTopLeftCorner(100, 200, cornerSize)).toBe(true)
        expect(rect.isInTopLeftCorner(110, 210, cornerSize)).toBe(true)
        expect(rect.isInTopLeftCorner(119, 219, cornerSize)).toBe(true)
      })

      test('should return false when point is outside top-left corner', () => {
        expect(rect.isInTopLeftCorner(120, 200, cornerSize)).toBe(false)
        expect(rect.isInTopLeftCorner(100, 220, cornerSize)).toBe(false)
        expect(rect.isInTopLeftCorner(99, 200, cornerSize)).toBe(false)
        expect(rect.isInTopLeftCorner(100, 199, cornerSize)).toBe(false)
      })
    })

    describe('isInTopRightCorner', () => {
      test('should return true when point is in top-right corner', () => {
        // Top-right corner area is from (right - cornerSize, top) to (right, top + cornerSize)
        // That's (380, 200) to (400, 220)
        expect(rect.isInTopRightCorner(380, 200, cornerSize)).toBe(true)
        expect(rect.isInTopRightCorner(390, 210, cornerSize)).toBe(true)
        expect(rect.isInTopRightCorner(399, 219, cornerSize)).toBe(true)
      })

      test('should return false when point is outside top-right corner', () => {
        expect(rect.isInTopRightCorner(379, 200, cornerSize)).toBe(false)
        expect(rect.isInTopRightCorner(400, 220, cornerSize)).toBe(false)
        expect(rect.isInTopRightCorner(401, 200, cornerSize)).toBe(false)
        expect(rect.isInTopRightCorner(400, 199, cornerSize)).toBe(false)
      })
    })

    describe('isInBottomLeftCorner', () => {
      test('should return true when point is in bottom-left corner', () => {
        // Bottom-left corner area is from (left, bottom - cornerSize) to (left + cornerSize, bottom)
        // That's (100, 580) to (120, 600)
        expect(rect.isInBottomLeftCorner(100, 580, cornerSize)).toBe(true)
        expect(rect.isInBottomLeftCorner(110, 590, cornerSize)).toBe(true)
        expect(rect.isInBottomLeftCorner(119, 599, cornerSize)).toBe(true)
      })

      test('should return false when point is outside bottom-left corner', () => {
        expect(rect.isInBottomLeftCorner(120, 600, cornerSize)).toBe(false)
        expect(rect.isInBottomLeftCorner(100, 579, cornerSize)).toBe(false)
        expect(rect.isInBottomLeftCorner(99, 600, cornerSize)).toBe(false)
        expect(rect.isInBottomLeftCorner(100, 601, cornerSize)).toBe(false)
      })
    })

    describe('isInBottomRightCorner', () => {
      test('should return true when point is in bottom-right corner', () => {
        // Bottom-right corner area is from (right - cornerSize, bottom - cornerSize) to (right, bottom)
        // That's (380, 580) to (400, 600)
        expect(rect.isInBottomRightCorner(380, 580, cornerSize)).toBe(true)
        expect(rect.isInBottomRightCorner(390, 590, cornerSize)).toBe(true)
        expect(rect.isInBottomRightCorner(399, 599, cornerSize)).toBe(true)
      })

      test('should return false when point is outside bottom-right corner', () => {
        expect(rect.isInBottomRightCorner(379, 600, cornerSize)).toBe(false)
        expect(rect.isInBottomRightCorner(400, 579, cornerSize)).toBe(false)
        expect(rect.isInBottomRightCorner(401, 600, cornerSize)).toBe(false)
        expect(rect.isInBottomRightCorner(400, 601, cornerSize)).toBe(false)
      })
    })
  })

  describe('edge cases', () => {
    test('should handle zero-sized corner areas', () => {
      expect(rect.findContainingCorner(100, 200, 0)).toBeUndefined()
      expect(rect.isInTopLeftCorner(100, 200, 0)).toBe(false)
    })

    test('should handle rectangles at origin', () => {
      const originRect = new Rectangle(0, 0, 100, 100)
      expect(originRect.findContainingCorner(0, 0, 10)).toBe('NW')
      // Bottom-right corner is at (90, 90) to (100, 100)
      expect(originRect.findContainingCorner(90, 90, 10)).toBe('SE')
    })

    test('should handle negative coordinates', () => {
      const negRect = new Rectangle(-50, -50, 100, 100)
      expect(negRect.findContainingCorner(-50, -50, 10)).toBe('NW')
      // Bottom-right corner is at (40, 40) to (50, 50)
      expect(negRect.findContainingCorner(40, 40, 10)).toBe('SE')
    })
  })
})
