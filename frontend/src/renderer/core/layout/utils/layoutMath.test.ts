import { describe, expect, it } from 'vitest'

import type { NodeLayout } from '@/renderer/core/layout/types'
import {
  REROUTE_RADIUS,
  boundsIntersect,
  calculateBounds,
  pointInBounds
} from '@/renderer/core/layout/utils/layoutMath'

describe('layoutMath utils', () => {
  describe('pointInBounds', () => {
    it('detects inclusion correctly', () => {
      const bounds = { x: 10, y: 10, width: 100, height: 50 }
      expect(pointInBounds({ x: 10, y: 10 }, bounds)).toBe(true)
      expect(pointInBounds({ x: 110, y: 60 }, bounds)).toBe(true)
      expect(pointInBounds({ x: 9, y: 10 }, bounds)).toBe(false)
      expect(pointInBounds({ x: 111, y: 10 }, bounds)).toBe(false)
      expect(pointInBounds({ x: 10, y: 61 }, bounds)).toBe(false)
    })

    it('works with zero-size bounds', () => {
      const zero = { x: 10, y: 20, width: 0, height: 0 }
      expect(pointInBounds({ x: 10, y: 20 }, zero)).toBe(true)
      expect(pointInBounds({ x: 10, y: 21 }, zero)).toBe(false)
      expect(pointInBounds({ x: 9, y: 20 }, zero)).toBe(false)
    })
  })

  describe('boundsIntersect', () => {
    it('detects intersection correctly', () => {
      const a = { x: 0, y: 0, width: 10, height: 10 }
      const b = { x: 5, y: 5, width: 10, height: 10 }
      const c = { x: 11, y: 0, width: 5, height: 5 }
      expect(boundsIntersect(a, b)).toBe(true)
      expect(boundsIntersect(a, c)).toBe(false)
    })

    it('treats touching edges as intersecting', () => {
      const a = { x: 0, y: 0, width: 10, height: 10 }
      const d = { x: 10, y: 0, width: 5, height: 5 } // touches at right edge
      expect(boundsIntersect(a, d)).toBe(true)
    })
  })

  describe('REROUTE_RADIUS', () => {
    it('exports a sensible reroute radius', () => {
      expect(REROUTE_RADIUS).toBeGreaterThan(0)
    })
  })

  describe('calculateBounds', () => {
    const createTestNode = (
      id: string,
      x: number,
      y: number,
      width: number,
      height: number
    ): NodeLayout => ({
      id,
      position: { x, y },
      size: { width, height },
      zIndex: 0,
      visible: true,
      bounds: { x, y, width, height }
    })

    it('calculates bounds for single node', () => {
      const nodes = [createTestNode('1', 10, 20, 100, 50)]
      const bounds = calculateBounds(nodes)

      expect(bounds).toEqual({
        x: 10,
        y: 20,
        width: 100,
        height: 50
      })
    })

    it('calculates combined bounds for multiple nodes', () => {
      const nodes = [
        createTestNode('1', 0, 0, 50, 50), // Top-left: (0,0) to (50,50)
        createTestNode('2', 100, 100, 30, 40), // Bottom-right: (100,100) to (130,140)
        createTestNode('3', 25, 75, 20, 10) // Middle: (25,75) to (45,85)
      ]
      const bounds = calculateBounds(nodes)

      expect(bounds).toEqual({
        x: 0, // leftmost
        y: 0, // topmost
        width: 130, // rightmost (130) - leftmost (0)
        height: 140 // bottommost (140) - topmost (0)
      })
    })

    it('handles nodes with negative positions', () => {
      const nodes = [
        createTestNode('1', -50, -30, 40, 20), // (-50,-30) to (-10,-10)
        createTestNode('2', 10, 15, 25, 35) // (10,15) to (35,50)
      ]
      const bounds = calculateBounds(nodes)

      expect(bounds).toEqual({
        x: -50,
        y: -30,
        width: 85, // 35 - (-50)
        height: 80 // 50 - (-30)
      })
    })

    it('handles empty array', () => {
      const bounds = calculateBounds([])

      expect(bounds).toEqual({
        x: Infinity,
        y: Infinity,
        width: -Infinity,
        height: -Infinity
      })
    })
  })
})
