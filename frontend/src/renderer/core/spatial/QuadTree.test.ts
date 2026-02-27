import { beforeEach, describe, expect, it } from 'vitest'

import { QuadTree } from '@/renderer/core/spatial/QuadTree'
import type { Bounds } from '@/renderer/core/spatial/QuadTree'

describe('QuadTree', () => {
  let quadTree: QuadTree<string>
  const worldBounds: Bounds = { x: 0, y: 0, width: 1000, height: 1000 }

  beforeEach(() => {
    quadTree = new QuadTree<string>(worldBounds, {
      maxDepth: 4,
      maxItemsPerNode: 4
    })
  })

  describe('insertion', () => {
    it('should insert items within bounds', () => {
      const success = quadTree.insert(
        'node1',
        { x: 100, y: 100, width: 50, height: 50 },
        'node1'
      )
      expect(success).toBe(true)
      expect(quadTree.size).toBe(1)
    })

    it('should reject items outside bounds', () => {
      const success = quadTree.insert(
        'node1',
        { x: -100, y: -100, width: 50, height: 50 },
        'node1'
      )
      expect(success).toBe(false)
      expect(quadTree.size).toBe(0)
    })

    it('should handle duplicate IDs by replacing', () => {
      quadTree.insert(
        'node1',
        { x: 100, y: 100, width: 50, height: 50 },
        'data1'
      )
      quadTree.insert(
        'node1',
        { x: 200, y: 200, width: 50, height: 50 },
        'data2'
      )

      expect(quadTree.size).toBe(1)
      const results = quadTree.query({
        x: 150,
        y: 150,
        width: 100,
        height: 100
      })
      expect(results).toContain('data2')
      expect(results).not.toContain('data1')
    })
  })

  describe('querying', () => {
    beforeEach(() => {
      // Insert test nodes in a grid pattern
      for (let x = 0; x < 10; x++) {
        for (let y = 0; y < 10; y++) {
          const id = `node_${x}_${y}`
          quadTree.insert(
            id,
            {
              x: x * 100,
              y: y * 100,
              width: 50,
              height: 50
            },
            id
          )
        }
      }
    })

    it('should find nodes within query bounds', () => {
      const results = quadTree.query({ x: 0, y: 0, width: 250, height: 250 })
      expect(results.length).toBe(9) // 3x3 grid
    })

    it('should return empty array for out-of-bounds query', () => {
      const results = quadTree.query({
        x: 2000,
        y: 2000,
        width: 100,
        height: 100
      })
      expect(results.length).toBe(0)
    })

    it('should handle partial overlaps', () => {
      const results = quadTree.query({ x: 25, y: 25, width: 100, height: 100 })
      expect(results.length).toBe(4) // 2x2 grid due to overlap
    })

    it('should handle large query areas efficiently', () => {
      const startTime = performance.now()
      const results = quadTree.query({ x: 0, y: 0, width: 1000, height: 1000 })
      const queryTime = performance.now() - startTime

      expect(results.length).toBe(100) // All nodes
      expect(queryTime).toBeLessThan(5) // Should be fast
    })
  })

  describe('removal', () => {
    it('should remove existing items', () => {
      quadTree.insert(
        'node1',
        { x: 100, y: 100, width: 50, height: 50 },
        'node1'
      )
      expect(quadTree.size).toBe(1)

      const success = quadTree.remove('node1')
      expect(success).toBe(true)
      expect(quadTree.size).toBe(0)
    })

    it('should handle removal of non-existent items', () => {
      const success = quadTree.remove('nonexistent')
      expect(success).toBe(false)
    })
  })

  describe('updating', () => {
    it('should update item position', () => {
      quadTree.insert(
        'node1',
        { x: 100, y: 100, width: 50, height: 50 },
        'node1'
      )

      const success = quadTree.update('node1', {
        x: 200,
        y: 200,
        width: 50,
        height: 50
      })
      expect(success).toBe(true)

      // Should not find at old position
      const oldResults = quadTree.query({
        x: 75,
        y: 75,
        width: 100,
        height: 100
      })
      expect(oldResults).not.toContain('node1')

      // Should find at new position
      const newResults = quadTree.query({
        x: 175,
        y: 175,
        width: 100,
        height: 100
      })
      expect(newResults).toContain('node1')
    })
  })

  describe('subdivision', () => {
    it('should subdivide when exceeding max items', () => {
      // Insert 5 items (max is 4) to trigger subdivision
      for (let i = 0; i < 5; i++) {
        quadTree.insert(
          `node${i}`,
          {
            x: i * 10,
            y: i * 10,
            width: 5,
            height: 5
          },
          `node${i}`
        )
      }

      expect(quadTree.size).toBe(5)

      // Verify all items can still be found
      const allResults = quadTree.query(worldBounds)
      expect(allResults.length).toBe(5)
    })
  })

  describe('performance', () => {
    it('should handle 1000 nodes efficiently', () => {
      const insertStart = performance.now()

      // Insert 1000 nodes
      for (let i = 0; i < 1000; i++) {
        const x = Math.random() * 900
        const y = Math.random() * 900
        quadTree.insert(
          `node${i}`,
          {
            x,
            y,
            width: 50,
            height: 50
          },
          `node${i}`
        )
      }

      const insertTime = performance.now() - insertStart
      expect(insertTime).toBeLessThan(50) // Should be fast

      // Query performance
      const queryStart = performance.now()
      const results = quadTree.query({
        x: 400,
        y: 400,
        width: 200,
        height: 200
      })
      const queryTime = performance.now() - queryStart

      expect(queryTime).toBeLessThan(2) // Queries should be very fast
      expect(results.length).toBeGreaterThan(0)
      expect(results.length).toBeLessThan(1000) // Should cull most nodes
    })
  })

  describe('edge cases', () => {
    it('should handle zero-sized bounds', () => {
      const success = quadTree.insert(
        'point',
        { x: 100, y: 100, width: 0, height: 0 },
        'point'
      )
      expect(success).toBe(true)

      const results = quadTree.query({ x: 99, y: 99, width: 2, height: 2 })
      expect(results).toContain('point')
    })

    it('should handle items spanning multiple quadrants', () => {
      const success = quadTree.insert(
        'large',
        {
          x: 400,
          y: 400,
          width: 200,
          height: 200
        },
        'large'
      )
      expect(success).toBe(true)

      // Should be found when querying any overlapping quadrant
      const topLeft = quadTree.query({ x: 0, y: 0, width: 500, height: 500 })
      const bottomRight = quadTree.query({
        x: 500,
        y: 500,
        width: 500,
        height: 500
      })

      expect(topLeft).toContain('large')
      expect(bottomRight).toContain('large')
    })
  })
})
