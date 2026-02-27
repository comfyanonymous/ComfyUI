import { beforeEach, describe, expect } from 'vitest'

import { LGraphNode, LiteGraph } from '@/lib/litegraph/src/litegraph'

import { test } from './__fixtures__/testExtensions'

describe('LGraphNode resize functionality', () => {
  let node: LGraphNode

  beforeEach(() => {
    // Set up LiteGraph constants needed for measure
    LiteGraph.NODE_TITLE_HEIGHT = 20

    node = new LGraphNode('Test Node')
    node.pos = [100, 100]
    node.size = [200, 150]

    // Create a mock canvas context for updateArea
    const mockCtx = {} as CanvasRenderingContext2D

    // Call updateArea to populate boundingRect
    node.updateArea(mockCtx)
  })

  describe('findResizeDirection', () => {
    describe('corners', () => {
      test('should detect NW (top-left) corner', () => {
        // With title bar, top is at y=80 (100 - 20)
        // Corner is from (100, 80) to (100 + 15, 80 + 15)
        expect(node.findResizeDirection(100, 80)).toBe('NW')
        expect(node.findResizeDirection(110, 90)).toBe('NW')
        expect(node.findResizeDirection(114, 94)).toBe('NW')
      })

      test('should detect NE (top-right) corner', () => {
        // Corner is from (300 - 15, 80) to (300, 80 + 15)
        expect(node.findResizeDirection(285, 80)).toBe('NE')
        expect(node.findResizeDirection(290, 90)).toBe('NE')
        expect(node.findResizeDirection(299, 94)).toBe('NE')
      })

      test('should detect SW (bottom-left) corner', () => {
        // Bottom is at y=250 (100 + 150)
        // Corner is from (100, 250 - 15) to (100 + 15, 250)
        expect(node.findResizeDirection(100, 235)).toBe('SW')
        expect(node.findResizeDirection(110, 240)).toBe('SW')
        expect(node.findResizeDirection(114, 249)).toBe('SW')
      })

      test('should detect SE (bottom-right) corner', () => {
        // Corner is from (300 - 15, 250 - 15) to (300, 250)
        expect(node.findResizeDirection(285, 235)).toBe('SE')
        expect(node.findResizeDirection(290, 240)).toBe('SE')
        expect(node.findResizeDirection(299, 249)).toBe('SE')
      })
    })

    describe('priority', () => {
      test('corners should have priority over edges', () => {
        // These points are technically on both corner and edge
        // Corner should win
        expect(node.findResizeDirection(100, 84)).toBe('NW') // Not "W"
        expect(node.findResizeDirection(104, 80)).toBe('NW') // Not "N"
      })
    })

    describe('negative cases', () => {
      test('should return undefined when outside node bounds', () => {
        expect(node.findResizeDirection(50, 50)).toBeUndefined()
        expect(node.findResizeDirection(350, 300)).toBeUndefined()
        expect(node.findResizeDirection(99, 150)).toBeUndefined()
        expect(node.findResizeDirection(301, 150)).toBeUndefined()
      })

      test('should return undefined when inside node but not on resize areas', () => {
        // Center of node (accounting for title bar offset)
        expect(node.findResizeDirection(200, 165)).toBeUndefined()
        // Just inside the edge threshold
        expect(node.findResizeDirection(106, 150)).toBeUndefined()
        expect(node.findResizeDirection(294, 150)).toBeUndefined()
        expect(node.findResizeDirection(150, 86)).toBeUndefined() // 80 + 6
        expect(node.findResizeDirection(150, 244)).toBeUndefined()
      })

      test('should return undefined when node is not resizable', () => {
        node.resizable = false
        expect(node.findResizeDirection(100, 100)).toBeUndefined()
        expect(node.findResizeDirection(300, 250)).toBeUndefined()
        expect(node.findResizeDirection(150, 100)).toBeUndefined()
      })
    })

    describe('edge cases', () => {
      test('should handle nodes at origin', () => {
        node.pos = [0, 0]
        node.size = [100, 100]

        // Update boundingRect with new position/size
        const mockCtx = {} as CanvasRenderingContext2D
        node.updateArea(mockCtx)

        expect(node.findResizeDirection(0, -20)).toBe('NW') // Account for title bar
        expect(node.findResizeDirection(99, 99)).toBe('SE') // Bottom-right corner (100-1, 100-1)
      })

      test('should handle very small nodes', () => {
        node.size = [20, 20] // Smaller than corner size

        // Update boundingRect with new size
        const mockCtx = {} as CanvasRenderingContext2D
        node.updateArea(mockCtx)

        // Corners still work (accounting for title bar offset)
        expect(node.findResizeDirection(100, 80)).toBe('NW')
        expect(node.findResizeDirection(119, 119)).toBe('SE')
      })
    })
  })

  describe('resizeEdgeSize static property', () => {
    test('should have default value of 5', () => {
      expect(LGraphNode.resizeEdgeSize).toBe(5)
    })
  })

  describe('resizeHandleSize static property', () => {
    test('should have default value of 15', () => {
      expect(LGraphNode.resizeHandleSize).toBe(15)
    })
  })
})
