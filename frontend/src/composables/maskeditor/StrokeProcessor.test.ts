import { describe, it, expect } from 'vitest'
import { StrokeProcessor } from './StrokeProcessor'
import type { Point } from '@/extensions/core/maskeditor/types'

describe('StrokeProcessor', () => {
  it('should generate equidistant points from irregular input', () => {
    const spacing = 10
    const processor = new StrokeProcessor(spacing)
    const outputPoints: Point[] = []

    // Simulate a horizontal line drawn with irregular speed
    // Points: (0,0) -> (5,0) -> (25,0) -> (30,0) -> (100,0)
    const inputPoints: Point[] = [
      { x: 0, y: 0 },
      { x: 5, y: 0 }, // dist 5
      { x: 25, y: 0 }, // dist 20
      { x: 30, y: 0 }, // dist 5
      { x: 100, y: 0 } // dist 70
    ]

    for (const p of inputPoints) {
      outputPoints.push(...processor.addPoint(p))
    }
    outputPoints.push(...processor.endStroke())

    // Verify we have points
    expect(outputPoints.length).toBeGreaterThan(0)

    // Verify spacing
    // Note: The first few points might be affected by the start condition,
    // but the middle section should be perfectly spaced.
    // Also, Catmull-Rom splines don't necessarily pass through control points in a straight line
    // if the points are collinear, they should be straight.

    // Let's check distances between consecutive points
    const distances: number[] = []
    for (let i = 1; i < outputPoints.length; i++) {
      const dx = outputPoints[i].x - outputPoints[i - 1].x
      const dy = outputPoints[i].y - outputPoints[i - 1].y
      distances.push(Math.hypot(dx, dy))
    }

    // Check that distances are close to spacing
    // We allow a small epsilon because of floating point and spline approximation
    // Filter out the very last segment which might be shorter (remainder)
    // But wait, our logic doesn't output the last point if it's not a full spacing step?
    // resampleSegment outputs points at [start + spacing, start + 2*spacing, ...]
    // It does NOT output the end point of the segment.
    // So all distances between output points should be exactly `spacing`.
    // EXCEPT possibly if the spline curvature makes the straight-line distance slightly different
    // from the arc length. But for a straight line input, it should be exact.

    // However, catmull-rom with collinear points IS a straight line.

    // Let's log the distances for debugging if test fails
    // console.log('Distances:', distances)

    // All distances should be approximately equal to spacing
    // We might have a gap between segments if the logic isn't perfect,
    // but within a segment it's guaranteed by resampleSegment.
    // The critical part is the transition between segments.

    for (let i = 0; i < distances.length; i++) {
      const d = distances[i]
      if (Math.abs(d - spacing) > 0.5) {
        console.log(
          `Distance mismatch at index ${i}: ${d} (expected ${spacing})`
        )
        console.log(`Point ${i}:`, outputPoints[i])
        console.log(`Point ${i + 1}:`, outputPoints[i + 1])
      }
      expect(d).toBeCloseTo(spacing, 1)
    }
  })

  it('should handle a simple 3-point stroke', () => {
    const spacing = 5
    const processor = new StrokeProcessor(spacing)
    const points: Point[] = []

    points.push(...processor.addPoint({ x: 0, y: 0 }))
    points.push(...processor.addPoint({ x: 10, y: 0 }))
    points.push(...processor.addPoint({ x: 20, y: 0 }))
    points.push(...processor.endStroke())

    expect(points.length).toBeGreaterThan(0)

    // Check distances
    for (let i = 1; i < points.length; i++) {
      const dx = points[i].x - points[i - 1].x
      const dy = points[i].y - points[i - 1].y
      const d = Math.hypot(dx, dy)
      expect(d).toBeCloseTo(spacing, 1)
    }
  })

  it('should handle a single point click', () => {
    const spacing = 5
    const processor = new StrokeProcessor(spacing)
    const points: Point[] = []

    points.push(...processor.addPoint({ x: 100, y: 100 }))
    points.push(...processor.endStroke())

    expect(points.length).toBe(1)
    expect(points[0]).toEqual({ x: 100, y: 100 })
  })
})
