import { describe, it, expect } from 'vitest'
import { resampleSegment } from './splineUtils'
import type { Point } from '@/extensions/core/maskeditor/types'

describe('Shift+Click Drawing Logic', () => {
  it('should generate equidistant points across connected segments', () => {
    const spacing = 4
    let remainder = spacing // Simulate start point already painted
    const outputPoints: Point[] = []

    // Define points: A -> B -> C
    // A(0,0) -> B(10,0) -> C(20,0)
    // Total length 20. Spacing 4.
    // Expected points at x = 4, 8, 12, 16, 20
    const pA = { x: 0, y: 0 }
    const pB = { x: 10, y: 0 }
    const pC = { x: 20, y: 0 }

    // Segment 1: A -> B
    const result1 = resampleSegment([pA, pB], spacing, remainder)
    outputPoints.push(...result1.points)
    remainder = result1.remainder

    // Verify intermediate state
    // Length 10. Spacing 4. Start offset 4.
    // Points at 4, 8. Next at 12.
    // Remainder = 12 - 10 = 2.
    expect(result1.points.length).toBe(2)
    expect(result1.points[0].x).toBeCloseTo(4)
    expect(result1.points[1].x).toBeCloseTo(8)
    expect(remainder).toBeCloseTo(2)

    // Segment 2: B -> C
    const result2 = resampleSegment([pB, pC], spacing, remainder)
    outputPoints.push(...result2.points)
    remainder = result2.remainder

    // Verify final state
    // Start offset 2. Points at 2, 6, 10 (relative to B).
    // Absolute x: 12, 16, 20.
    expect(result2.points.length).toBe(3)
    expect(result2.points[0].x).toBeCloseTo(12)
    expect(result2.points[1].x).toBeCloseTo(16)
    expect(result2.points[2].x).toBeCloseTo(20)

    // Verify all distances
    // Note: The first point is at distance `spacing` from start (0,0)
    // Subsequent points are `spacing` apart.
    let prevX = 0
    for (const p of outputPoints) {
      const dist = p.x - prevX
      expect(dist).toBeCloseTo(spacing)
      prevX = p.x
    }
  })

  it('should handle segments shorter than spacing', () => {
    const spacing = 10
    let remainder = spacing // Simulate start point already painted

    // A(0,0) -> B(5,0) -> C(15,0)
    const pA = { x: 0, y: 0 }
    const pB = { x: 5, y: 0 }
    const pC = { x: 15, y: 0 }

    // Segment 1: A -> B (Length 5)
    // Spacing 10. No points should be generated.
    // Remainder should be 5 (next point needs 5 more units).
    const result1 = resampleSegment([pA, pB], spacing, remainder)
    expect(result1.points.length).toBe(0)
    expect(result1.remainder).toBeCloseTo(5)
    remainder = result1.remainder

    // Segment 2: B -> C (Length 10)
    // Start offset 5. First point at 5 (relative to B).
    // Absolute x = 10.
    // Next point at 15 (relative to B). Segment ends at 10.
    // Remainder = 15 - 10 = 5.
    const result2 = resampleSegment([pB, pC], spacing, remainder)
    expect(result2.points.length).toBe(1)
    expect(result2.points[0].x).toBeCloseTo(10)
    expect(result2.remainder).toBeCloseTo(5)
  })
})
