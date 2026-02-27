// TODO: Fix these tests after migration
import { test as baseTest } from 'vitest'

import type { Point, Rect } from '@/lib/litegraph/src/interfaces'
import {
  addDirectionalOffset,
  containsCentre,
  containsRect,
  createBounds,
  dist2,
  distance,
  findPointOnCurve,
  getOrientation,
  isInRect,
  isInRectangle,
  isInsideRectangle,
  isPointInRect,
  overlapBounding,
  rotateLink,
  snapPoint
} from '@/lib/litegraph/src/measure'
import { LinkDirection } from '@/lib/litegraph/src/types/globalEnums'

const test = baseTest.extend({})

test('distance calculates correct distance between two points', ({
  expect
}) => {
  expect(distance([0, 0], [3, 4])).toBe(5) // 3-4-5 triangle
  expect(distance([1, 1], [4, 5])).toBe(5) // Same triangle, shifted
  expect(distance([0, 0], [0, 0])).toBe(0) // Same point
})

test('dist2 calculates squared distance between points', ({ expect }) => {
  expect(dist2(0, 0, 3, 4)).toBe(25) // 3-4-5 triangle squared
  expect(dist2(1, 1, 4, 5)).toBe(25) // Same triangle, shifted
  expect(dist2(0, 0, 0, 0)).toBe(0) // Same point
})

test('isInRectangle correctly identifies points inside rectangle', ({
  expect
}) => {
  // Test points inside
  expect(isInRectangle(5, 5, 0, 0, 10, 10)).toBe(true)
  // Test points on edges (should be true)
  expect(isInRectangle(0, 5, 0, 0, 10, 10)).toBe(true)
  expect(isInRectangle(5, 0, 0, 0, 10, 10)).toBe(true)
  // Test points outside
  expect(isInRectangle(-1, 5, 0, 0, 10, 10)).toBe(false)
  expect(isInRectangle(11, 5, 0, 0, 10, 10)).toBe(false)
})

test('isPointInRect correctly identifies points inside rectangle', ({
  expect
}) => {
  const rect: Rect = [0, 0, 10, 10]
  expect(isPointInRect([5, 5], rect)).toBe(true)
  expect(isPointInRect([-1, 5], rect)).toBe(false)
})

test('overlapBounding correctly identifies overlapping rectangles', ({
  expect
}) => {
  const rect1: Rect = [0, 0, 10, 10]
  const rect2: Rect = [5, 5, 10, 10]
  const rect3: Rect = [20, 20, 10, 10]

  expect(overlapBounding(rect1, rect2)).toBe(true)
  expect(overlapBounding(rect1, rect3)).toBe(false)
})

test('containsCentre correctly identifies if rectangle contains center of another', ({
  expect
}) => {
  const container: Rect = [0, 0, 20, 20]
  const inside: Rect = [5, 5, 10, 10] // Center at 10,10
  const outside: Rect = [15, 15, 10, 10] // Center at 20,20

  expect(containsCentre(container, inside)).toBe(true)
  expect(containsCentre(container, outside)).toBe(false)
})

test('addDirectionalOffset correctly adds offsets', ({ expect }) => {
  const point: Point = [10, 10]

  // Test each direction
  addDirectionalOffset(5, LinkDirection.RIGHT, point)
  expect(point).toEqual([15, 10])

  point[0] = 10 // Reset X
  addDirectionalOffset(5, LinkDirection.LEFT, point)
  expect(point).toEqual([5, 10])

  point[0] = 10 // Reset X
  addDirectionalOffset(5, LinkDirection.DOWN, point)
  expect(point).toEqual([10, 15])

  point[1] = 10 // Reset Y
  addDirectionalOffset(5, LinkDirection.UP, point)
  expect(point).toEqual([10, 5])
})

test('findPointOnCurve correctly interpolates curve points', ({ expect }) => {
  const out: Point = [0, 0]
  const start: Point = [0, 0]
  const end: Point = [10, 10]
  const controlA: Point = [0, 10]
  const controlB: Point = [10, 0]

  // Test midpoint
  findPointOnCurve(out, start, end, controlA, controlB, 0.5)
  expect(out[0]).toBeCloseTo(5)
  expect(out[1]).toBeCloseTo(5)
})

test('snapPoint correctly snaps points to grid', ({ expect }) => {
  const point: Point = [12.3, 18.7]

  // Snap to 5
  snapPoint(point, 5)
  expect(point).toEqual([10, 20])

  // Test with no snap
  const point2: Point = [12.3, 18.7]
  expect(snapPoint(point2, 0)).toBe(false)
  expect(point2).toEqual([12.3, 18.7])

  const point3: Point = [15, 24.499]
  expect(snapPoint(point3, 10)).toBe(true)
  expect(point3).toEqual([20, 20])
})

test('createBounds correctly creates bounding box', ({ expect }) => {
  const objects = [
    { boundingRect: [0, 0, 10, 10] as Rect },
    { boundingRect: [5, 5, 10, 10] as Rect }
  ]

  const defaultBounds = createBounds(objects)
  expect(defaultBounds).toEqual([-10, -10, 35, 35])

  const bounds = createBounds(objects, 5)
  expect(bounds).toEqual([-5, -5, 25, 25])

  // Test empty set
  expect(createBounds([])).toBe(null)
})

test('isInsideRectangle handles edge cases differently from isInRectangle', ({
  expect
}) => {
  // isInsideRectangle returns false when point is exactly on left or top edge
  expect(isInsideRectangle(0, 5, 0, 0, 10, 10)).toBe(false)
  expect(isInsideRectangle(5, 0, 0, 0, 10, 10)).toBe(false)

  // Points just inside
  expect(isInsideRectangle(0.1, 5, 0, 0, 10, 10)).toBe(true)
  expect(isInsideRectangle(5, 0.1, 0, 0, 10, 10)).toBe(true)

  // Points clearly inside
  expect(isInsideRectangle(5, 5, 0, 0, 10, 10)).toBe(true)

  // Points outside
  expect(isInsideRectangle(-1, 5, 0, 0, 10, 10)).toBe(false)
  expect(isInsideRectangle(11, 5, 0, 0, 10, 10)).toBe(false)
})

test('containsRect correctly identifies nested rectangles', ({ expect }) => {
  const container: Rect = [0, 0, 20, 20]

  // Fully contained rectangle
  const inside: Rect = [5, 5, 10, 10]
  expect(containsRect(container, inside)).toBe(true)

  // Partially overlapping rectangle
  const partial: Rect = [15, 15, 10, 10]
  expect(containsRect(container, partial)).toBe(false)

  // Completely outside rectangle
  const outside: Rect = [30, 30, 10, 10]
  expect(containsRect(container, outside)).toBe(false)

  // Same size rectangle at same position (should return false)
  const identical: Rect = [0, 0, 20, 20]
  expect(containsRect(container, identical)).toBe(false)

  // Larger rectangle (should return false)
  const larger: Rect = [-5, -5, 30, 30]
  expect(containsRect(container, larger)).toBe(false)
})

test('rotateLink correctly rotates offsets between directions', ({
  expect
}) => {
  const testCases = [
    {
      offset: [10, 5] as Point,
      from: LinkDirection.LEFT,
      to: LinkDirection.RIGHT,
      expected: [-10, -5]
    },
    {
      offset: [10, 5] as Point,
      from: LinkDirection.LEFT,
      to: LinkDirection.UP,
      expected: [5, -10]
    },
    {
      offset: [10, 5] as Point,
      from: LinkDirection.LEFT,
      to: LinkDirection.DOWN,
      expected: [-5, 10]
    },
    {
      offset: [10, 5] as Point,
      from: LinkDirection.RIGHT,
      to: LinkDirection.LEFT,
      expected: [-10, -5]
    },
    {
      offset: [10, 5] as Point,
      from: LinkDirection.UP,
      to: LinkDirection.DOWN,
      expected: [-10, -5]
    }
  ]

  for (const { offset, from, to, expected } of testCases) {
    const testOffset = [...offset] as Point
    rotateLink(testOffset, from, to)
    expect(testOffset).toEqual(expected)
  }

  // Test no rotation when directions are the same
  const sameDir = [10, 5] as Point
  rotateLink(sameDir, LinkDirection.LEFT, LinkDirection.LEFT)
  expect(sameDir).toEqual([10, 5])

  // Test center/none cases
  const centerCase = [10, 5] as Point
  rotateLink(centerCase, LinkDirection.LEFT, LinkDirection.CENTER)
  expect(centerCase).toEqual([10, 5])

  const noneCase = [10, 5] as Point
  rotateLink(noneCase, LinkDirection.LEFT, LinkDirection.NONE)
  expect(noneCase).toEqual([10, 5])
})

test('getOrientation correctly determines point position relative to line', ({
  expect
}) => {
  const lineStart: Point = [0, 0]
  const lineEnd: Point = [10, 10]

  // Point to the left of the line
  expect(getOrientation(lineStart, lineEnd, 0, 10)).toBeLessThan(0)

  // Point to the right of the line
  expect(getOrientation(lineStart, lineEnd, 10, 0)).toBeGreaterThan(0)

  // Point on the line
  expect(getOrientation(lineStart, lineEnd, 5, 5)).toBe(0)

  // Test with horizontal line
  const hLineEnd: Point = [10, 0]
  expect(getOrientation(lineStart, hLineEnd, 5, 5)).toBeLessThan(0) // Above line
  expect(getOrientation(lineStart, hLineEnd, 5, -5)).toBeGreaterThan(0) // Below line

  // Test with vertical line
  const vLineEnd: Point = [0, 10]
  expect(getOrientation(lineStart, vLineEnd, 5, 5)).toBeGreaterThan(0) // Right of line
  expect(getOrientation(lineStart, vLineEnd, -5, 5)).toBeLessThan(0) // Left of line
})

test('isInRect correctly identifies if point coordinates are inside rectangle', ({
  expect
}) => {
  const rect: Rect = [0, 0, 10, 10]

  // Points inside
  expect(isInRect(5, 5, rect)).toBe(true)

  // Points on edges (should be true for left/top, false for right/bottom)
  expect(isInRect(0, 5, rect)).toBe(true) // Left edge
  expect(isInRect(5, 0, rect)).toBe(true) // Top edge
  expect(isInRect(10, 5, rect)).toBe(false) // Right edge
  expect(isInRect(5, 10, rect)).toBe(false) // Bottom edge

  // Points at corners
  expect(isInRect(0, 0, rect)).toBe(true) // Top-left
  expect(isInRect(10, 0, rect)).toBe(false) // Top-right
  expect(isInRect(0, 10, rect)).toBe(false) // Bottom-left
  expect(isInRect(10, 10, rect)).toBe(false) // Bottom-right

  // Points outside
  expect(isInRect(-1, 5, rect)).toBe(false)
  expect(isInRect(11, 5, rect)).toBe(false)
  expect(isInRect(5, -1, rect)).toBe(false)
  expect(isInRect(5, 11, rect)).toBe(false)
})
