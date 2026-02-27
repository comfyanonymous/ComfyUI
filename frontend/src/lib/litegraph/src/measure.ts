import type { HasBoundingRect, Point, ReadOnlyRect, Rect } from './interfaces'
import { Alignment, LinkDirection, hasFlag } from './types/globalEnums'

/**
 * Calculates the distance between two points (2D vector)
 * @param a Point a as `x, y`
 * @param b Point b as `x, y`
 * @returns Distance between point {@link a} & {@link b}
 */
export function distance(a: Readonly<Point>, b: Readonly<Point>): number {
  return Math.sqrt(
    (b[0] - a[0]) * (b[0] - a[0]) + (b[1] - a[1]) * (b[1] - a[1])
  )
}

/**
 * Calculates the distance2 (squared) between two points (2D vector).
 * Much faster when only comparing distances (closest/furthest point).
 * @param x1 Origin point X
 * @param y1 Origin point Y
 * @param x2 Destination point X
 * @param y2 Destination point Y
 * @returns Distance2 (squared) between point [{@link x1}, {@link y1}] & [{@link x2}, {@link y2}]
 */
export function dist2(x1: number, y1: number, x2: number, y2: number): number {
  return (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
}

/**
 * Determines whether a point is inside a rectangle.
 *
 * Otherwise identical to {@link isInsideRectangle}, it also returns `true` if `x` equals `left` or `y` equals `top`.
 * @param x Point x
 * @param y Point y
 * @param left Rect x
 * @param top Rect y
 * @param width Rect width
 * @param height Rect height
 * @returns `true` if the point is inside the rect, otherwise `false`
 */
export function isInRectangle(
  x: number,
  y: number,
  left: number,
  top: number,
  width: number,
  height: number
): boolean {
  return x >= left && x < left + width && y >= top && y < top + height
}

/**
 * Determines whether a {@link Point} is inside a {@link Rect}.
 * @param point The point to check, as `x, y`
 * @param rect The rectangle, as `x, y, width, height`
 * @returns `true` if the point is inside the rect, otherwise `false`
 */
export function isPointInRect(
  point: Readonly<Point>,
  rect: ReadOnlyRect
): boolean {
  return (
    point[0] >= rect[0] &&
    point[0] < rect[0] + rect[2] &&
    point[1] >= rect[1] &&
    point[1] < rect[1] + rect[3]
  )
}

/**
 * Determines whether the point represented by {@link x}, {@link y} is inside a {@link Rect}.
 * @param x X co-ordinate of the point to check
 * @param y Y co-ordinate of the point to check
 * @param rect The rectangle, as `x, y, width, height`
 * @returns `true` if the point is inside the rect, otherwise `false`
 */
export function isInRect(x: number, y: number, rect: ReadOnlyRect): boolean {
  return (
    x >= rect[0] &&
    x < rect[0] + rect[2] &&
    y >= rect[1] &&
    y < rect[1] + rect[3]
  )
}

/**
 * Determines whether a point (`x, y`) is inside a rectangle.
 *
 * This is the original litegraph implementation.  It returns `false` if `x` is equal to `left`, or `y` is equal to `top`.
 * @deprecated
 * Use {@link isInRectangle} to match inclusive of top left.
 * This function returns a false negative when an integer point (e.g. pixel) is on the leftmost or uppermost edge of a rectangle.
 * @param x Point x
 * @param y Point y
 * @param left Rect x
 * @param top Rect y
 * @param width Rect width
 * @param height Rect height
 * @returns `true` if the point is inside the rect, otherwise `false`
 */
export function isInsideRectangle(
  x: number,
  y: number,
  left: number,
  top: number,
  width: number,
  height: number
): boolean {
  return left < x && left + width > x && top < y && top + height > y
}

/**
 * Determines if two rectangles have any overlap
 * @param a Rectangle A as `x, y, width, height`
 * @param b Rectangle B as `x, y, width, height`
 * @returns `true` if rectangles overlap, otherwise `false`
 */
export function overlapBounding(a: ReadOnlyRect, b: ReadOnlyRect): boolean {
  const aRight = a[0] + a[2]
  const aBottom = a[1] + a[3]
  const bRight = b[0] + b[2]
  const bBottom = b[1] + b[3]

  return a[0] > bRight || a[1] > bBottom || aRight < b[0] || aBottom < b[1]
    ? false
    : true
}

/**
 * Returns the centre of a rectangle.
 * @param rect The rectangle, as `x, y, width, height`
 * @returns The centre of the rectangle, as `x, y`
 */
export function getCentre(rect: ReadOnlyRect): Point {
  return [rect[0] + rect[2] * 0.5, rect[1] + rect[3] * 0.5]
}

/**
 * Determines if rectangle {@link a} contains the centre point of rectangle {@link b}.
 * @param a Container rectangle A as `x, y, width, height`
 * @param b Sub-rectangle B as `x, y, width, height`
 * @returns `true` if {@link a} contains most of {@link b}, otherwise `false`
 */
export function containsCentre(a: ReadOnlyRect, b: ReadOnlyRect): boolean {
  const centreX = b[0] + b[2] * 0.5
  const centreY = b[1] + b[3] * 0.5
  return isInRect(centreX, centreY, a)
}

/**
 * Determines if rectangle {@link a} wholly contains rectangle {@link b}.
 * @param a Container rectangle A as `x, y, width, height`
 * @param b Sub-rectangle B as `x, y, width, height`
 * @returns `true` if {@link a} wholly contains {@link b}, otherwise `false`
 */
export function containsRect(a: ReadOnlyRect, b: ReadOnlyRect): boolean {
  const aRight = a[0] + a[2]
  const aBottom = a[1] + a[3]
  const bRight = b[0] + b[2]
  const bBottom = b[1] + b[3]

  const identical =
    a[0] === b[0] && a[1] === b[1] && aRight === bRight && aBottom === bBottom

  return (
    !identical &&
    a[0] <= b[0] &&
    a[1] <= b[1] &&
    aRight >= bRight &&
    aBottom >= bBottom
  )
}

/**
 * Adds an offset in the specified direction to {@link out}
 * @param amount Amount of offset to add
 * @param direction Direction to add the offset to
 * @param out The {@link Point} to add the offset to
 */
export function addDirectionalOffset(
  amount: number,
  direction: LinkDirection,
  out: Point
): void {
  switch (direction) {
    case LinkDirection.LEFT:
      out[0] -= amount
      return
    case LinkDirection.RIGHT:
      out[0] += amount
      return
    case LinkDirection.UP:
      out[1] -= amount
      return
    case LinkDirection.DOWN:
      out[1] += amount
      return
    // LinkDirection.CENTER: Nothing to do.
  }
}

/**
 * Rotates an offset in 90° increments.
 *
 * Swaps/flips axis values of a 2D vector offset - effectively rotating
 * {@link offset} by 90°
 * @param offset The zero-based offset to rotate
 * @param from Direction to rotate from
 * @param to Direction to rotate to
 */
export function rotateLink(
  offset: Point,
  from: LinkDirection,
  to: LinkDirection
): void {
  let x: number
  let y: number

  // Normalise to left
  switch (from) {
    case to:
    case LinkDirection.CENTER:
    case LinkDirection.NONE:
    default:
      // Nothing to do
      return

    case LinkDirection.LEFT:
      x = offset[0]
      y = offset[1]
      break
    case LinkDirection.RIGHT:
      x = -offset[0]
      y = -offset[1]
      break
    case LinkDirection.UP:
      x = -offset[1]
      y = offset[0]
      break
    case LinkDirection.DOWN:
      x = offset[1]
      y = -offset[0]
      break
  }

  // Apply new direction
  switch (to) {
    case LinkDirection.CENTER:
    case LinkDirection.NONE:
      // Nothing to do
      return

    case LinkDirection.LEFT:
      offset[0] = x
      offset[1] = y
      break
    case LinkDirection.RIGHT:
      offset[0] = -x
      offset[1] = -y
      break
    case LinkDirection.UP:
      offset[0] = y
      offset[1] = -x
      break
    case LinkDirection.DOWN:
      offset[0] = -y
      offset[1] = x
      break
  }
}

/**
 * Check if a point is to to the left or right of a line.
 * Project a line from lineStart -> lineEnd.  Determine if point is to the left
 * or right of that projection.
 * {@link https://www.geeksforgeeks.org/orientation-3-ordered-points/}
 * @param lineStart The start point of the line
 * @param lineEnd The end point of the line
 * @param x X co-ordinate of the point to check
 * @param y Y co-ordinate of the point to check
 * @returns 0 if all three points are in a straight line, a negative value if
 * point is to the left of the projected line, or positive if the point is to
 * the right
 */
export function getOrientation(
  lineStart: Readonly<Point>,
  lineEnd: Readonly<Point>,
  x: number,
  y: number
): number {
  return (
    (lineEnd[1] - lineStart[1]) * (x - lineEnd[0]) -
    (lineEnd[0] - lineStart[0]) * (y - lineEnd[1])
  )
}

/**
 * @param out The array to store the point in
 * @param a Start point
 * @param b End point
 * @param controlA Start curve control point
 * @param controlB End curve control point
 * @param t Time: factor of distance to travel along the curve (e.g 0.25 is 25% along the curve)
 */
export function findPointOnCurve(
  out: Point,
  a: Readonly<Point>,
  b: Readonly<Point>,
  controlA: Readonly<Point>,
  controlB: Readonly<Point>,
  t: number = 0.5
): void {
  const iT = 1 - t

  const c1 = iT * iT * iT
  const c2 = 3 * (iT * iT) * t
  const c3 = 3 * iT * (t * t)
  const c4 = t * t * t

  out[0] = c1 * a[0] + c2 * controlA[0] + c3 * controlB[0] + c4 * b[0]
  out[1] = c1 * a[1] + c2 * controlA[1] + c3 * controlB[1] + c4 * b[1]
}

export function createBounds(
  objects: Iterable<HasBoundingRect>,
  padding: number = 10
): ReadOnlyRect | null {
  const bounds: Rect = [Infinity, Infinity, -Infinity, -Infinity]

  for (const obj of objects) {
    const rect = obj.boundingRect
    bounds[0] = Math.min(bounds[0], rect[0])
    bounds[1] = Math.min(bounds[1], rect[1])
    bounds[2] = Math.max(bounds[2], rect[0] + rect[2])
    bounds[3] = Math.max(bounds[3], rect[1] + rect[3])
  }
  if (!bounds.every((x) => isFinite(x))) return null

  return [
    bounds[0] - padding,
    bounds[1] - padding,
    bounds[2] - bounds[0] + 2 * padding,
    bounds[3] - bounds[1] + 2 * padding
  ]
}

/**
 * Snaps the provided {@link Point} or {@link Rect} ({@link pos}) to a grid of size {@link snapTo}.
 * @param pos The point that will be snapped
 * @param snapTo The value to round up/down by (multiples thereof)
 * @returns `true` if snapTo is truthy, otherwise `false`
 * @remarks `NaN` propagates through this function and does not affect return value.
 */
export function snapPoint(pos: Point | Rect, snapTo: number): boolean {
  if (!snapTo) return false

  pos[0] = snapTo * Math.round(pos[0] / snapTo)
  pos[1] = snapTo * Math.round(pos[1] / snapTo)
  return true
}

/**
 * Aligns a {@link Rect} relative to the edges or centre of a {@link container} rectangle.
 *
 * With no {@link inset}, the element will be placed on the interior of the {@link container},
 * with their edges lined up on the {@link anchors}.  A positive {@link inset} moves the element towards the centre,
 * negative will push it outside the {@link container}.
 * @param rect The bounding rect of the element to align.
 * If using the element's pos/size backing store, this function will move the element.
 * @param anchors The direction(s) to anchor the element to
 * @param container The rectangle inside which to align the element
 * @param inset Relative offset from each {@link anchors} edge, with positive always leading to the centre, as an `[x, y]` point
 * @returns The original {@link rect}, modified in place.
 */
export function alignToContainer(
  rect: Rect,
  anchors: Alignment,
  [containerX, containerY, containerWidth, containerHeight]: ReadOnlyRect,
  [insetX, insetY]: Readonly<Point> = [0, 0]
): Rect {
  if (hasFlag(anchors, Alignment.Left)) {
    // Left
    rect[0] = containerX + insetX
  } else if (hasFlag(anchors, Alignment.Right)) {
    // Right
    rect[0] = containerX + containerWidth - insetX - rect[2]
  } else if (hasFlag(anchors, Alignment.Centre)) {
    // Horizontal centre
    rect[0] = containerX + containerWidth * 0.5 - rect[2] * 0.5
  }

  if (hasFlag(anchors, Alignment.Top)) {
    // Top
    rect[1] = containerY + insetY
  } else if (hasFlag(anchors, Alignment.Bottom)) {
    // Bottom
    rect[1] = containerY + containerHeight - insetY - rect[3]
  } else if (hasFlag(anchors, Alignment.Middle)) {
    // Vertical middle
    rect[1] = containerY + containerHeight * 0.5 - rect[3] * 0.5
  }
  return rect
}

/**
 * Aligns a {@link Rect} relative to the edges of {@link other}.
 *
 * With no {@link outset}, the element will be placed on the exterior of the {@link other},
 * with their edges lined up on the {@link anchors}.  A positive {@link outset} moves the element away from the {@link other},
 * negative will push it inside the {@link other}.
 * @param rect The bounding rect of the element to align.
 * If using the element's pos/size backing store, this function will move the element.
 * @param anchors The direction(s) to anchor the element to
 * @param other The rectangle to align {@link rect} to
 * @param outset Relative offset from each {@link anchors} edge, with positive always moving away from the centre of the {@link other}, as an `[x, y]` point
 * @returns The original {@link rect}, modified in place.
 */
export function alignOutsideContainer(
  rect: Rect,
  anchors: Alignment,
  [otherX, otherY, otherWidth, otherHeight]: ReadOnlyRect,
  [outsetX, outsetY]: Readonly<Point> = [0, 0]
): Rect {
  if (hasFlag(anchors, Alignment.Left)) {
    // Left
    rect[0] = otherX - outsetX - rect[2]
  } else if (hasFlag(anchors, Alignment.Right)) {
    // Right
    rect[0] = otherX + otherWidth + outsetX
  } else if (hasFlag(anchors, Alignment.Centre)) {
    // Horizontal centre
    rect[0] = otherX + otherWidth * 0.5 - rect[2] * 0.5
  }

  if (hasFlag(anchors, Alignment.Top)) {
    // Top
    rect[1] = otherY - outsetY - rect[3]
  } else if (hasFlag(anchors, Alignment.Bottom)) {
    // Bottom
    rect[1] = otherY + otherHeight + outsetY
  } else if (hasFlag(anchors, Alignment.Middle)) {
    // Vertical middle
    rect[1] = otherY + otherHeight * 0.5 - rect[3] * 0.5
  }
  return rect
}
