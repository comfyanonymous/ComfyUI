import type { Bounds, NodeLayout, Point } from '@/renderer/core/layout/types'

export const REROUTE_RADIUS = 8

export function pointInBounds(point: Point, bounds: Bounds): boolean {
  return (
    point.x >= bounds.x &&
    point.x <= bounds.x + bounds.width &&
    point.y >= bounds.y &&
    point.y <= bounds.y + bounds.height
  )
}

export function boundsIntersect(a: Bounds, b: Bounds): boolean {
  return !(
    a.x + a.width < b.x ||
    b.x + b.width < a.x ||
    a.y + a.height < b.y ||
    b.y + b.height < a.y
  )
}

export function calculateBounds(nodes: NodeLayout[]): Bounds {
  let minX = Infinity,
    minY = Infinity
  let maxX = -Infinity,
    maxY = -Infinity

  for (const node of nodes) {
    const bounds = node.bounds
    minX = Math.min(minX, bounds.x)
    minY = Math.min(minY, bounds.y)
    maxX = Math.max(maxX, bounds.x + bounds.width)
    maxY = Math.max(maxY, bounds.y + bounds.height)
  }

  return {
    x: minX,
    y: minY,
    width: maxX - minX,
    height: maxY - minY
  }
}
