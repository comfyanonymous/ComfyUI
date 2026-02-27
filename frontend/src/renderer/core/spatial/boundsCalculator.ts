/**
 * Spatial bounds calculations for node layouts
 */

interface SpatialBounds {
  minX: number
  minY: number
  maxX: number
  maxY: number
  width: number
  height: number
}

export interface PositionedNode {
  pos: [number, number]
  size: [number, number]
}

/**
 * Calculate the spatial bounding box of positioned nodes
 */
export function calculateNodeBounds(
  nodes: PositionedNode[]
): SpatialBounds | null {
  if (!nodes || nodes.length === 0) {
    return null
  }

  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity

  for (const node of nodes) {
    const x = node.pos[0]
    const y = node.pos[1]
    const width = node.size[0]
    const height = node.size[1]

    minX = Math.min(minX, x)
    minY = Math.min(minY, y)
    maxX = Math.max(maxX, x + width)
    maxY = Math.max(maxY, y + height)
  }

  return {
    minX,
    minY,
    maxX,
    maxY,
    width: maxX - minX,
    height: maxY - minY
  }
}

/**
 * Enforce minimum viewport dimensions for better visualization
 */
export function enforceMinimumBounds(
  bounds: SpatialBounds,
  minWidth: number = 2500,
  minHeight: number = 2000
): SpatialBounds {
  let { minX, minY, maxX, maxY, width, height } = bounds

  if (width < minWidth) {
    const padding = (minWidth - width) / 2
    minX -= padding
    maxX += padding
    width = minWidth
  }

  if (height < minHeight) {
    const padding = (minHeight - height) / 2
    minY -= padding
    maxY += padding
    height = minHeight
  }

  return { minX, minY, maxX, maxY, width, height }
}

/**
 * Calculate the scale factor to fit bounds within a viewport
 */
export function calculateMinimapScale(
  bounds: SpatialBounds,
  viewportWidth: number,
  viewportHeight: number,
  padding: number = 0.9
): number {
  if (bounds.width === 0 || bounds.height === 0) {
    return 1
  }

  const scaleX = viewportWidth / bounds.width
  const scaleY = viewportHeight / bounds.height

  return Math.min(scaleX, scaleY) * padding
}
