/**
 * Composable for managing transform state synchronized with LiteGraph canvas
 *
 * This composable is a critical part of the hybrid rendering architecture that
 * allows Vue components to render in perfect alignment with LiteGraph's canvas.
 *
 * ## Core Concept
 *
 * LiteGraph uses a canvas for rendering connections, grid, and handling interactions.
 * Vue components need to render nodes on top of this canvas. The challenge is
 * synchronizing the coordinate systems:
 *
 * - LiteGraph: Uses canvas coordinates with its own transform matrix
 * - Vue/DOM: Uses screen coordinates with CSS transforms
 *
 * ## Solution: Transform Container Pattern
 *
 * Instead of transforming individual nodes (O(n) complexity), we:
 * 1. Mirror LiteGraph's transform matrix to a single CSS container
 * 2. Place all Vue nodes as children with simple absolute positioning
 * 3. Achieve O(1) transform updates regardless of node count
 *
 * ## Coordinate Systems
 *
 * - **Canvas coordinates**: LiteGraph's internal coordinate system
 * - **Screen coordinates**: Browser's viewport coordinate system
 * - **Transform sync**: camera.x/y/z mirrors canvas.ds.offset/scale
 *
 * ## Performance Benefits
 *
 * - GPU acceleration via CSS transforms
 * - No layout thrashing (only transform changes)
 * - Efficient viewport culling calculations
 * - Scales to 1000+ nodes while maintaining 60 FPS
 *
 * @example
 * ```typescript
 * const { camera, transformStyle, canvasToScreen } = useTransformState()
 *
 * // In template
 * <div :style="transformStyle">
 *   <NodeComponent
 *     v-for="node in nodes"
 *     :style="{ left: node.x + 'px', top: node.y + 'px' }"
 *   />
 * </div>
 *
 * // Convert coordinates
 * const screenPos = canvasToScreen({ x: nodeX, y: nodeY })
 * ```
 */
import { computed, reactive, readonly } from 'vue'

import type { LGraphCanvas } from '@/lib/litegraph/src/litegraph'
import { createSharedComposable } from '@vueuse/core'

interface Point {
  x: number
  y: number
}

interface Camera {
  x: number
  y: number
  z: number // scale/zoom
}

function useTransformStateIndividual() {
  // Reactive state mirroring LiteGraph's canvas transform
  const camera = reactive<Camera>({
    x: 0,
    y: 0,
    z: 1
  })

  // Computed transform string for CSS
  const transformStyle = computed(() => ({
    // Match LiteGraph DragAndScale.toCanvasContext():
    // ctx.scale(scale); ctx.translate(offset)
    // CSS applies right-to-left, so "scale() translate()" -> translate first, then scale
    // Effective mapping: screen = (canvas + offset) * scale
    transform: `scale(${camera.z}) translate(${camera.x}px, ${camera.y}px)`,
    transformOrigin: '0 0'
  }))

  /**
   * Synchronizes Vue's reactive camera state with LiteGraph's canvas transform
   *
   * Called every frame via RAF to ensure Vue components stay aligned with canvas.
   * This is the heart of the hybrid rendering system - it bridges the gap between
   * LiteGraph's canvas transforms and Vue's reactive system.
   *
   * @param canvas - LiteGraph canvas instance with DragAndScale (ds) transform state
   */
  function syncWithCanvas(canvas: LGraphCanvas) {
    if (!canvas || !canvas.ds) return

    // Mirror LiteGraph's transform state to Vue's reactive state
    // ds.offset = pan offset, ds.scale = zoom level
    camera.x = canvas.ds.offset[0]
    camera.y = canvas.ds.offset[1]
    camera.z = canvas.ds.scale || 1
  }

  /**
   * Converts canvas coordinates to screen coordinates
   *
   * Applies the same transform that LiteGraph uses for rendering.
   * Essential for positioning Vue components to align with canvas elements.
   *
   * Formula: screen = (canvas + offset) * scale
   *
   * @param point - Point in canvas coordinate system
   * @returns Point in screen coordinate system
   */
  function canvasToScreen(point: Point): Point {
    return {
      x: (point.x + camera.x) * camera.z,
      y: (point.y + camera.y) * camera.z
    }
  }

  /**
   * Converts screen coordinates to canvas coordinates
   *
   * Inverse of canvasToScreen. Useful for hit testing and converting
   * mouse events back to canvas space.
   *
   * Formula: canvas = screen / scale - offset
   *
   * @param point - Point in screen coordinate system
   * @returns Point in canvas coordinate system
   */
  const screenToCanvas = (point: Point): Point => {
    return {
      x: point.x / camera.z - camera.x,
      y: point.y / camera.z - camera.y
    }
  }

  // Get node's screen bounds for culling
  function getNodeScreenBounds(
    pos: [number, number],
    size: [number, number]
  ): DOMRect {
    const topLeft = canvasToScreen({ x: pos[0], y: pos[1] })
    const width = size[0] * camera.z
    const height = size[1] * camera.z

    return new DOMRect(topLeft.x, topLeft.y, width, height)
  }

  // Helper: Calculate zoom-adjusted margin for viewport culling
  function calculateAdjustedMargin(baseMargin: number): number {
    if (camera.z < 0.1) return Math.min(baseMargin * 5, 2.0)
    if (camera.z > 3.0) return Math.max(baseMargin * 0.5, 0.05)
    return baseMargin
  }

  // Helper: Check if node is too small to be visible at current zoom
  function isNodeTooSmall(nodeSize: [number, number]): boolean {
    const nodeScreenSize = Math.max(nodeSize[0], nodeSize[1]) * camera.z
    return nodeScreenSize < 4
  }

  // Helper: Calculate expanded viewport bounds with margin
  function getExpandedViewportBounds(
    viewport: { width: number; height: number },
    margin: number
  ) {
    const marginX = viewport.width * margin
    const marginY = viewport.height * margin
    return {
      left: -marginX,
      right: viewport.width + marginX,
      top: -marginY,
      bottom: viewport.height + marginY
    }
  }

  // Helper: Test if node intersects with viewport bounds
  function testViewportIntersection(
    screenPos: { x: number; y: number },
    nodeSize: [number, number],
    bounds: { left: number; right: number; top: number; bottom: number }
  ): boolean {
    const nodeRight = screenPos.x + nodeSize[0] * camera.z
    const nodeBottom = screenPos.y + nodeSize[1] * camera.z

    return !(
      nodeRight < bounds.left ||
      screenPos.x > bounds.right ||
      nodeBottom < bounds.top ||
      screenPos.y > bounds.bottom
    )
  }

  // Check if node is within viewport with frustum and size-based culling
  function isNodeInViewport(
    nodePos: [number, number],
    nodeSize: [number, number],
    viewport: { width: number; height: number },
    margin: number = 0.2
  ): boolean {
    // Early exit for tiny nodes
    if (isNodeTooSmall(nodeSize)) return false

    const screenPos = canvasToScreen({ x: nodePos[0], y: nodePos[1] })
    const adjustedMargin = calculateAdjustedMargin(margin)
    const bounds = getExpandedViewportBounds(viewport, adjustedMargin)

    return testViewportIntersection(screenPos, nodeSize, bounds)
  }

  // Get viewport bounds in canvas coordinates (for spatial index queries)
  function getViewportBounds(
    viewport: { width: number; height: number },
    margin: number = 0.2
  ) {
    const marginX = viewport.width * margin
    const marginY = viewport.height * margin

    const topLeft = screenToCanvas({ x: -marginX, y: -marginY })
    const bottomRight = screenToCanvas({
      x: viewport.width + marginX,
      y: viewport.height + marginY
    })

    return {
      x: topLeft.x,
      y: topLeft.y,
      width: bottomRight.x - topLeft.x,
      height: bottomRight.y - topLeft.y
    }
  }

  return {
    camera: readonly(camera),
    transformStyle,
    syncWithCanvas,
    canvasToScreen,
    screenToCanvas,
    getNodeScreenBounds,
    isNodeInViewport,
    getViewportBounds
  }
}

export const useTransformState = createSharedComposable(
  useTransformStateIndividual
)
