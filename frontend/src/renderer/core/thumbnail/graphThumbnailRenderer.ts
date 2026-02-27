import type { LGraph } from '@/lib/litegraph/src/litegraph'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import {
  calculateMinimapScale,
  calculateNodeBounds
} from '@/renderer/core/spatial/boundsCalculator'

import { renderMinimapToCanvas } from '../../extensions/minimap/minimapCanvasRenderer'

/**
 * Create a thumbnail of the current canvas's active graph.
 * Used by workflow thumbnail generation.
 */
export function createGraphThumbnail(): string | null {
  const canvasStore = useCanvasStore()
  const workflowStore = useWorkflowStore()

  const graph = workflowStore.activeSubgraph || canvasStore.canvas?.graph
  if (!graph || !graph._nodes || graph._nodes.length === 0) {
    return null
  }

  const width = 250
  const height = 200

  // Calculate bounds using spatial calculator
  const bounds = calculateNodeBounds(graph._nodes)
  if (!bounds) {
    return null
  }

  const scale = calculateMinimapScale(bounds, width, height)

  // Create detached canvas
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height

  // Render the minimap
  renderMinimapToCanvas(canvas, graph as LGraph, {
    bounds,
    scale,
    settings: {
      nodeColors: true,
      showLinks: false,
      showGroups: true,
      renderBypass: false,
      renderError: false
    },
    width,
    height
  })

  const dataUrl = canvas.toDataURL()

  // Explicit cleanup (optional but good practice)
  const ctx = canvas.getContext('2d')
  if (ctx) {
    ctx.clearRect(0, 0, width, height)
  }

  return dataUrl
}
