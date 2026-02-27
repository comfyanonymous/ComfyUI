import { ref } from 'vue'
import type { Ref, ShallowRef } from 'vue'

import type { LGraph } from '@/lib/litegraph/src/litegraph'

import { renderMinimapToCanvas } from '../minimapCanvasRenderer'
import type { UpdateFlags } from '../types'

export function useMinimapRenderer(
  canvasRef: Readonly<ShallowRef<HTMLCanvasElement | null>>,
  graph: Ref<LGraph | null>,
  bounds: Ref<{ minX: number; minY: number; width: number; height: number }>,
  scale: Ref<number>,
  updateFlags: Ref<UpdateFlags>,
  settings: {
    nodeColors: Ref<boolean>
    showLinks: Ref<boolean>
    showGroups: Ref<boolean>
    renderBypass: Ref<boolean>
    renderError: Ref<boolean>
  },
  width: number,
  height: number
) {
  const needsFullRedraw = ref(true)
  const needsBoundsUpdate = ref(true)

  const renderMinimap = () => {
    const g = graph.value
    if (!canvasRef.value || !g) return

    const ctx = canvasRef.value.getContext('2d')
    if (!ctx) return

    // Fast path for 0 nodes - just show background
    if (!g._nodes || g._nodes.length === 0) {
      ctx.clearRect(0, 0, width, height)
      return
    }

    const needsRedraw =
      needsFullRedraw.value ||
      updateFlags.value.nodes ||
      updateFlags.value.connections

    if (needsRedraw) {
      renderMinimapToCanvas(canvasRef.value, g, {
        bounds: bounds.value,
        scale: scale.value,
        settings: {
          nodeColors: settings.nodeColors.value,
          showLinks: settings.showLinks.value,
          showGroups: settings.showGroups.value,
          renderBypass: settings.renderBypass.value,
          renderError: settings.renderError.value
        },
        width,
        height
      })

      needsFullRedraw.value = false
      updateFlags.value.nodes = false
      updateFlags.value.connections = false
    }
  }

  const updateMinimap = (
    updateBounds: () => void,
    updateViewport: () => void
  ) => {
    if (needsBoundsUpdate.value || updateFlags.value.bounds) {
      updateBounds()
      needsBoundsUpdate.value = false
      updateFlags.value.bounds = false
      needsFullRedraw.value = true
      // When bounds change, we need to update the viewport position
      updateFlags.value.viewport = true
    }

    if (
      needsFullRedraw.value ||
      updateFlags.value.nodes ||
      updateFlags.value.connections
    ) {
      renderMinimap()
    }

    // Update viewport if needed (e.g., after bounds change)
    if (updateFlags.value.viewport) {
      updateViewport()
      updateFlags.value.viewport = false
    }
  }

  const forceFullRedraw = () => {
    needsFullRedraw.value = true
    updateFlags.value.bounds = true
    updateFlags.value.nodes = true
    updateFlags.value.connections = true
    updateFlags.value.viewport = true
  }

  return {
    needsFullRedraw,
    needsBoundsUpdate,
    renderMinimap,
    updateMinimap,
    forceFullRedraw
  }
}
