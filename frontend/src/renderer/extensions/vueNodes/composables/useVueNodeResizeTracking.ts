/**
 * Generic Vue Element Tracking System
 *
 * Automatically tracks DOM size and position changes for Vue-rendered elements
 * and syncs them to the layout store. Uses a single shared ResizeObserver for
 * performance, with elements identified by configurable data attributes.
 *
 * Supports different element types (nodes, slots, widgets, etc.) with
 * customizable data attributes and update handlers.
 */
import { getCurrentInstance, onMounted, onUnmounted, toValue, watch } from 'vue'
import type { MaybeRefOrGetter } from 'vue'

import { useDocumentVisibility } from '@vueuse/core'

import { useSharedCanvasPositionConversion } from '@/composables/element/useCanvasPositionConversion'
import { LiteGraph } from '@/lib/litegraph/src/litegraph'
import { layoutStore } from '@/renderer/core/layout/store/layoutStore'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import type { Bounds, NodeId } from '@/renderer/core/layout/types'
import { LayoutSource } from '@/renderer/core/layout/types'

import { syncNodeSlotLayoutsFromDOM } from './useSlotElementTracking'

/**
 * Generic update item for element bounds tracking
 */
interface ElementBoundsUpdate {
  /** Element identifier (could be nodeId, widgetId, slotId, etc.) */
  id: string
  /** Updated bounds */
  bounds: Bounds
}

/**
 * Configuration for different types of tracked elements
 */
interface ElementTrackingConfig {
  /** Data attribute name (e.g., 'nodeId') */
  dataAttribute: string
  /** Handler for processing bounds updates */
  updateHandler: (updates: ElementBoundsUpdate[]) => void
}

/**
 * Registry of tracking configurations by element type
 */
const trackingConfigs: Map<string, ElementTrackingConfig> = new Map([
  [
    'node',
    {
      dataAttribute: 'nodeId',
      updateHandler: (updates) => {
        const nodeUpdates = updates.map(({ id, bounds }) => ({
          nodeId: id as NodeId,
          bounds
        }))
        layoutStore.batchUpdateNodeBounds(nodeUpdates)
      }
    }
  ]
])

// Elements whose ResizeObserver fired while the tab was hidden
const deferredElements = new Set<HTMLElement>()
const visibility = useDocumentVisibility()

watch(visibility, (state) => {
  if (state !== 'visible' || deferredElements.size === 0) return

  // Re-observe deferred elements to trigger fresh measurements
  for (const element of deferredElements) {
    if (element.isConnected) {
      resizeObserver.observe(element)
    }
  }
  deferredElements.clear()
})

// Single ResizeObserver instance for all Vue elements
const resizeObserver = new ResizeObserver((entries) => {
  if (useCanvasStore().linearMode) return

  // Skip measurements when tab is hidden — bounding rects are unreliable
  if (visibility.value === 'hidden') {
    for (const entry of entries) {
      if (entry.target instanceof HTMLElement) {
        deferredElements.add(entry.target)
        resizeObserver.unobserve(entry.target)
      }
    }
    return
  }

  // Canvas is ready when this code runs; no defensive guards needed.
  const conv = useSharedCanvasPositionConversion()
  // Group updates by type, then flush via each config's handler
  const updatesByType = new Map<string, ElementBoundsUpdate[]>()
  // Track nodes whose slots should be resynced after node size changes
  const nodesNeedingSlotResync = new Set<string>()

  for (const entry of entries) {
    if (!(entry.target instanceof HTMLElement)) continue
    const element = entry.target

    // Find which type this element belongs to
    let elementType: string | undefined
    let elementId: string | undefined

    for (const [type, config] of trackingConfigs) {
      const id = element.dataset[config.dataAttribute]
      if (id) {
        elementType = type
        elementId = id
        break
      }
    }

    if (!elementType || !elementId) continue

    // Use borderBoxSize when available; fall back to contentRect for older engines/tests
    // Border box is the border included FULL wxh DOM value.
    const borderBox = Array.isArray(entry.borderBoxSize)
      ? entry.borderBoxSize[0]
      : {
          inlineSize: entry.contentRect.width,
          blockSize: entry.contentRect.height
        }
    const width = borderBox.inlineSize
    const height = borderBox.blockSize

    // Screen-space rect
    const rect = element.getBoundingClientRect()
    const [cx, cy] = conv.clientPosToCanvasPos([rect.left, rect.top])
    const topLeftCanvas = { x: cx, y: cy }
    const bounds: Bounds = {
      x: topLeftCanvas.x,
      y: topLeftCanvas.y + LiteGraph.NODE_TITLE_HEIGHT,
      width: Math.max(0, width),
      height: Math.max(0, height)
    }

    let updates = updatesByType.get(elementType)
    if (!updates) {
      updates = []
      updatesByType.set(elementType, updates)
    }
    updates.push({ id: elementId, bounds })

    // If this entry is a node, mark it for slot layout resync
    if (elementType === 'node' && elementId) {
      nodesNeedingSlotResync.add(elementId)
    }
  }

  layoutStore.setSource(LayoutSource.DOM)

  // Flush per-type
  for (const [type, updates] of updatesByType) {
    const config = trackingConfigs.get(type)
    if (config && updates.length) config.updateHandler(updates)
  }

  // After node bounds are updated, refresh slot cached offsets and layouts
  if (nodesNeedingSlotResync.size > 0) {
    for (const nodeId of nodesNeedingSlotResync) {
      syncNodeSlotLayoutsFromDOM(nodeId)
    }
  }
})

/**
 * Tracks DOM element size/position changes for a Vue component and syncs to layout store
 *
 * Sets up automatic ResizeObserver tracking when the component mounts and cleans up
 * when unmounted. The tracked element is identified by a data attribute set on the
 * component's root DOM element.
 *
 * @param appIdentifier - Application-level identifier for this tracked element (not a DOM ID)
 *                       Example: node ID like 'node-123', widget ID like 'widget-456'
 * @param trackingType - Type of element being tracked, determines which tracking config to use
 *                      Example: 'node' for Vue nodes, 'widget' for UI widgets
 *
 * @example
 * ```ts
 * // Track a Vue node component with ID 'my-node-123'
 * useVueElementTracking('my-node-123', 'node')
 *
 * // Would set data-node-id="my-node-123" on the component's root element
 * // and sync size changes to layoutStore.batchUpdateNodeBounds()
 * ```
 */
export function useVueElementTracking(
  appIdentifierMaybe: MaybeRefOrGetter<string>,
  trackingType: string
) {
  const appIdentifier = toValue(appIdentifierMaybe)
  onMounted(() => {
    const element = getCurrentInstance()?.proxy?.$el
    if (!(element instanceof HTMLElement) || !appIdentifier) return

    const config = trackingConfigs.get(trackingType)
    if (!config) return

    // Set the data attribute expected by the RO pipeline for this type
    element.dataset[config.dataAttribute] = appIdentifier
    resizeObserver.observe(element)
  })

  onUnmounted(() => {
    const element = getCurrentInstance()?.proxy?.$el
    if (!(element instanceof HTMLElement)) return

    const config = trackingConfigs.get(trackingType)
    if (!config) return

    // Remove the data attribute and observer
    delete element.dataset[config.dataAttribute]
    resizeObserver.unobserve(element)
  })
}
