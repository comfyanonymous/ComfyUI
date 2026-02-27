import { useThrottleFn } from '@vueuse/core'
import { ref, watch } from 'vue'
import type { Ref } from 'vue'

import type {
  LGraph,
  LGraphNode,
  LGraphTriggerEvent
} from '@/lib/litegraph/src/litegraph'
import type { NodeId } from '@/platform/workflow/validation/schemas/workflowSchema'
import { layoutStore } from '@/renderer/core/layout/store/layoutStore'
import { api } from '@/scripts/api'

import { MinimapDataSourceFactory } from '../data/MinimapDataSourceFactory'
import type { UpdateFlags } from '../types'

interface GraphCallbacks {
  onNodeAdded?: (node: LGraphNode) => void
  onNodeRemoved?: (node: LGraphNode) => void
  onConnectionChange?: (node: LGraphNode) => void
  onTrigger?: (event: LGraphTriggerEvent) => void
}

export function useMinimapGraph(
  graph: Ref<LGraph | null>,
  onGraphChanged: () => void
) {
  const nodeStatesCache = new Map<NodeId, string>()
  const linksCache = ref<string>('')
  const lastNodeCount = ref(0)
  const updateFlags = ref<UpdateFlags>({
    bounds: false,
    nodes: false,
    connections: false,
    viewport: false
  })

  // Track LayoutStore version for change detection
  const layoutStoreVersion = layoutStore.getVersion()

  // Map to store original callbacks per graph ID
  const originalCallbacksMap = new Map<string, GraphCallbacks>()

  const handleGraphChangedThrottled = useThrottleFn(() => {
    onGraphChanged()
  }, 500)

  const setupEventListeners = () => {
    const g = graph.value
    if (!g) return

    // Check if we've already wrapped this graph's callbacks
    if (originalCallbacksMap.has(g.id)) {
      return
    }

    // Store the original callbacks for this graph
    const originalCallbacks: GraphCallbacks = {
      onNodeAdded: g.onNodeAdded,
      onNodeRemoved: g.onNodeRemoved,
      onConnectionChange: g.onConnectionChange,
      onTrigger: g.onTrigger
    }
    originalCallbacksMap.set(g.id, originalCallbacks)

    g.onNodeAdded = function (node: LGraphNode) {
      originalCallbacks.onNodeAdded?.call(this, node)
      void handleGraphChangedThrottled()
    }

    g.onNodeRemoved = function (node: LGraphNode) {
      originalCallbacks.onNodeRemoved?.call(this, node)
      nodeStatesCache.delete(node.id)
      void handleGraphChangedThrottled()
    }

    g.onConnectionChange = function (node: LGraphNode) {
      originalCallbacks.onConnectionChange?.call(this, node)
      void handleGraphChangedThrottled()
    }

    g.onTrigger = function (event: LGraphTriggerEvent) {
      originalCallbacks.onTrigger?.call(this, event)

      // Listen for visual property changes that affect minimap rendering
      if (
        event.type === 'node:property:changed' &&
        (event.property === 'mode' ||
          event.property === 'bgcolor' ||
          event.property === 'color')
      ) {
        // Invalidate cache for this node to force redraw
        nodeStatesCache.delete(String(event.nodeId))
        void handleGraphChangedThrottled()
      }
    }
  }

  const cleanupEventListeners = (oldGraph?: LGraph) => {
    const g = oldGraph || graph.value
    if (!g) return

    const originalCallbacks = originalCallbacksMap.get(g.id)
    if (!originalCallbacks) {
      // Graph was never set up (e.g., minimap destroyed before init) - nothing to clean up
      return
    }

    g.onNodeAdded = originalCallbacks.onNodeAdded
    g.onNodeRemoved = originalCallbacks.onNodeRemoved
    g.onConnectionChange = originalCallbacks.onConnectionChange
    g.onTrigger = originalCallbacks.onTrigger

    originalCallbacksMap.delete(g.id)
  }

  const checkForChangesInternal = () => {
    const g = graph.value
    if (!g) return false

    let structureChanged = false
    let positionChanged = false
    let connectionChanged = false

    // Use unified data source for change detection
    const dataSource = MinimapDataSourceFactory.create(g)

    // Check for node count changes
    const currentNodeCount = dataSource.getNodeCount()
    if (currentNodeCount !== lastNodeCount.value) {
      structureChanged = true
      lastNodeCount.value = currentNodeCount
    }

    // Check for node position/size changes
    const nodes = dataSource.getNodes()
    for (const node of nodes) {
      const nodeId = node.id
      const currentState = `${node.x},${node.y},${node.width},${node.height}`

      if (nodeStatesCache.get(nodeId) !== currentState) {
        positionChanged = true
        nodeStatesCache.set(nodeId, currentState)
      }
    }

    // Clean up removed nodes from cache
    const currentNodeIds = new Set(nodes.map((n) => n.id))
    for (const [nodeId] of nodeStatesCache) {
      if (!currentNodeIds.has(nodeId)) {
        nodeStatesCache.delete(nodeId)
        structureChanged = true
      }
    }

    // TODO: update when Layoutstore tracks links
    const currentLinks = JSON.stringify(g.links || {})
    if (currentLinks !== linksCache.value) {
      connectionChanged = true
      linksCache.value = currentLinks
    }

    if (structureChanged || positionChanged) {
      updateFlags.value.bounds = true
      updateFlags.value.nodes = true
    }

    if (connectionChanged) {
      updateFlags.value.connections = true
    }

    return structureChanged || positionChanged || connectionChanged
  }

  const init = () => {
    setupEventListeners()
    api.addEventListener('graphChanged', handleGraphChangedThrottled)

    watch(layoutStoreVersion, () => {
      void handleGraphChangedThrottled()
    })
  }

  const destroy = () => {
    cleanupEventListeners()
    api.removeEventListener('graphChanged', handleGraphChangedThrottled)
    nodeStatesCache.clear()
  }

  const clearCache = () => {
    nodeStatesCache.clear()
    linksCache.value = ''
    lastNodeCount.value = 0
  }

  return {
    updateFlags,
    setupEventListeners,
    cleanupEventListeners,
    checkForChanges: checkForChangesInternal,
    init,
    destroy,
    clearCache
  }
}
