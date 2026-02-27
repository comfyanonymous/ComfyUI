import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'

import { st } from '@/i18n'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'
import type { LGraphNode, NodeId } from '@/lib/litegraph/src/litegraph'
import { app } from '@/scripts/app'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import type { NodeLocatorId } from '@/types/nodeIdentification'
import { getNodeByLocatorId } from '@/utils/graphTraversalUtil'
import { resolveNodeDisplayName } from '@/utils/nodeTitleUtil'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'

/**
 * Unique identifier for a favorited widget.
 * Combines node locator ID and widget name to locate a widget in the graph.
 */
interface FavoritedWidgetId {
  /** The node locator ID in the graph */
  nodeLocatorId: NodeLocatorId
  /** The widget name on the node */
  widgetName: string
}

/**
 * A favorited widget with its resolved runtime instance.
 * The widget instance may be null if the node or widget no longer exists.
 */
interface FavoritedWidget extends FavoritedWidgetId {
  /** The resolved node instance (null if node was deleted) */
  node: LGraphNode | null
  /** The resolved widget instance (null if widget no longer exists) */
  widget: IBaseWidget | null
  /** Display label for the favorited item */
  label: string
}

export interface ValidFavoritedWidget extends FavoritedWidget {
  node: LGraphNode
  widget: IBaseWidget
}

/**
 * Storage format for persisted favorited widgets.
 * Stored in workflow.extra.favoritedWidgets.
 */
interface FavoritedWidgetStorage {
  /** Array of favorited widget identifiers */
  favorites: FavoritedWidgetId[]
}

/**
 * Store for managing favorited/starred widgets.
 *
 * Favorited widgets can be accessed and edited from the right side panel
 * without needing to select the corresponding node. This store manages:
 * - Persisting favorited widget IDs per workflow
 * - Resolving widget IDs to actual widget instances
 * - Handling cases where nodes/widgets are deleted
 *
 * Design decisions:
 * - Scope: Per-workflow (not global user preference)
 * - Identifier: node locator ID + widget.name
 * - Persistence: Stored in workflow.extra.favoritedWidgets (serialized with workflow)
 * - Future: Can be extended for Linear Mode
 */
export const useFavoritedWidgetsStore = defineStore('favoritedWidgets', () => {
  const workflowStore = useWorkflowStore()
  const canvasStore = useCanvasStore()

  /** In-memory array of favorited widget IDs, ordered for display */
  const favoritedIds = ref<string[]>([])

  /**
   * Generate a unique string key for a favorited widget ID.
   */
  function getFavoriteKey(id: FavoritedWidgetId): string {
    return JSON.stringify([id.nodeLocatorId, id.widgetName])
  }

  /**
   * Parse a favorite key back into a FavoritedWidgetId.
   */
  function parseFavoriteKey(key: string): FavoritedWidgetId | null {
    try {
      const [nodeLocatorId, widgetName] = JSON.parse(key) as [string, string]
      if (!nodeLocatorId || !widgetName) return null
      return { nodeLocatorId, widgetName }
    } catch {
      const separatorIndex = key.indexOf(':')
      if (separatorIndex === -1) return null
      const nodeLocatorId = key.slice(0, separatorIndex)
      const widgetName = key.slice(separatorIndex + 1)
      if (!nodeLocatorId || !widgetName) return null
      return { nodeLocatorId, widgetName }
    }
  }

  function normalizeFavoritedId(
    id: FavoritedWidgetId | { nodeId?: unknown; widgetName?: unknown } | null
  ): FavoritedWidgetId | null {
    if (!id || !id.widgetName) return null

    if ('nodeLocatorId' in id && id.nodeLocatorId) {
      return {
        nodeLocatorId: String(id.nodeLocatorId),
        widgetName: String(id.widgetName)
      }
    }

    if ('nodeId' in id && id.nodeId !== undefined) {
      return {
        nodeLocatorId: workflowStore.nodeIdToNodeLocatorId(id.nodeId as NodeId),
        widgetName: String(id.widgetName)
      }
    }

    return null
  }

  function createFavoriteId(
    node: LGraphNode,
    widgetName: string
  ): FavoritedWidgetId {
    return {
      nodeLocatorId: workflowStore.nodeToNodeLocatorId(node),
      widgetName
    }
  }

  /**
   * Load favorited widgets from the current workflow's extra data.
   */
  function loadFromWorkflow() {
    const graph = app.rootGraph
    if (!graph) {
      favoritedIds.value = []
      return
    }

    try {
      const storedData = graph.extra?.favoritedWidgets as
        | FavoritedWidgetStorage
        | undefined

      if (storedData?.favorites) {
        const normalized = storedData.favorites
          .map((fav) => normalizeFavoritedId(fav))
          .filter((fav): fav is FavoritedWidgetId => fav !== null)
        favoritedIds.value = normalized.map(getFavoriteKey)
      } else {
        favoritedIds.value = []
      }
    } catch (error) {
      console.error('Failed to load favorited widgets from workflow:', error)
      favoritedIds.value = []
    }
  }

  /**
   * Save favorited widgets to the current workflow's extra data.
   * Marks the workflow as modified.
   */
  function saveToWorkflow() {
    const graph = app.rootGraph
    if (!graph) return

    try {
      const favorites: FavoritedWidgetId[] = favoritedIds.value
        .map(parseFavoriteKey)
        .filter((id): id is FavoritedWidgetId => id !== null)

      const data: FavoritedWidgetStorage = { favorites }

      // Ensure extra object exists
      graph.extra ??= {}
      graph.extra.favoritedWidgets = data

      // Mark the workflow as modified
      canvasStore.canvas?.setDirty(true, true)
    } catch (error) {
      console.error('Failed to save favorited widgets to workflow:', error)
    }
  }

  /**
   * Resolve a favorited widget ID to its actual widget instance.
   * Returns null if the node or widget no longer exists.
   */
  function resolveWidget(id: FavoritedWidgetId): FavoritedWidget {
    const graph = app.rootGraph
    if (!graph) {
      return {
        ...id,
        node: null,
        widget: null,
        label: `${id.widgetName} (graph not loaded)`
      }
    }

    const node = getNodeByLocatorId(graph, id.nodeLocatorId)
    if (!node) {
      return {
        ...id,
        node: null,
        widget: null,
        label: `${id.widgetName} (node deleted)`
      }
    }

    const widget = node.widgets?.find((w) => w.name === id.widgetName)
    if (!widget) {
      return {
        ...id,
        node,
        widget: null,
        label: `${id.widgetName} (widget not found)`
      }
    }

    const fallbackNodeTitle = st('rightSidePanel.fallbackNodeTitle', 'Node')
    const nodeTitle = resolveNodeDisplayName(node, {
      emptyLabel: fallbackNodeTitle,
      untitledLabel: fallbackNodeTitle,
      st
    })
    const widgetLabel = widget.label || widget.name
    return {
      ...id,
      node,
      widget,
      label: `${nodeTitle} / ${widgetLabel}`
    }
  }

  /**
   * Get all favorited widgets with their resolved instances.
   * Widgets that no longer exist will have null node/widget properties.
   */
  const favoritedWidgets = computed((): FavoritedWidget[] => {
    return favoritedIds.value
      .map(parseFavoriteKey)
      .filter((id): id is FavoritedWidgetId => id !== null)
      .map(resolveWidget)
  })

  /**
   * Get only the valid favorited widgets (where both node and widget exist).
   */
  const validFavoritedWidgets = computed((): ValidFavoritedWidget[] => {
    return favoritedWidgets.value.filter(
      (fw) => fw.node !== null && fw.widget !== null
    ) as ValidFavoritedWidget[]
  })

  /**
   * Check if a widget is favorited.
   */
  function isFavorited(node: LGraphNode, widgetName: string): boolean {
    return favoritedIds.value.includes(
      getFavoriteKey(createFavoriteId(node, widgetName))
    )
  }

  /**
   * Add a widget to favorites.
   */
  function addFavorite(node: LGraphNode, widgetName: string) {
    const key = getFavoriteKey(createFavoriteId(node, widgetName))
    if (favoritedIds.value.includes(key)) return

    favoritedIds.value.push(key)
    saveToWorkflow()
  }

  /**
   * Remove a widget from favorites.
   */
  function removeFavorite(node: LGraphNode, widgetName: string) {
    const key = getFavoriteKey(createFavoriteId(node, widgetName))
    const index = favoritedIds.value.indexOf(key)
    if (index === -1) return

    favoritedIds.value.splice(index, 1)
    saveToWorkflow()
  }

  /**
   * Toggle a widget's favorite status.
   */
  function toggleFavorite(node: LGraphNode, widgetName: string) {
    if (isFavorited(node, widgetName)) {
      removeFavorite(node, widgetName)
    } else {
      addFavorite(node, widgetName)
    }
  }

  /**
   * Clear all favorites for the current workflow.
   */
  function clearFavorites() {
    favoritedIds.value = []
    saveToWorkflow()
  }

  /**
   * Remove invalid favorites (where node or widget no longer exists).
   * Useful for cleanup after loading a workflow.
   */
  function pruneInvalidFavorites() {
    const validKeys = validFavoritedWidgets.value.map((fw) =>
      getFavoriteKey({
        nodeLocatorId: fw.nodeLocatorId,
        widgetName: fw.widgetName
      })
    )
    const validSet = new Set(validKeys)

    const filteredIds = favoritedIds.value.filter((key) => validSet.has(key))

    if (filteredIds.length !== favoritedIds.value.length) {
      favoritedIds.value = filteredIds
      saveToWorkflow()
    }
  }

  /**
   * Reorder favorites based on the provided array of widgets.
   * Used when dragging and dropping favorites to reorder them.
   */
  function reorderFavorites(reorderedWidgets: ValidFavoritedWidget[]) {
    favoritedIds.value = reorderedWidgets.map((fw) =>
      getFavoriteKey({
        nodeLocatorId: fw.nodeLocatorId,
        widgetName: fw.widgetName
      })
    )
    saveToWorkflow()
  }

  // Watch for workflow changes and reload favorites from workflow.extra
  watch(
    () => workflowStore.activeWorkflow?.path,
    () => {
      loadFromWorkflow()
    },
    { immediate: true }
  )

  return {
    // State
    favoritedWidgets,
    validFavoritedWidgets,

    // Actions
    isFavorited,
    addFavorite,
    removeFavorite,
    toggleFavorite,
    clearFavorites,
    pruneInvalidFavorites,
    reorderFavorites
  }
})
