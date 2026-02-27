import { computed, ref, watchEffect } from 'vue'

import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { isLGraphNode } from '@/utils/litegraphUtil'

interface RefreshableItem {
  refresh: () => Promise<void> | void
}

const isRefreshableWidget = (widget: unknown): widget is RefreshableItem =>
  widget != null &&
  typeof widget === 'object' &&
  'refresh' in widget &&
  typeof widget.refresh === 'function'

/**
 * Tracks selected nodes and their refreshable widgets
 */
export const useRefreshableSelection = () => {
  const graphStore = useCanvasStore()
  const selectedNodes = ref<LGraphNode[]>([])

  watchEffect(() => {
    selectedNodes.value = graphStore.selectedItems.filter(isLGraphNode)
  })

  const refreshableWidgets = computed<RefreshableItem[]>(() =>
    selectedNodes.value.flatMap((node) => {
      if (!node.widgets) return []
      const items: RefreshableItem[] = []
      for (const widget of node.widgets) {
        if (isRefreshableWidget(widget)) {
          items.push(widget)
        }
      }
      return items
    })
  )

  const isRefreshable = computed(() => refreshableWidgets.value.length > 0)

  async function refreshSelected() {
    if (!isRefreshable.value) return

    await Promise.all(refreshableWidgets.value.map((item) => item.refresh()))
  }

  return {
    isRefreshable,
    refreshSelected
  }
}
