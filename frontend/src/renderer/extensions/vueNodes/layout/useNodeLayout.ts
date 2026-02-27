import { computed, onUnmounted, toValue } from 'vue'
import type { MaybeRefOrGetter } from 'vue'

import { useLayoutMutations } from '@/renderer/core/layout/operations/layoutMutations'
import { layoutStore } from '@/renderer/core/layout/store/layoutStore'
import { LayoutSource } from '@/renderer/core/layout/types'
import type { Point } from '@/renderer/core/layout/types'

/**
 * Composable for individual Vue node components
 * Uses customRef for shared write access with Canvas renderer
 */
export function useNodeLayout(nodeIdMaybe: MaybeRefOrGetter<string>) {
  const nodeId = toValue(nodeIdMaybe)
  const mutations = useLayoutMutations()

  // Get the customRef for this node (shared write access)
  const layoutRef = layoutStore.getNodeLayoutRef(nodeId)

  // Clean up refs and triggers when Vue component unmounts
  onUnmounted(() => {
    layoutStore.cleanupNodeRef(nodeId)
  })

  // Computed properties for easy access
  const position = computed(() => {
    const layout = layoutRef.value
    const pos = layout?.position ?? { x: 0, y: 0 }
    return pos
  })
  const size = computed(
    () => layoutRef.value?.size ?? { width: 200, height: 100 }
  )

  const zIndex = computed(() => layoutRef.value?.zIndex ?? 0)

  /**
   * Update node position directly (without drag)
   */
  function moveNodeTo(position: Point) {
    mutations.setSource(LayoutSource.Vue)
    mutations.moveNode(nodeId, position)
  }

  return {
    // Reactive state (via customRef)
    position,
    size,
    zIndex,

    // Mutations
    moveNodeTo
  }
}
