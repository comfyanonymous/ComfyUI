import { computed } from 'vue'

import { useSelectionState } from '@/composables/graph/useSelectionState'
import { LGraphGroup } from '@/lib/litegraph/src/litegraph'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useTitleEditorStore } from '@/renderer/core/canvas/canvasStore'
import { app } from '@/scripts/app'

/**
 * Composable encapsulating logic for framing currently selected nodes into a group.
 */
export function useFrameNodes() {
  const settingStore = useSettingStore()
  const titleEditorStore = useTitleEditorStore()
  const { hasMultipleSelection } = useSelectionState()

  const canFrame = computed(() => hasMultipleSelection.value)

  const frameNodes = () => {
    const { canvas } = app
    if (!canvas.selectedItems?.size) return
    const group = new LGraphGroup()
    const padding = settingStore.get('Comfy.GroupSelectedNodes.Padding')
    group.resizeTo(canvas.selectedItems, padding)
    canvas.graph?.add(group)
    titleEditorStore.titleEditorTarget = group
  }

  return { frameNodes, canFrame }
}
