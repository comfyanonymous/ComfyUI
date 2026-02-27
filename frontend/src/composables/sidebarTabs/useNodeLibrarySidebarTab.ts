import { computed, markRaw, reactive } from 'vue'

import NodeLibrarySidebarTab from '@/components/sidebar/tabs/NodeLibrarySidebarTab.vue'
import NodeLibrarySidebarTabV2 from '@/components/sidebar/tabs/NodeLibrarySidebarTabV2.vue'
import { useSettingStore } from '@/platform/settings/settingStore'
import type { SidebarTabExtension } from '@/types/extensionTypes'

export function useNodeLibrarySidebarTab(): SidebarTabExtension {
  const settingStore = useSettingStore()
  const component = computed(() =>
    settingStore.get('Comfy.NodeLibrary.NewDesign')
      ? markRaw(NodeLibrarySidebarTabV2)
      : markRaw(NodeLibrarySidebarTab)
  )

  return reactive({
    id: 'node-library',
    icon: 'icon-[comfy--node]',
    title: 'sideToolbar.nodeLibrary',
    tooltip: 'sideToolbar.nodeLibrary',
    label: 'sideToolbar.labels.nodes',
    component,
    type: 'vue' as const
  })
}
