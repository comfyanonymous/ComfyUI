import { markRaw } from 'vue'

import AssetsSidebarTab from '@/components/sidebar/tabs/AssetsSidebarTab.vue'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useAssetsSidebarBadgeStore } from '@/stores/workspace/assetsSidebarBadgeStore'
import type { SidebarTabExtension } from '@/types/extensionTypes'

export const useAssetsSidebarTab = (): SidebarTabExtension => {
  return {
    id: 'assets',
    icon: 'icon-[comfy--image-ai-edit]',
    title: 'sideToolbar.assets',
    tooltip: 'sideToolbar.assets',
    label: 'sideToolbar.labels.assets',
    component: markRaw(AssetsSidebarTab),
    type: 'vue',
    iconBadge: () => {
      const settingStore = useSettingStore()

      if (!settingStore.get('Comfy.Queue.QPOV2')) {
        return null
      }

      const assetsSidebarBadgeStore = useAssetsSidebarBadgeStore()
      const count = assetsSidebarBadgeStore.unseenAddedAssetsCount
      return count > 0 ? count.toString() : null
    }
  }
}
