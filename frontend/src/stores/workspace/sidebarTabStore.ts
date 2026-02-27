import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'

import { useAssetsSidebarTab } from '@/composables/sidebarTabs/useAssetsSidebarTab'
import { useJobHistorySidebarTab } from '@/composables/sidebarTabs/useJobHistorySidebarTab'
import { useModelLibrarySidebarTab } from '@/composables/sidebarTabs/useModelLibrarySidebarTab'
import { useNodeLibrarySidebarTab } from '@/composables/sidebarTabs/useNodeLibrarySidebarTab'
import { t, te } from '@/i18n'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useWorkflowsSidebarTab } from '@/platform/workflow/management/composables/useWorkflowsSidebarTab'
import { useCommandStore } from '@/stores/commandStore'
import { useMenuItemStore } from '@/stores/menuItemStore'
import type { SidebarTabExtension } from '@/types/extensionTypes'

export const useSidebarTabStore = defineStore('sidebarTab', () => {
  const sidebarTabs = ref<SidebarTabExtension[]>([])
  const activeSidebarTabId = ref<string | null>(null)

  const activeSidebarTab = computed<SidebarTabExtension | null>(() => {
    return (
      sidebarTabs.value.find((tab) => tab.id === activeSidebarTabId.value) ??
      null
    )
  })

  const toggleSidebarTab = (tabId: string) => {
    activeSidebarTabId.value = activeSidebarTabId.value === tabId ? null : tabId
  }

  const registerSidebarTab = (
    tab: SidebarTabExtension,
    options?: { prepend?: boolean }
  ) => {
    sidebarTabs.value = options?.prepend
      ? [tab, ...sidebarTabs.value]
      : [...sidebarTabs.value, tab]

    // Generate label in format "Toggle X Sidebar"
    const labelFunction = () => {
      const tabTitle = te(tab.title) ? t(tab.title) : tab.title
      return `Toggle ${tabTitle} Sidebar`
    }
    const tooltipFunction = tab.tooltip
      ? te(String(tab.tooltip))
        ? () => t(String(tab.tooltip))
        : String(tab.tooltip)
      : undefined

    const menubarLabelFunction = () => {
      const menubarLabelKeys: Record<string, string> = {
        'node-library': 'sideToolbar.nodeLibrary',
        'model-library': 'sideToolbar.modelLibrary',
        workflows: 'sideToolbar.workflows',
        assets: 'sideToolbar.assets',
        'job-history': 'queue.jobHistory'
      }

      const key = menubarLabelKeys[tab.id]
      if (key && te(key)) {
        return t(key)
      }

      return tab.title
    }

    useCommandStore().registerCommand({
      id: `Workspace.ToggleSidebarTab.${tab.id}`,
      icon: typeof tab.icon === 'string' ? tab.icon : undefined,
      label: labelFunction,
      menubarLabel: menubarLabelFunction,
      tooltip: tooltipFunction,
      versionAdded: '1.3.9',
      category: 'view-controls' as const,
      function: async () => {
        const settingStore = useSettingStore()
        const commandStore = useCommandStore()

        if (
          tab.id === 'model-library' &&
          settingStore.get('Comfy.Assets.UseAssetAPI')
        ) {
          await commandStore.commands
            .find((cmd) => cmd.id === 'Comfy.BrowseModelAssets')
            ?.function?.()
          return
        }

        toggleSidebarTab(tab.id)
      },
      active: () => activeSidebarTab.value?.id === tab.id,
      source: 'System'
    })
  }

  const unregisterSidebarTab = (id: string) => {
    const index = sidebarTabs.value.findIndex((tab) => tab.id === id)
    if (index !== -1) {
      const tab = sidebarTabs.value[index]
      if (tab.type === 'custom' && tab.destroy) {
        tab.destroy()
      }
      const newSidebarTabs = [...sidebarTabs.value]
      newSidebarTabs.splice(index, 1)
      sidebarTabs.value = newSidebarTabs
      if (activeSidebarTabId.value === id) {
        activeSidebarTabId.value = null
      }
    }
  }

  /**
   * Register the core sidebar tabs.
   */
  const registerCoreSidebarTabs = () => {
    const settingStore = useSettingStore()
    const jobHistoryTabId = 'job-history'
    const syncJobHistoryTab = (enabled: boolean) => {
      const hasJobHistoryTab = sidebarTabs.value.some(
        (tab) => tab.id === jobHistoryTabId
      )
      if (enabled && !hasJobHistoryTab) {
        registerSidebarTab(useJobHistorySidebarTab(), { prepend: true })
      } else if (!enabled && hasJobHistoryTab) {
        unregisterSidebarTab(jobHistoryTabId)
      }
    }

    syncJobHistoryTab(settingStore.get('Comfy.Queue.QPOV2'))
    watch(
      () => settingStore.get('Comfy.Queue.QPOV2'),
      (enabled) => syncJobHistoryTab(enabled)
    )

    registerSidebarTab(useAssetsSidebarTab())
    registerSidebarTab(useNodeLibrarySidebarTab())
    registerSidebarTab(useModelLibrarySidebarTab())
    registerSidebarTab(useWorkflowsSidebarTab())

    const menuStore = useMenuItemStore()

    menuStore.registerCommands(
      ['View'],
      [
        'Workspace.ToggleBottomPanel',
        'Comfy.BrowseTemplates',
        'Workspace.ToggleFocusMode',
        'Comfy.ToggleCanvasInfo',
        'Comfy.Canvas.ToggleMinimap',
        'Comfy.Canvas.ToggleLinkVisibility'
      ]
    )

    menuStore.registerCommands(
      ['View'],
      ['Comfy.Canvas.ZoomIn', 'Comfy.Canvas.ZoomOut', 'Comfy.Canvas.FitView']
    )
  }

  return {
    sidebarTabs,
    activeSidebarTabId,
    activeSidebarTab,
    toggleSidebarTab,
    registerSidebarTab,
    unregisterSidebarTab,
    registerCoreSidebarTabs
  }
})
