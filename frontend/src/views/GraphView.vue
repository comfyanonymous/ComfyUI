<template>
  <div class="comfyui-body grid h-full w-full overflow-hidden">
    <div id="comfyui-body-top" class="comfyui-body-top" />
    <div id="comfyui-body-bottom" class="comfyui-body-bottom" />
    <div id="comfyui-body-left" class="comfyui-body-left" />
    <div id="comfyui-body-right" class="comfyui-body-right" />
    <div
      v-show="!linearMode"
      id="graph-canvas-container"
      ref="graphCanvasContainerRef"
      class="graph-canvas-container"
    >
      <GraphCanvas @ready="onGraphReady" />
    </div>
    <LinearView v-if="linearMode" />
    <BuilderToolbar v-if="isBuilderMode" />
  </div>

  <GlobalToast />
  <InviteAcceptedToast />
  <RerouteMigrationToast />
  <ModelImportProgressDialog />
  <AssetExportProgressDialog />
  <ManagerProgressToast />
  <UnloadWindowConfirmDialog v-if="!isDesktop" />
  <MenuHamburger />
</template>

<script setup lang="ts">
import { useEventListener, useIntervalFn } from '@vueuse/core'
import { storeToRefs } from 'pinia'
import type { ToastMessageOptions } from 'primevue/toast'
import { useToast } from 'primevue/usetoast'
import {
  computed,
  nextTick,
  onBeforeUnmount,
  onMounted,
  ref,
  watch,
  watchEffect
} from 'vue'
import { useI18n } from 'vue-i18n'

import { runWhenGlobalIdle } from '@/base/common/async'
import MenuHamburger from '@/components/MenuHamburger.vue'
import UnloadWindowConfirmDialog from '@/components/dialog/UnloadWindowConfirmDialog.vue'
import GraphCanvas from '@/components/graph/GraphCanvas.vue'
import GlobalToast from '@/components/toast/GlobalToast.vue'
import InviteAcceptedToast from '@/platform/workspace/components/toasts/InviteAcceptedToast.vue'
import RerouteMigrationToast from '@/components/toast/RerouteMigrationToast.vue'
import { useBrowserTabTitle } from '@/composables/useBrowserTabTitle'
import { useCoreCommands } from '@/composables/useCoreCommands'
import { useQueuePolling } from '@/platform/remote/comfyui/useQueuePolling'
import { useErrorHandling } from '@/composables/useErrorHandling'
import { useProgressFavicon } from '@/composables/useProgressFavicon'
import { SERVER_CONFIG_ITEMS } from '@/constants/serverConfig'
import type { ServerConfig, ServerConfigValue } from '@/constants/serverConfig'
import { i18n, loadLocale } from '@/i18n'
import AssetExportProgressDialog from '@/platform/assets/components/AssetExportProgressDialog.vue'
import ModelImportProgressDialog from '@/platform/assets/components/ModelImportProgressDialog.vue'
import { isCloud, isDesktop } from '@/platform/distribution/types'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useTelemetry } from '@/platform/telemetry'
import { useFrontendVersionMismatchWarning } from '@/platform/updates/common/useFrontendVersionMismatchWarning'
import { useVersionCompatibilityStore } from '@/platform/updates/common/versionCompatibilityStore'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import type { StatusWsMessageStatus } from '@/schemas/apiSchema'
import { api } from '@/scripts/api'
import { app } from '@/scripts/app'
import { setupAutoQueueHandler } from '@/services/autoQueueService'
import { useKeybindingService } from '@/platform/keybindings/keybindingService'
import { useAppMode } from '@/composables/useAppMode'
import { useAssetsStore } from '@/stores/assetsStore'
import { useCommandStore } from '@/stores/commandStore'
import { useExecutionStore } from '@/stores/executionStore'
import { useFirebaseAuthStore } from '@/stores/firebaseAuthStore'
import { useMenuItemStore } from '@/stores/menuItemStore'
import { useModelStore } from '@/stores/modelStore'
import { useNodeDefStore, useNodeFrequencyStore } from '@/stores/nodeDefStore'
import {
  useQueuePendingTaskCountStore,
  useQueueStore
} from '@/stores/queueStore'
import { useServerConfigStore } from '@/stores/serverConfigStore'
import { useBottomPanelStore } from '@/stores/workspace/bottomPanelStore'
import { useColorPaletteStore } from '@/stores/workspace/colorPaletteStore'
import { useSidebarTabStore } from '@/stores/workspace/sidebarTabStore'
import { electronAPI } from '@/utils/envUtil'
import BuilderToolbar from '@/components/builder/BuilderToolbar.vue'
import LinearView from '@/views/LinearView.vue'
import ManagerProgressToast from '@/workbench/extensions/manager/components/ManagerProgressToast.vue'

setupAutoQueueHandler()
useProgressFavicon()
useBrowserTabTitle()

const { t } = useI18n()
const toast = useToast()
const settingStore = useSettingStore()
const executionStore = useExecutionStore()
const colorPaletteStore = useColorPaletteStore()
const queueStore = useQueueStore()
const assetsStore = useAssetsStore()
const versionCompatibilityStore = useVersionCompatibilityStore()
const graphCanvasContainerRef = ref<HTMLDivElement | null>(null)
const { isBuilderMode } = useAppMode()
const { linearMode } = storeToRefs(useCanvasStore())

const telemetry = useTelemetry()
const firebaseAuthStore = useFirebaseAuthStore()
let hasTrackedLogin = false

watch(
  () => colorPaletteStore.completedActivePalette,
  (newTheme) => {
    const DARK_THEME_CLASS = 'dark-theme'
    if (newTheme.light_theme) {
      document.body.classList.remove(DARK_THEME_CLASS)
    } else {
      document.body.classList.add(DARK_THEME_CLASS)
    }

    if (isDesktop) {
      electronAPI().changeTheme({
        color: 'rgba(0, 0, 0, 0)',
        symbolColor: newTheme.colors.comfy_base['input-text']
      })
    }
  },
  { immediate: true }
)

if (isDesktop) {
  watch(
    () => queueStore.tasks,
    (newTasks, oldTasks) => {
      // Report tasks that previously running but are now completed (i.e. in history)
      const oldRunningTaskIds = new Set(
        oldTasks.filter((task) => task.isRunning).map((task) => task.jobId)
      )
      newTasks
        .filter((task) => oldRunningTaskIds.has(task.jobId) && task.isHistory)
        .forEach((task) => {
          electronAPI().Events.incrementUserProperty(
            `execution:${task.displayStatus.toLowerCase()}`,
            1
          )
          electronAPI().Events.trackEvent('execution', {
            status: task.displayStatus.toLowerCase()
          })
        })
    },
    { deep: true }
  )
}

watchEffect(() => {
  const fontSize = settingStore.get('Comfy.TextareaWidget.FontSize')
  document.documentElement.style.setProperty(
    '--comfy-textarea-font-size',
    `${fontSize}px`
  )
})

watchEffect(() => {
  const padding = settingStore.get('Comfy.TreeExplorer.ItemPadding')
  document.documentElement.style.setProperty(
    '--comfy-tree-explorer-item-padding',
    `${padding}px`
  )
})

watchEffect(async () => {
  const locale = settingStore.get('Comfy.Locale')
  if (locale) {
    // Load the locale dynamically if not already loaded
    try {
      await loadLocale(locale)
      // Type assertion is safe here as loadLocale validates the locale exists
      i18n.global.locale.value = locale as typeof i18n.global.locale.value
    } catch (error) {
      console.error(`Failed to switch to locale "${locale}":`, error)
    }
  }
})

const useNewMenu = computed(() => {
  return settingStore.get('Comfy.UseNewMenu')
})
watchEffect(() => {
  if (useNewMenu.value === 'Disabled') {
    app.ui.menuContainer.style.setProperty('display', 'block')
    app.ui.restoreMenuPosition()
  } else {
    app.ui.menuContainer.style.setProperty('display', 'none')
  }
})

watchEffect(() => {
  queueStore.maxHistoryItems = settingStore.get('Comfy.Queue.MaxHistoryItems')
})

const coreCommands = useCoreCommands()
useCommandStore().registerCommands(coreCommands)
useMenuItemStore().registerCoreMenuCommands()
useKeybindingService().registerCoreKeybindings()
useSidebarTabStore().registerCoreSidebarTabs()
void useBottomPanelStore().registerCoreBottomPanelTabs()

useQueuePolling()
const queuePendingTaskCountStore = useQueuePendingTaskCountStore()
const sidebarTabStore = useSidebarTabStore()

const onStatus = async (e: CustomEvent<StatusWsMessageStatus>) => {
  queuePendingTaskCountStore.update(e)
  await queueStore.update()
  // Only update assets if the assets sidebar is currently open
  // When sidebar is closed, AssetsSidebarTab.vue will refresh on mount
  if (sidebarTabStore.activeSidebarTabId === 'assets' || linearMode.value) {
    await assetsStore.updateHistory()
  }
}

const onExecutionSuccess = async () => {
  await queueStore.update()
  // Only update assets if the assets sidebar is currently open
  // When sidebar is closed, AssetsSidebarTab.vue will refresh on mount
  if (sidebarTabStore.activeSidebarTabId === 'assets' || linearMode.value) {
    await assetsStore.updateHistory()
  }
}

const reconnectingMessage: ToastMessageOptions = {
  severity: 'error',
  summary: t('g.reconnecting')
}

const onReconnecting = () => {
  if (!settingStore.get('Comfy.Toast.DisableReconnectingToast')) {
    toast.remove(reconnectingMessage)
    toast.add(reconnectingMessage)
  }
}

const onReconnected = () => {
  if (!settingStore.get('Comfy.Toast.DisableReconnectingToast')) {
    toast.remove(reconnectingMessage)
    toast.add({
      severity: 'success',
      summary: t('g.reconnected'),
      life: 2000
    })
  }
}

useEventListener(api, 'status', onStatus)
useEventListener(api, 'execution_success', onExecutionSuccess)
useEventListener(api, 'reconnecting', onReconnecting)
useEventListener(api, 'reconnected', onReconnected)

onMounted(() => {
  executionStore.bindExecutionEvents()

  try {
    // Relocate the legacy menu container to the graph canvas container so it is below other elements
    graphCanvasContainerRef.value?.prepend(app.ui.menuContainer)
  } catch (e) {
    console.error('Failed to init ComfyUI frontend', e)
  }
})

onBeforeUnmount(() => {
  executionStore.unbindExecutionEvents()
})

useEventListener(window, 'keydown', useKeybindingService().keybindHandler)

const { wrapWithErrorHandling, wrapWithErrorHandlingAsync } = useErrorHandling()

// Initialize version mismatch warning in setup context
// It will be triggered automatically when the store is ready
useFrontendVersionMismatchWarning({ immediate: true })

void nextTick(() => {
  versionCompatibilityStore.initialize().catch((error) => {
    console.warn('Version compatibility check failed:', error)
  })
})

const onGraphReady = () => {
  runWhenGlobalIdle(() => {
    // Track user login when app is ready in graph view (cloud only)
    if (isCloud && firebaseAuthStore.isAuthenticated && !hasTrackedLogin) {
      telemetry?.trackUserLoggedIn()
      hasTrackedLogin = true
    }

    // Set up page visibility tracking (cloud only)
    if (isCloud && telemetry) {
      useEventListener(document, 'visibilitychange', () => {
        telemetry.trackPageVisibilityChanged({
          visibility_state: document.visibilityState as 'visible' | 'hidden'
        })
      })
    }

    // Set up tab count tracking (cloud only)
    if (isCloud && telemetry) {
      const tabCountChannel = new BroadcastChannel('comfyui-tab-count')
      const activeTabs = new Map<string, number>()
      const currentTabId = crypto.randomUUID()

      // Listen for heartbeats from other tabs
      tabCountChannel.onmessage = (event) => {
        if (
          event.data.type === 'heartbeat' &&
          event.data.tabId !== currentTabId
        ) {
          activeTabs.set(event.data.tabId, Date.now())
        }
      }

      // 5-minute heartbeat interval
      useIntervalFn(() => {
        const now = Date.now()

        // Clean up stale tabs (no heartbeat for 45 seconds)
        activeTabs.forEach((lastHeartbeat, tabId) => {
          if (now - lastHeartbeat > 45000) {
            activeTabs.delete(tabId)
          }
        })

        // Broadcast our heartbeat
        tabCountChannel.postMessage({
          type: 'heartbeat',
          tabId: currentTabId
        })

        // Track tab count (include current tab)
        const tabCount = activeTabs.size + 1
        telemetry.trackTabCount({ tab_count: tabCount })
      }, 60000 * 5)

      // Send initial heartbeat
      tabCountChannel.postMessage({ type: 'heartbeat', tabId: currentTabId })
    }

    // Setting values now available after comfyApp.setup.
    // Load keybindings.
    wrapWithErrorHandling(useKeybindingService().registerUserKeybindings)()

    // Load server config
    wrapWithErrorHandling(useServerConfigStore().loadServerConfig)(
      SERVER_CONFIG_ITEMS as ServerConfig<ServerConfigValue>[],
      settingStore.get('Comfy.Server.ServerConfigValues')
    )

    // Load model folders
    void wrapWithErrorHandlingAsync(useModelStore().loadModelFolders)()

    // Non-blocking load of node frequencies
    void wrapWithErrorHandlingAsync(
      useNodeFrequencyStore().loadNodeFrequencies
    )()

    // Node defs now available after comfyApp.setup.
    // Explicitly initialize nodeSearchService to avoid indexing delay when
    // node search is triggered
    useNodeDefStore().nodeSearchService.searchNode('')
  }, 1000)
}
</script>

<style scoped>
.comfyui-body {
  grid-template-columns: auto 1fr auto;
  grid-template-rows: auto 1fr auto;
}

/**
  +------------------+------------------+------------------+
  |                                                        |
  |  .comfyui-body-                                        |
  |       top                                              |
  | (spans all cols)                                       |
  |                                                        |
  +------------------+------------------+------------------+
  |                  |                  |                  |
  | .comfyui-body-   |   #graph-canvas  | .comfyui-body-   |
  |      left        |                  |      right       |
  |                  |                  |                  |
  |                  |                  |                  |
  +------------------+------------------+------------------+
  |                                                        |
  |  .comfyui-body-                                        |
  |      bottom                                            |
  | (spans all cols)                                       |
  |                                                        |
  +------------------+------------------+------------------+
*/

.comfyui-body-top {
  order: -5;
  /* Span across all columns */
  grid-column: 1/-1;
  /* Position at the first row */
  grid-row: 1;
  /* Top menu bar dropdown needs to be above of graph canvas splitter overlay which is z-index: 999 */
  /* Top menu bar z-index needs to be higher than bottom menu bar z-index as by default
  pysssss's image feed is located at body-bottom, and it can overlap with the queue button, which
  is located in body-top. */
  z-index: 1001;
  display: flex;
  flex-direction: column;
}

.comfyui-body-left {
  order: -4;
  /* Position in the first column */
  grid-column: 1;
  /* Position below the top element */
  grid-row: 2;
  z-index: 10;
  display: flex;
}

.graph-canvas-container {
  width: 100%;
  height: 100%;
  order: -3;
  grid-column: 2;
  grid-row: 2;
  position: relative;
  overflow: clip;
}

.comfyui-body-right {
  order: -2;
  z-index: 10;
  grid-column: 3;
  grid-row: 2;
}

.comfyui-body-bottom {
  order: 4;
  /* Span across all columns */
  grid-column: 1/-1;
  grid-row: 3;
  /* Bottom menu bar dropdown needs to be above of graph canvas splitter overlay which is z-index: 999 */
  z-index: 1000;
  display: flex;
  flex-direction: column;
}
</style>
