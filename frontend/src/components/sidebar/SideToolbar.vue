<template>
  <nav
    ref="sideToolbarRef"
    data-testid="side-toolbar"
    class="side-tool-bar-container flex h-full flex-col items-center bg-transparent [.floating-sidebar]:-mr-2"
    :class="{
      'small-sidebar': isSmall,
      'connected-sidebar pointer-events-auto': isConnected,
      'floating-sidebar': !isConnected,
      'overflowing-sidebar': isOverflowing,
      'border-r border-[var(--interface-stroke)] shadow-interface': isConnected
    }"
  >
    <div
      :class="
        isOverflowing
          ? 'side-tool-bar-container overflow-y-auto'
          : 'flex flex-col h-full'
      "
    >
      <div ref="topToolbarRef" :class="groupClasses">
        <ComfyMenuButton />
        <SidebarIcon
          v-for="tab in tabs"
          :key="tab.id"
          :icon="tab.icon"
          :icon-badge="tab.iconBadge"
          :tooltip="tab.tooltip"
          :tooltip-suffix="getTabTooltipSuffix(tab)"
          :label="tab.label || tab.title"
          :is-small="isSmall"
          :selected="tab.id === selectedTab?.id"
          :class="tab.id + '-tab-button'"
          @click="onTabClick(tab)"
        />
        <SidebarTemplatesButton />
      </div>

      <div ref="bottomToolbarRef" class="mt-auto" :class="groupClasses">
        <SidebarLogoutIcon
          v-if="userStore.isMultiUserServer"
          :is-small="isSmall"
        />
        <SidebarHelpCenterIcon v-if="!isIntegratedTabBar" :is-small="isSmall" />
        <SidebarBottomPanelToggleButton v-if="!isCloud" :is-small="isSmall" />
        <SidebarShortcutsToggleButton :is-small="isSmall" />
        <SidebarSettingsButton :is-small="isSmall" />
      </div>
    </div>
    <HelpCenterPopups :is-small="isSmall" />
  </nav>
</template>

<script setup lang="ts">
import { useResizeObserver } from '@vueuse/core'
import { debounce } from 'es-toolkit/compat'
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'

import HelpCenterPopups from '@/components/helpcenter/HelpCenterPopups.vue'
import ComfyMenuButton from '@/components/sidebar/ComfyMenuButton.vue'
import SidebarBottomPanelToggleButton from '@/components/sidebar/SidebarBottomPanelToggleButton.vue'
import SidebarSettingsButton from '@/components/sidebar/SidebarSettingsButton.vue'
import SidebarShortcutsToggleButton from '@/components/sidebar/SidebarShortcutsToggleButton.vue'
import { isCloud } from '@/platform/distribution/types'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useTelemetry } from '@/platform/telemetry'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useCommandStore } from '@/stores/commandStore'
import { useKeybindingStore } from '@/platform/keybindings/keybindingStore'
import { useUserStore } from '@/stores/userStore'
import { useWorkspaceStore } from '@/stores/workspaceStore'
import type { SidebarTabExtension } from '@/types/extensionTypes'
import { cn } from '@/utils/tailwindUtil'

import SidebarHelpCenterIcon from './SidebarHelpCenterIcon.vue'
import SidebarIcon from './SidebarIcon.vue'
import SidebarLogoutIcon from './SidebarLogoutIcon.vue'
import SidebarTemplatesButton from './SidebarTemplatesButton.vue'

const { t } = useI18n()
const workspaceStore = useWorkspaceStore()
const settingStore = useSettingStore()
const userStore = useUserStore()
const commandStore = useCommandStore()
const canvasStore = useCanvasStore()
const sideToolbarRef = ref<HTMLElement>()
const topToolbarRef = ref<HTMLElement>()
const bottomToolbarRef = ref<HTMLElement>()

const isSmall = computed(
  () => settingStore.get('Comfy.Sidebar.Size') === 'small'
)
const sidebarLocation = computed<'left' | 'right'>(() =>
  settingStore.get('Comfy.Sidebar.Location')
)
const sidebarStyle = computed(() => settingStore.get('Comfy.Sidebar.Style'))
const isIntegratedTabBar = computed(
  () => settingStore.get('Comfy.UI.TabBarLayout') === 'Integrated'
)
const isConnected = computed(
  () =>
    selectedTab.value ||
    isOverflowing.value ||
    sidebarStyle.value === 'connected'
)

const tabs = computed(() => workspaceStore.getSidebarTabs())
const selectedTab = computed(() => workspaceStore.sidebarTab.activeSidebarTab)

/**
 * Handle sidebar tab icon click.
 * - Emits UI button telemetry for known tabs
 * - Delegates to the corresponding toggle command
 */
const onTabClick = async (item: SidebarTabExtension) => {
  const telemetry = useTelemetry()

  const isNodeLibraryTab = item.id === 'node-library'
  const isModelLibraryTab = item.id === 'model-library'
  const isWorkflowsTab = item.id === 'workflows'
  const isAssetsTab = item.id === 'assets'

  if (isNodeLibraryTab)
    telemetry?.trackUiButtonClicked({
      button_id: 'sidebar_tab_node_library_selected'
    })
  else if (isModelLibraryTab)
    telemetry?.trackUiButtonClicked({
      button_id: 'sidebar_tab_model_library_selected'
    })
  else if (isWorkflowsTab)
    telemetry?.trackUiButtonClicked({
      button_id: 'sidebar_tab_workflows_selected'
    })
  else if (isAssetsTab)
    telemetry?.trackUiButtonClicked({
      button_id: 'sidebar_tab_assets_media_selected'
    })

  await commandStore.commands
    .find((cmd) => cmd.id === `Workspace.ToggleSidebarTab.${item.id}`)
    ?.function?.()
}

const keybindingStore = useKeybindingStore()
const getTabTooltipSuffix = (tab: SidebarTabExtension) => {
  const shortcut = keybindingStore
    .getKeybindingByCommandId(`Workspace.ToggleSidebarTab.${tab.id}`)
    ?.combo.toString()
  return shortcut ? t('g.shortcutSuffix', { shortcut }) : ''
}

const isOverflowing = ref(false)
const groupClasses = computed(() =>
  cn(
    'sidebar-item-group flex flex-col items-center overflow-hidden flex-shrink-0',
    !isConnected.value && 'rounded-lg shadow-interface pointer-events-auto'
  )
)

const ENTER_OVERFLOW_MARGIN = 20
const EXIT_OVERFLOW_MARGIN = 50

const checkOverflow = debounce(() => {
  if (!sideToolbarRef.value || !topToolbarRef.value || !bottomToolbarRef.value)
    return

  const containerHeight = sideToolbarRef.value.clientHeight
  const topHeight = topToolbarRef.value.scrollHeight
  const bottomHeight = bottomToolbarRef.value.scrollHeight
  const contentHeight = topHeight + bottomHeight

  if (isOverflowing.value) {
    isOverflowing.value = containerHeight < contentHeight + EXIT_OVERFLOW_MARGIN
  } else {
    isOverflowing.value =
      containerHeight < contentHeight + ENTER_OVERFLOW_MARGIN
  }
}, 16)

onMounted(() => {
  if (!sideToolbarRef.value) return

  const overflowObserver = useResizeObserver(
    sideToolbarRef.value,
    checkOverflow
  )

  checkOverflow()

  onBeforeUnmount(() => {
    overflowObserver.stop()
  })

  watch(
    [isSmall, sidebarLocation],
    async () => {
      if (canvasStore.canvas) {
        if (sidebarLocation.value === 'left') {
          await nextTick()
          canvasStore.canvas.fpsInfoLocation = [
            sideToolbarRef.value?.getBoundingClientRect()?.right,
            null
          ]
        } else {
          canvasStore.canvas.fpsInfoLocation = null
        }
        canvasStore.canvas.setDirty(false, true)
      }
    },
    { immediate: true }
  )
})
</script>

<style>
/* Global CSS variables for sidebar
 * These variables need to be global (not scoped) because they are used by
 * teleported components like WhatsNewPopup that render outside the sidebar
 * but need to reference sidebar dimensions for proper positioning.
 */
:root {
  --sidebar-padding: 4px;
  --sidebar-icon-size: 1rem;

  --sidebar-default-floating-width: 48px;
  --sidebar-default-connected-width: calc(
    var(--sidebar-default-floating-width) + var(--sidebar-padding) * 2
  );
  --sidebar-default-item-height: 56px;

  --sidebar-small-floating-width: 48px;
  --sidebar-small-connected-width: calc(
    var(--sidebar-small-floating-width) + var(--sidebar-padding) * 2
  );
  --sidebar-small-item-height: 48px;

  --sidebar-width: var(--sidebar-default-floating-width);
  --sidebar-item-height: var(--sidebar-default-item-height);
}

:root:has(.side-tool-bar-container.small-sidebar) {
  --sidebar-width: var(--sidebar-small-floating-width);
  --sidebar-item-height: var(--sidebar-small-item-height);
}

:root:has(.side-tool-bar-container.connected-sidebar) {
  --sidebar-width: var(--sidebar-default-connected-width);
}

:root:has(.side-tool-bar-container.small-sidebar.connected-sidebar) {
  --sidebar-width: var(--sidebar-small-connected-width);
}
</style>

<style scoped>
.floating-sidebar {
  padding: var(--sidebar-padding);
}

.floating-sidebar .sidebar-item-group {
  border-color: var(--p-panel-border-color);
}

.connected-sidebar {
  padding: var(--sidebar-padding) 0;
  background-color: var(--comfy-menu-bg);
}

.sidebar-item-group {
  background-color: var(--comfy-menu-bg);
  border: 1px solid transparent;
}

.overflowing-sidebar :deep(.comfy-menu-button-wrapper) {
  position: sticky;
  top: 0;
  z-index: 1;
  background-color: var(--comfy-menu-bg);
}
</style>
