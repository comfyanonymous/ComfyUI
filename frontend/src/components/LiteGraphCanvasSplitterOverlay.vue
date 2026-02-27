<template>
  <div
    class="w-full h-full absolute top-0 left-0 z-999 pointer-events-none flex flex-col"
  >
    <slot name="workflow-tabs" />

    <div
      class="pointer-events-none flex flex-1 overflow-hidden"
      :class="{
        'flex-row': sidebarLocation === 'left',
        'flex-row-reverse': sidebarLocation === 'right'
      }"
    >
      <div class="side-toolbar-container">
        <slot name="side-toolbar" />
      </div>

      <Splitter
        :key="splitterRefreshKey"
        class="bg-transparent pointer-events-none border-none flex-1 overflow-hidden"
        :state-key="sidebarStateKey"
        state-storage="local"
        @resizestart="onResizestart"
      >
        <!-- First panel: sidebar when left, properties when right -->
        <SplitterPanel
          v-if="
            !focusMode && (sidebarLocation === 'left' || showOffsideSplitter)
          "
          :class="
            sidebarLocation === 'left'
              ? cn(
                  'side-bar-panel bg-comfy-menu-bg pointer-events-auto',
                  sidebarPanelVisible && 'min-w-78'
                )
              : 'bg-comfy-menu-bg pointer-events-auto'
          "
          :min-size="sidebarLocation === 'left' ? 10 : 15"
          :size="20"
          :style="firstPanelStyle"
          :role="sidebarLocation === 'left' ? 'complementary' : undefined"
          :aria-label="
            sidebarLocation === 'left' ? t('sideToolbar.sidebar') : undefined
          "
        >
          <slot
            v-if="sidebarLocation === 'left' && sidebarPanelVisible"
            name="side-bar-panel"
          />
          <slot
            v-else-if="sidebarLocation === 'right'"
            name="right-side-panel"
          />
        </SplitterPanel>

        <!-- Main panel (always present) -->
        <SplitterPanel :size="80" class="flex flex-col">
          <slot name="topmenu" :sidebar-panel-visible />

          <Splitter
            class="bg-transparent pointer-events-none border-none splitter-overlay-bottom mr-1 mb-1 ml-1 flex-1"
            layout="vertical"
            :pt:gutter="
              cn(
                'rounded-tl-lg rounded-tr-lg ',
                !(bottomPanelVisible && !focusMode) && 'hidden'
              )
            "
            state-key="bottom-panel-splitter"
            state-storage="local"
            @resizestart="onResizestart"
          >
            <SplitterPanel class="graph-canvas-panel relative">
              <slot name="graph-canvas-panel" />
            </SplitterPanel>
            <SplitterPanel
              v-show="bottomPanelVisible && !focusMode"
              class="bottom-panel border border-(--p-panel-border-color) max-w-full overflow-x-auto bg-comfy-menu-bg pointer-events-auto rounded-lg"
            >
              <slot name="bottom-panel" />
            </SplitterPanel>
          </Splitter>
        </SplitterPanel>

        <!-- Last panel: properties when left, sidebar when right -->
        <SplitterPanel
          v-if="
            !focusMode && (sidebarLocation === 'right' || showOffsideSplitter)
          "
          :class="
            sidebarLocation === 'right'
              ? cn(
                  'side-bar-panel bg-comfy-menu-bg pointer-events-auto',
                  sidebarPanelVisible && 'min-w-78'
                )
              : 'bg-comfy-menu-bg pointer-events-auto'
          "
          :min-size="sidebarLocation === 'right' ? 10 : 15"
          :size="20"
          :style="lastPanelStyle"
          :role="sidebarLocation === 'right' ? 'complementary' : undefined"
          :aria-label="
            sidebarLocation === 'right' ? t('sideToolbar.sidebar') : undefined
          "
        >
          <slot v-if="sidebarLocation === 'left'" name="right-side-panel" />
          <slot
            v-else-if="sidebarLocation === 'right' && sidebarPanelVisible"
            name="side-bar-panel"
          />
        </SplitterPanel>
      </Splitter>
    </div>
  </div>
</template>

<script setup lang="ts">
import { cn } from '@comfyorg/tailwind-utils'
import { storeToRefs } from 'pinia'
import Splitter from 'primevue/splitter'
import type { SplitterResizeStartEvent } from 'primevue/splitter'
import SplitterPanel from 'primevue/splitterpanel'
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import { useSettingStore } from '@/platform/settings/settingStore'
import { useAppMode } from '@/composables/useAppMode'
import { useBottomPanelStore } from '@/stores/workspace/bottomPanelStore'
import { useRightSidePanelStore } from '@/stores/workspace/rightSidePanelStore'
import { useSidebarTabStore } from '@/stores/workspace/sidebarTabStore'
import { useWorkspaceStore } from '@/stores/workspaceStore'

const workspaceStore = useWorkspaceStore()
const settingStore = useSettingStore()
const rightSidePanelStore = useRightSidePanelStore()
const sidebarTabStore = useSidebarTabStore()
const { t } = useI18n()
const sidebarLocation = computed<'left' | 'right'>(() =>
  settingStore.get('Comfy.Sidebar.Location')
)

const unifiedWidth = computed(() =>
  settingStore.get('Comfy.Sidebar.UnifiedWidth')
)

const { focusMode } = storeToRefs(workspaceStore)

const { mode } = useAppMode()
const { activeSidebarTabId, activeSidebarTab } = storeToRefs(sidebarTabStore)
const { bottomPanelVisible } = storeToRefs(useBottomPanelStore())
const { isOpen: rightSidePanelVisible } = storeToRefs(rightSidePanelStore)
const showOffsideSplitter = computed(
  () => rightSidePanelVisible.value || mode.value === 'builder:select'
)

const sidebarPanelVisible = computed(() => activeSidebarTab.value !== null)

const sidebarStateKey = computed(() => {
  return unifiedWidth.value
    ? 'unified-sidebar'
    : // When no tab is active, use a default key to maintain state
      (activeSidebarTabId.value ?? 'default-sidebar')
})

/**
 * Avoid triggering default behaviors during drag-and-drop, such as text selection.
 */
function onResizestart({ originalEvent: event }: SplitterResizeStartEvent) {
  event.preventDefault()
}

/*
 * Force refresh the splitter when right panel visibility or sidebar location changes
 * to recalculate the width and panel order
 */
const splitterRefreshKey = computed(() => {
  return `main-splitter${rightSidePanelVisible.value ? '-with-right-panel' : ''}-${sidebarLocation.value}`
})

const firstPanelStyle = computed(() => {
  if (sidebarLocation.value === 'left') {
    return { display: sidebarPanelVisible.value ? 'flex' : 'none' }
  }
  return undefined
})

const lastPanelStyle = computed(() => {
  if (sidebarLocation.value === 'right') {
    return { display: sidebarPanelVisible.value ? 'flex' : 'none' }
  }
  return undefined
})
</script>

<style scoped>
:deep(.p-splitter-gutter) {
  pointer-events: auto;
}

:deep(.p-splitter-gutter:hover),
:deep(.p-splitter-gutter[data-p-gutter-resizing='true']) {
  transition: background-color 0.2s ease 300ms;
  background-color: var(--p-primary-color);
}

/* Hide sidebar gutter when sidebar is not visible */
:deep(.side-bar-panel[style*='display: none'] + .p-splitter-gutter),
:deep(.p-splitter-gutter + .side-bar-panel[style*='display: none']) {
  display: none;
}

.splitter-overlay-bottom :deep(.p-splitter-gutter) {
  transform: translateY(5px);
}
</style>
