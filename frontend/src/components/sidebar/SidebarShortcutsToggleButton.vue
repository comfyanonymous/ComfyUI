<template>
  <SidebarIcon
    icon="icon-[lucide--keyboard]"
    :label="$t('shortcuts.shortcuts')"
    :tooltip="tooltipText"
    :selected="isShortcutsPanelVisible"
    @click="toggleShortcutsPanel"
  />
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import { useTelemetry } from '@/platform/telemetry'
import { useCommandStore } from '@/stores/commandStore'
import { useBottomPanelStore } from '@/stores/workspace/bottomPanelStore'

import SidebarIcon from './SidebarIcon.vue'

const { t } = useI18n()
const bottomPanelStore = useBottomPanelStore()
const commandStore = useCommandStore()
const command = commandStore.getCommand('Workspace.ToggleBottomPanel.Shortcuts')
const { formatKeySequence } = commandStore

const isShortcutsPanelVisible = computed(
  () => bottomPanelStore.activePanel === 'shortcuts'
)

const tooltipText = computed(
  () => `${t('shortcuts.keyboardShortcuts')} (${formatKeySequence(command)})`
)

/**
 * Toggle keyboard shortcuts panel and track UI button click.
 */
const toggleShortcutsPanel = () => {
  useTelemetry()?.trackUiButtonClicked({
    button_id: 'sidebar_shortcuts_panel_toggled'
  })
  bottomPanelStore.togglePanel('shortcuts')
}
</script>
