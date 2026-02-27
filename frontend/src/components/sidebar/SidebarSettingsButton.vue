<template>
  <SidebarIcon
    icon="icon-[lucide--settings]"
    :label="$t('g.settings')"
    :tooltip="tooltipText"
    @click="showSettingsDialog"
  />
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import { useTelemetry } from '@/platform/telemetry'
import { useCommandStore } from '@/stores/commandStore'

import SidebarIcon from './SidebarIcon.vue'

const { t } = useI18n()
const { getCommand, formatKeySequence } = useCommandStore()
const command = getCommand('Comfy.ShowSettingsDialog')

const tooltipText = computed(
  () => `${t('g.settings')} (${formatKeySequence(command)})`
)

/**
 * Toggle keyboard shortcuts panel and track UI button click.
 */
const showSettingsDialog = () => {
  command.function()
  useTelemetry()?.trackUiButtonClicked({
    button_id: 'sidebar_settings_button_clicked'
  })
}
</script>
