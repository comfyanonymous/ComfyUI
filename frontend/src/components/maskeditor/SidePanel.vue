<template>
  <div
    class="flex flex-col gap-3 pb-3 h-full !items-stretch bg-[var(--comfy-menu-bg)] overflow-y-auto w-55 px-2.5"
  >
    <div class="w-full min-h-full">
      <SettingsPanelContainer />

      <div class="w-full h-0.5 bg-[var(--border-color)] mt-6 mb-1.5" />

      <ImageLayerSettingsPanel :tool-manager="toolManager" />
    </div>
  </div>
</template>

<script setup lang="ts">
import type { useToolManager } from '@/composables/maskeditor/useToolManager'

import ImageLayerSettingsPanel from './ImageLayerSettingsPanel.vue'
import SettingsPanelContainer from './SettingsPanelContainer.vue'

const { toolManager } = defineProps<{
  toolManager?: ReturnType<typeof useToolManager>
}>()
</script>
