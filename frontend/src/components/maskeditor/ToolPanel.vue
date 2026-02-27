<template>
  <div class="h-full z-8888 flex flex-col justify-between bg-comfy-menu-bg">
    <div class="flex flex-col">
      <div
        v-for="tool in allTools"
        :key="tool"
        :class="[
          'maskEditor_toolPanelContainer hover:bg-secondary-background-hover',
          { maskEditor_toolPanelContainerSelected: currentTool === tool }
        ]"
        @click="onToolSelect(tool)"
      >
        <div
          class="flex items-center justify-center"
          v-html="iconsHtml[tool]"
        ></div>
        <div class="maskEditor_toolPanelIndicator"></div>
      </div>
    </div>

    <div
      class="flex flex-col items-center cursor-pointer rounded-md mb-2 transition-colors duration-200 hover:bg-secondary-background-hover"
      :title="t('maskEditor.clickToResetZoom')"
      @click="onResetZoom"
    >
      <span class="text-sm text-text-secondary">{{ zoomText }}</span>
      <span class="text-xs text-text-secondary">{{ dimensionsText }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import type { useToolManager } from '@/composables/maskeditor/useToolManager'
import { iconsHtml } from '@/extensions/core/maskeditor/constants'
import type { Tools } from '@/extensions/core/maskeditor/types'
import { allTools } from '@/extensions/core/maskeditor/types'
import { useMaskEditorStore } from '@/stores/maskEditorStore'

const { toolManager } = defineProps<{
  toolManager: ReturnType<typeof useToolManager>
}>()

const { t } = useI18n()
const store = useMaskEditorStore()

const onToolSelect = (tool: Tools) => {
  toolManager.switchTool(tool)
}

const currentTool = computed(() => store.currentTool)

const zoomText = computed(() => `${Math.round(store.displayZoomRatio * 100)}%`)
const dimensionsText = computed(() => {
  const img = store.image
  return img ? `${img.width}x${img.height}` : ' '
})

const onResetZoom = () => {
  store.resetZoom()
}
</script>
