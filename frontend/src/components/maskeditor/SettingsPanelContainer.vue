<template>
  <div class="maskEditor_sidePanel">
    <div class="maskEditor_sidePanelContainer">
      <component :is="currentPanelComponent" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { Component } from 'vue'

import { Tools } from '@/extensions/core/maskeditor/types'
import { useMaskEditorStore } from '@/stores/maskEditorStore'

import BrushSettingsPanel from './BrushSettingsPanel.vue'
import ColorSelectSettingsPanel from './ColorSelectSettingsPanel.vue'
import PaintBucketSettingsPanel from './PaintBucketSettingsPanel.vue'

const currentPanelComponent = computed<Component>(() => {
  const tool = useMaskEditorStore().currentTool

  if (tool === Tools.MaskBucket) {
    return PaintBucketSettingsPanel
  } else if (tool === Tools.MaskColorFill) {
    return ColorSelectSettingsPanel
  } else {
    return BrushSettingsPanel
  }
})
</script>
