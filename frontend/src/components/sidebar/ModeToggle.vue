<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { useKeybindingStore } from '@/platform/keybindings/keybindingStore'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useCommandStore } from '@/stores/commandStore'

const { t } = useI18n()
const canvasStore = useCanvasStore()
const keybindingStore = useKeybindingStore()

const keybindingSuffix = computed(() => {
  const shortcut = keybindingStore
    .getKeybindingByCommandId('Comfy.ToggleLinear')
    ?.combo.toString()
  return shortcut ? t('g.shortcutSuffix', { shortcut }) : ''
})

function toggleLinearMode() {
  useCommandStore().execute('Comfy.ToggleLinear', {
    metadata: { source: 'button' }
  })
}
</script>
<template>
  <div
    data-testid="mode-toggle"
    class="p-1 bg-secondary-background rounded-lg w-10"
  >
    <Button
      v-tooltip="{
        value: t('linearMode.linearMode') + keybindingSuffix,
        showDelay: 300,
        hideDelay: 300
      }"
      size="icon"
      :variant="canvasStore.linearMode ? 'inverted' : 'secondary'"
      @click="toggleLinearMode"
    >
      <i class="icon-[lucide--panels-top-left]" />
    </Button>
    <Button
      v-tooltip="{
        value: t('linearMode.graphMode') + keybindingSuffix,
        showDelay: 300,
        hideDelay: 300
      }"
      size="icon"
      :variant="canvasStore.linearMode ? 'secondary' : 'inverted'"
      @click="toggleLinearMode"
    >
      <i class="icon-[comfy--workflow]" />
    </Button>
  </div>
</template>
