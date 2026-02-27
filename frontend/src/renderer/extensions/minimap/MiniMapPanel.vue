<template>
  <div
    :class="
      cn(
        'minimap-panel flex flex-col gap-2 bg-comfy-menu-bg p-3 text-sm shadow-interface',
        isMobile ? 'mb-2' : 'mr-2'
      )
    "
    :style="panelStyles"
  >
    <div class="flex items-center gap-2">
      <Checkbox
        input-id="node-colors"
        name="node-colors"
        :model-value="nodeColors"
        binary
        @update:model-value="
          (value) => $emit('updateOption', 'Comfy.Minimap.NodeColors', value)
        "
      />
      <i class="icon-[lucide--palette]" />
      <label for="node-colors">{{ $t('minimap.nodeColors') }}</label>
    </div>

    <div class="flex items-center gap-2">
      <Checkbox
        input-id="show-links"
        name="show-links"
        :model-value="showLinks"
        binary
        @update:model-value="
          (value) => $emit('updateOption', 'Comfy.Minimap.ShowLinks', value)
        "
      />
      <i class="icon-[lucide--route]" />
      <label for="show-links">{{ $t('minimap.showLinks') }}</label>
    </div>

    <div class="flex items-center gap-2">
      <Checkbox
        input-id="show-groups"
        name="show-groups"
        :model-value="showGroups"
        binary
        @update:model-value="
          (value) => $emit('updateOption', 'Comfy.Minimap.ShowGroups', value)
        "
      />
      <i class="icon-[lucide--frame]" />
      <label for="show-groups">{{ $t('minimap.showGroups') }}</label>
    </div>

    <div class="flex items-center gap-2">
      <Checkbox
        input-id="render-bypass"
        name="render-bypass"
        :model-value="renderBypass"
        binary
        @update:model-value="
          (value) =>
            $emit('updateOption', 'Comfy.Minimap.RenderBypassState', value)
        "
      />
      <i class="icon-[lucide--circle-slash-2]" />
      <label for="render-bypass">{{ $t('minimap.renderBypassState') }}</label>
    </div>

    <div class="flex items-center gap-2">
      <Checkbox
        input-id="render-error"
        name="render-error"
        :model-value="renderError"
        binary
        @update:model-value="
          (value) =>
            $emit('updateOption', 'Comfy.Minimap.RenderErrorState', value)
        "
      />
      <i class="icon-[lucide--message-circle-warning]" />
      <label for="render-error">{{ $t('minimap.renderErrorState') }}</label>
    </div>
  </div>
</template>

<script setup lang="ts">
import Checkbox from 'primevue/checkbox'
import type { CSSProperties, Ref } from 'vue'

import type { MinimapSettingsKey } from '@/renderer/extensions/minimap/types'
import { cn } from '@/utils/tailwindUtil'

defineProps<{
  panelStyles: CSSProperties
  nodeColors: boolean
  showLinks: boolean
  showGroups: boolean
  renderBypass: boolean
  renderError: boolean
  isMobile: Ref<boolean> | boolean
}>()

defineEmits<{
  updateOption: [key: MinimapSettingsKey, value: boolean]
}>()
</script>
