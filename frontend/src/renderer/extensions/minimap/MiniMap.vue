<template>
  <div
    v-if="visible && initialized"
    ref="minimapRef"
    :class="
      cn(
        'minimap-main-container absolute right-0 bottom-[54px] z-1000 flex',
        isMobile ? 'flex-col' : 'flex-row'
      )
    "
  >
    <MiniMapPanel
      v-if="showOptionsPanel"
      :panel-styles="panelStyles"
      :node-colors="nodeColors"
      :show-links="showLinks"
      :show-groups="showGroups"
      :render-bypass="renderBypass"
      :render-error="renderError"
      :is-mobile="isMobile"
      @update-option="updateOption"
    />

    <div
      ref="containerRef"
      class="litegraph-minimap relative border border-interface-stroke bg-comfy-menu-bg shadow-interface"
      :style="containerStyles"
    >
      <Button
        class="absolute top-0 left-0 z-10"
        size="icon"
        variant="muted-textonly"
        :aria-label="$t('g.settings')"
        @click.stop="toggleOptionsPanel"
      >
        <i class="icon-[lucide--settings-2]" />
      </Button>
      <Button
        class="absolute top-0 right-0 z-10"
        size="icon"
        variant="muted-textonly"
        :aria-label="$t('g.close')"
        data-testid="close-minmap-button"
        @click.stop="() => commandStore.execute('Comfy.Canvas.ToggleMinimap')"
      >
        <i class="icon-[lucide--x]" />
      </Button>

      <hr
        class="absolute top-6 h-px border-0 bg-node-component-border"
        :style="{
          width: containerStyles.width
        }"
      />

      <canvas
        ref="canvasRef"
        :width="width"
        :height="height"
        class="minimap-canvas"
      />

      <div class="minimap-viewport" :style="viewportStyles" />

      <div
        class="absolute inset-0 touch-none"
        @pointerdown="handlePointerDown"
        @pointermove="handlePointerMove"
        @pointerup="handlePointerUp"
        @pointerleave="handlePointerUp"
        @pointercancel="handlePointerCancel"
        @wheel="handleWheel"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { breakpointsTailwind, useBreakpoints } from '@vueuse/core'
import { onMounted, onUnmounted, ref, useTemplateRef } from 'vue'

import Button from '@/components/ui/button/Button.vue'
import { useMinimap } from '@/renderer/extensions/minimap/composables/useMinimap'
import { useCommandStore } from '@/stores/commandStore'
import { cn } from '@/utils/tailwindUtil'

import MiniMapPanel from './MiniMapPanel.vue'

const isMobile = useBreakpoints(breakpointsTailwind).smaller('md')

const commandStore = useCommandStore()
const minimapRef = ref<HTMLDivElement>()
const containerRef = useTemplateRef<HTMLDivElement>('containerRef')
const canvasRef = useTemplateRef<HTMLCanvasElement>('canvasRef')

const {
  initialized,
  visible,
  containerStyles,
  viewportStyles,
  width,
  height,
  panelStyles,
  nodeColors,
  showLinks,
  showGroups,
  renderBypass,
  renderError,
  updateOption,
  destroy,
  handlePointerDown,
  handlePointerMove,
  handlePointerUp,
  handlePointerCancel,
  handleWheel,
  setMinimapRef
} = useMinimap({
  containerRefMaybe: containerRef,
  canvasRefMaybe: canvasRef
})

const showOptionsPanel = ref(false)

const toggleOptionsPanel = () => {
  showOptionsPanel.value = !showOptionsPanel.value
}

onMounted(() => {
  if (minimapRef.value) {
    setMinimapRef(minimapRef.value)
  }
})

onUnmounted(() => {
  destroy()
})
</script>

<style scoped>
.litegraph-minimap {
  overflow: hidden;
}

.minimap-canvas {
  display: block;
  width: 100%;
  height: 100%;
  pointer-events: none;
}

.minimap-viewport {
  position: absolute;
  top: 0;
  left: 0;
  pointer-events: none;
}
</style>
