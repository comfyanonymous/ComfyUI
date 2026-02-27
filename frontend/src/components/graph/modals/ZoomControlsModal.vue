<template>
  <div
    v-if="visible"
    class="absolute right-0 bottom-[62px] z-1300 flex w-[250px] justify-center border-0! bg-inherit!"
  >
    <div
      class="w-4/5 rounded-lg border border-interface-stroke bg-interface-panel-surface p-2 text-text-primary shadow-lg select-none"
      :style="filteredMinimapStyles"
      @click.stop
    >
      <div class="flex flex-col gap-1">
        <div
          class="flex cursor-pointer items-center justify-between rounded px-3 py-2 text-sm hover:bg-node-component-surface-hovered"
          @mousedown="startRepeat('Comfy.Canvas.ZoomIn')"
          @mouseup="stopRepeat"
          @mouseleave="stopRepeat"
        >
          <span class="font-medium">{{ $t('graphCanvasMenu.zoomIn') }}</span>
          <span class="text-[9px] text-text-primary">{{
            zoomInCommandText
          }}</span>
        </div>

        <div
          class="flex cursor-pointer items-center justify-between rounded px-3 py-2 text-sm hover:bg-node-component-surface-hovered"
          @mousedown="startRepeat('Comfy.Canvas.ZoomOut')"
          @mouseup="stopRepeat"
          @mouseleave="stopRepeat"
        >
          <span class="font-medium">{{ $t('graphCanvasMenu.zoomOut') }}</span>
          <span class="text-[9px] text-text-primary">{{
            zoomOutCommandText
          }}</span>
        </div>

        <div
          class="flex cursor-pointer items-center justify-between rounded px-3 py-2 text-sm hover:bg-node-component-surface-hovered"
          @click="executeCommand('Comfy.Canvas.FitView')"
        >
          <span class="font-medium">{{ $t('zoomControls.zoomToFit') }}</span>
          <span class="text-[9px] text-text-primary">{{
            zoomToFitCommandText
          }}</span>
        </div>

        <div
          ref="zoomInputContainer"
          class="zoomInputContainer flex items-center gap-1 rounded bg-input-surface p-2"
        >
          <InputNumber
            :default-value="canvasStore.appScalePercentage"
            :min="1"
            :max="1000"
            :show-buttons="false"
            :use-grouping="false"
            :unstyled="true"
            input-class="bg-transparent border-none outline-hidden text-sm shadow-none my-0 w-full"
            fluid
            @input="applyZoom"
            @keyup.enter="applyZoom"
          />
          <span class="flex-shrink-0 text-sm text-text-primary">%</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { InputNumberInputEvent } from 'primevue'
import { InputNumber } from 'primevue'
import { computed, nextTick, ref, watch } from 'vue'

import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useMinimap } from '@/renderer/extensions/minimap/composables/useMinimap'
import { useCommandStore } from '@/stores/commandStore'

const minimap = useMinimap()
const commandStore = useCommandStore()
const canvasStore = useCanvasStore()
const { formatKeySequence } = useCommandStore()

interface Props {
  visible: boolean
}

const props = defineProps<Props>()

const interval = ref<number | null>(null)

const applyZoom = (val: InputNumberInputEvent) => {
  const inputValue = val.value as number
  if (isNaN(inputValue) || inputValue < 1 || inputValue > 1000) {
    return
  }
  canvasStore.setAppZoomFromPercentage(inputValue)
}

const executeCommand = (command: string) => {
  void commandStore.execute(command)
}

const startRepeat = (command: string) => {
  if (interval.value) return
  const cmd = () => commandStore.execute(command)
  void cmd()
  interval.value = window.setInterval(cmd, 100)
}

const stopRepeat = () => {
  if (interval.value) {
    clearInterval(interval.value)
    interval.value = null
  }
}
const filteredMinimapStyles = computed(() => {
  return {
    ...minimap.containerStyles.value,
    height: undefined,
    width: undefined
  }
})
const zoomInCommandText = computed(() =>
  formatKeySequence(commandStore.getCommand('Comfy.Canvas.ZoomIn'))
)
const zoomOutCommandText = computed(() =>
  formatKeySequence(commandStore.getCommand('Comfy.Canvas.ZoomOut'))
)
const zoomToFitCommandText = computed(() =>
  formatKeySequence(commandStore.getCommand('Comfy.Canvas.FitView'))
)
const zoomInputContainer = ref<HTMLDivElement | null>(null)

watch(
  () => props.visible,
  async (newVal) => {
    if (newVal) {
      await nextTick()
      const input = zoomInputContainer.value?.querySelector(
        'input'
      ) as HTMLInputElement
      input?.focus()
    }
  }
)
</script>
<style>
.zoomInputContainer:focus-within {
  border: 1px solid var(--color-white);
}
</style>
