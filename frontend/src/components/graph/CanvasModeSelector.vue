<template>
  <Button
    ref="buttonRef"
    variant="secondary"
    class="group h-8 rounded-none! bg-comfy-menu-bg p-0 transition-none! hover:rounded-lg! hover:bg-interface-button-hover-surface!"
    :style="buttonStyles"
    @click="toggle"
  >
    <div class="flex items-center gap-1 pr-0.5">
      <div
        class="rounded-lg bg-interface-panel-selected-surface p-2 group-hover:bg-interface-button-hover-surface"
      >
        <i :class="currentModeIcon" class="block h-4 w-4" />
      </div>
      <i class="icon-[lucide--chevron-down] block h-4 w-4 pr-1.5" />
    </div>
  </Button>

  <Popover
    ref="popover"
    :auto-z-index="true"
    :base-z-index="1000"
    :dismissable="true"
    :close-on-escape="true"
    unstyled
    :pt="popoverPt"
  >
    <div class="flex flex-col gap-1">
      <div
        class="flex cursor-pointer items-center justify-between px-3 py-2 text-sm hover:bg-node-component-surface-hovered"
        @click="setMode('select')"
      >
        <div class="flex items-center gap-2">
          <i class="icon-[lucide--mouse-pointer-2] h-4 w-4" />
          <span>{{ $t('graphCanvasMenu.select') }}</span>
        </div>
        <span class="text-[9px] text-text-primary">{{
          unlockCommandText
        }}</span>
      </div>

      <div
        class="flex cursor-pointer items-center justify-between rounded px-3 py-2 text-sm hover:bg-node-component-surface-hovered"
        @click="setMode('hand')"
      >
        <div class="flex items-center gap-2">
          <i class="icon-[lucide--hand] h-4 w-4" />
          <span>{{ $t('graphCanvasMenu.hand') }}</span>
        </div>
        <span class="text-[9px] text-text-primary">{{ lockCommandText }}</span>
      </div>
    </div>
  </Popover>
</template>

<script setup lang="ts">
import Popover from 'primevue/popover'
import type { ComponentPublicInstance } from 'vue'
import { computed, ref } from 'vue'

import Button from '@/components/ui/button/Button.vue'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { useCommandStore } from '@/stores/commandStore'

interface Props {
  buttonStyles?: Record<string, string>
}

defineProps<Props>()
const buttonRef = ref<ComponentPublicInstance | null>(null)
const popover = ref<InstanceType<typeof Popover>>()
const commandStore = useCommandStore()
const canvasStore = useCanvasStore()

const isCanvasReadOnly = computed(() => canvasStore.canvas?.read_only ?? false)

const currentModeIcon = computed(() =>
  isCanvasReadOnly.value
    ? 'icon-[lucide--hand]'
    : 'icon-[lucide--mouse-pointer-2]'
)

const unlockCommandText = computed(() =>
  commandStore
    .formatKeySequence(commandStore.getCommand('Comfy.Canvas.Unlock'))
    .toUpperCase()
)

const lockCommandText = computed(() =>
  commandStore
    .formatKeySequence(commandStore.getCommand('Comfy.Canvas.Lock'))
    .toUpperCase()
)

const toggle = (event: Event) => {
  const el = buttonRef.value?.$el || buttonRef.value
  popover.value?.toggle(event, el)
}

const setMode = (mode: 'select' | 'hand') => {
  if (mode === 'select' && isCanvasReadOnly.value) {
    void commandStore.execute('Comfy.Canvas.Unlock')
  } else if (mode === 'hand' && !isCanvasReadOnly.value) {
    void commandStore.execute('Comfy.Canvas.Lock')
  }
  popover.value?.hide()
}

const popoverPt = computed(() => ({
  root: {
    class: 'absolute z-50 -translate-y-2'
  },
  content: {
    class: [
      'mb-2 text-text-primary',
      'shadow-lg border border-interface-stroke',
      'bg-nav-background',
      'rounded-lg',
      'p-2 px-3',
      'min-w-39',
      'select-none'
    ]
  }
}))
</script>
