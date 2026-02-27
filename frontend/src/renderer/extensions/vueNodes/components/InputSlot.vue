<template>
  <div v-if="renderError" class="node-error p-1 text-xs text-red-500">⚠️</div>
  <div
    v-else
    v-tooltip.left="tooltipConfig"
    :class="
      cn(
        'lg-slot lg-slot--input flex items-center group rounded-r-lg m-0',
        'cursor-crosshair',
        dotOnly ? 'lg-slot--dot-only' : 'pr-6',
        {
          'lg-slot--connected': props.connected,
          'lg-slot--compatible': props.compatible,
          'opacity-40': shouldDim
        },
        props.socketless && 'pointer-events-none invisible'
      )
    "
  >
    <!-- Connection Dot -->
    <SlotConnectionDot
      ref="connectionDotRef"
      :class="
        cn(
          '-translate-x-1/2 w-3',
          hasError &&
            'before:ring-2 before:ring-error before:ring-offset-0 before:size-4 before:absolute before:rounded-full before:pointer-events-none'
        )
      "
      :slot-data
      @click="onClick"
      @dblclick="onDoubleClick"
      @pointerdown="onPointerDown"
    />

    <!-- Slot Name -->
    <div class="h-full flex items-center min-w-0">
      <span
        v-if="!props.dotOnly && !hasNoLabel"
        :class="
          cn(
            'truncate text-node-component-slot-text',
            hasError && 'text-error font-medium'
          )
        "
      >
        {{
          slotData.label ||
          slotData.localized_name ||
          (slotData.name ?? `Input ${index}`)
        }}
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onErrorCaptured, ref, watchEffect } from 'vue'
import type { ComponentPublicInstance } from 'vue'

import { useErrorHandling } from '@/composables/useErrorHandling'
import type { INodeSlot } from '@/lib/litegraph/src/litegraph'
import { useSlotLinkDragUIState } from '@/renderer/core/canvas/links/slotLinkDragUIState'
import { getSlotKey } from '@/renderer/core/layout/slots/slotIdentifier'
import { useNodeTooltips } from '@/renderer/extensions/vueNodes/composables/useNodeTooltips'
import { useSlotElementTracking } from '@/renderer/extensions/vueNodes/composables/useSlotElementTracking'
import { useSlotLinkInteraction } from '@/renderer/extensions/vueNodes/composables/useSlotLinkInteraction'
import { cn } from '@/utils/tailwindUtil'

import SlotConnectionDot from './SlotConnectionDot.vue'

interface InputSlotProps {
  slotData: INodeSlot
  compatible?: boolean
  connected?: boolean
  dotOnly?: boolean
  hasError?: boolean
  index: number
  nodeType?: string
  nodeId?: string
  socketless?: boolean
}

const props = defineProps<InputSlotProps>()

const hasNoLabel = computed(
  () =>
    !props.slotData.label &&
    !props.slotData.localized_name &&
    props.slotData.name === ''
)
const dotOnly = computed(() => props.dotOnly || hasNoLabel.value)

const renderError = ref<string | null>(null)
const { toastErrorHandler } = useErrorHandling()

const { getInputSlotTooltip, createTooltipConfig } = useNodeTooltips(
  props.nodeType || ''
)

const tooltipConfig = computed(() => {
  const slotName = props.slotData.localized_name || props.slotData.name || ''
  const tooltipText = getInputSlotTooltip(slotName)
  const fallbackText = tooltipText || `Input: ${slotName}`
  return createTooltipConfig(fallbackText)
})

onErrorCaptured((error) => {
  renderError.value = error.message
  toastErrorHandler(error)
  return false
})

const { state: dragState } = useSlotLinkDragUIState()
const slotKey = computed(() =>
  getSlotKey(props.nodeId ?? '', props.index, true)
)
const shouldDim = computed(() => {
  if (!dragState.active) return false
  return !dragState.compatible.get(slotKey.value)
})

const connectionDotRef = ref<ComponentPublicInstance<{
  slotElRef: HTMLElement | undefined
}> | null>(null)
const slotElRef = ref<HTMLElement | null>(null)

watchEffect(() => {
  const el = connectionDotRef.value?.slotElRef
  slotElRef.value = el || null
})

useSlotElementTracking({
  nodeId: props.nodeId ?? '',
  index: props.index,
  type: 'input',
  element: slotElRef
})

const { onClick, onDoubleClick, onPointerDown } = useSlotLinkInteraction({
  nodeId: props.nodeId ?? '',
  index: props.index,
  type: 'input'
})
</script>
