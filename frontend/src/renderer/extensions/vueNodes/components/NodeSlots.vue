<template>
  <div v-if="renderError" class="node-error p-2 text-sm text-red-500">
    {{ st('nodeErrors.slots', 'Node Slots Error') }}
  </div>
  <div v-else :class="cn('flex justify-between min-w-0', unifiedWrapperClass)">
    <div
      v-if="filteredInputs.length"
      :class="cn('flex flex-col min-w-0', unifiedDotsClass)"
    >
      <InputSlot
        v-for="(input, index) in filteredInputs"
        :key="`input-${input.name}`"
        :slot-data="input"
        :node-type="nodeData?.type || ''"
        :node-id="nodeData?.id != null ? String(nodeData.id) : ''"
        :index="getActualInputIndex(input, index)"
      />
    </div>

    <div
      v-if="nodeData?.outputs?.length"
      :class="cn('ml-auto flex flex-col min-w-0', unifiedDotsClass)"
    >
      <OutputSlot
        v-for="(output, index) in nodeData.outputs"
        :key="`output-${output.name}`"
        :slot-data="output"
        :node-type="nodeData?.type || ''"
        :node-id="nodeData?.id != null ? String(nodeData.id) : ''"
        :index="index"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onErrorCaptured, ref } from 'vue'

import type { VueNodeData } from '@/composables/graph/useGraphNodeManager'
import { useErrorHandling } from '@/composables/useErrorHandling'
import { st } from '@/i18n'
import type { INodeSlot } from '@/lib/litegraph/src/litegraph'
import {
  linkedWidgetedInputs,
  nonWidgetedInputs
} from '@/renderer/extensions/vueNodes/utils/nodeDataUtils'
import { cn } from '@/utils/tailwindUtil'

import InputSlot from './InputSlot.vue'
import OutputSlot from './OutputSlot.vue'

interface NodeSlotsProps {
  nodeData: VueNodeData
  unified?: boolean
}

const { nodeData, unified = false } = defineProps<NodeSlotsProps>()

const linkedWidgetInputs = computed(() =>
  unified ? linkedWidgetedInputs(nodeData) : []
)

const filteredInputs = computed(() => [
  ...nonWidgetedInputs(nodeData),
  ...linkedWidgetInputs.value
])

const unifiedWrapperClass = computed((): string =>
  cn(
    unified &&
      'absolute inset-0 items-center pointer-events-none opacity-0 z-30'
  )
)
const unifiedDotsClass = computed((): string =>
  cn(
    unified &&
      'grid grid-cols-1 grid-rows-1 gap-0 [&>*]:row-span-full [&>*]:col-span-full place-items-center'
  )
)

// Get the actual index of an input slot in the node's inputs array
// (accounting for filtered widget slots)
const getActualInputIndex = (
  input: INodeSlot,
  filteredIndex: number
): number => {
  if (!nodeData?.inputs) return filteredIndex

  // Find the actual index in the unfiltered inputs array
  const actualIndex = nodeData.inputs.findIndex((i) => i === input)
  return actualIndex !== -1 ? actualIndex : filteredIndex
}

// Error boundary implementation
const renderError = ref<string | null>(null)
const { toastErrorHandler } = useErrorHandling()

onErrorCaptured((error) => {
  renderError.value = error.message
  toastErrorHandler(error)
  return false
})
</script>
