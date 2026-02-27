<template>
  <div
    class="flex w-50 flex-col overflow-hidden rounded-2xl bg-(--base-background) border border-(--border-default)"
  >
    <div ref="previewContainerRef" class="overflow-hidden p-3">
      <div ref="previewWrapperRef" class="origin-top-left scale-50">
        <LGraphNodePreview :node-def="nodeDef" position="relative" />
      </div>
    </div>

    <!-- Content Section -->
    <div class="flex flex-col gap-2 p-3 pt-1">
      <!-- Title -->
      <h3 class="text-xs font-semibold text-foreground m-0">
        {{ nodeDef.display_name }}
      </h3>

      <!-- Category Path -->
      <p
        v-if="showCategoryPath && nodeDef.category"
        class="text-xs text-muted-foreground -mt-1"
      >
        {{ nodeDef.category.replaceAll('/', ' > ') }}
      </p>

      <!-- Badges -->
      <div class="flex flex-wrap gap-2 empty:hidden">
        <NodePricingBadge :node-def="nodeDef" />
        <NodeProviderBadge :node-def="nodeDef" />
      </div>

      <!-- Description -->
      <p
        v-if="nodeDef.description"
        class="text-[11px] font-normal leading-normal text-muted-foreground m-0"
      >
        {{ nodeDef.description }}
      </p>

      <!-- Divider -->
      <div
        v-if="(inputs.length > 0 || outputs.length > 0) && showInputsAndOutputs"
        class="border-t border-border-default"
      />

      <!-- Inputs Section -->
      <div
        v-if="inputs.length > 0 && showInputsAndOutputs"
        class="flex flex-col gap-1"
      >
        <h4
          class="text-xxs font-semibold uppercase tracking-wide text-muted-foreground m-0"
        >
          {{ $t('nodeHelpPage.inputs') }}
        </h4>
        <div
          v-for="input in inputs"
          :key="input.name"
          class="flex items-center justify-between gap-2 text-xxs"
        >
          <span class="shrink-0 text-foreground">{{ input.name }}</span>
          <span class="min-w-0 truncate text-muted-foreground">{{
            input.type
          }}</span>
        </div>
      </div>

      <!-- Outputs Section -->
      <div
        v-if="outputs.length > 0 && showInputsAndOutputs"
        class="flex flex-col gap-1"
      >
        <h4
          class="text-xxs font-semibold uppercase tracking-wide text-muted-foreground m-0"
        >
          {{ $t('nodeHelpPage.outputs') }}
        </h4>
        <div
          v-for="output in outputs"
          :key="output.name"
          class="flex items-center justify-between gap-2 text-xxs"
        >
          <span class="shrink-0 text-foreground">{{ output.name }}</span>
          <span class="min-w-0 truncate text-muted-foreground">{{
            output.type
          }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { useResizeObserver } from '@vueuse/core'
import { computed, ref } from 'vue'

import NodePricingBadge from '@/components/node/NodePricingBadge.vue'
import NodeProviderBadge from '@/components/node/NodeProviderBadge.vue'
import LGraphNodePreview from '@/renderer/extensions/vueNodes/components/LGraphNodePreview.vue'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'

const SCALE_FACTOR = 0.5
const PREVIEW_CONTAINER_PADDING_PX = 24 // p-3 top + bottom (12px × 2)

const {
  nodeDef,
  showInputsAndOutputs = true,
  showCategoryPath = false
} = defineProps<{
  nodeDef: ComfyNodeDefImpl
  showInputsAndOutputs?: boolean
  showCategoryPath?: boolean
}>()

const previewContainerRef = ref<HTMLElement>()
const previewWrapperRef = ref<HTMLElement>()

useResizeObserver(previewWrapperRef, (entries) => {
  const entry = entries[0]
  if (entry && previewContainerRef.value) {
    const scaledHeight = entry.contentRect.height * SCALE_FACTOR
    previewContainerRef.value.style.height = `${scaledHeight + PREVIEW_CONTAINER_PADDING_PX}px`
  }
})

const inputs = computed(() => {
  if (!nodeDef.inputs) return []
  return Object.entries(nodeDef.inputs)
    .filter(([_, input]) => !input.hidden)
    .map(([name, input]) => ({
      name,
      type: input.type
    }))
})

const outputs = computed(() => {
  if (!nodeDef.outputs) return []
  return nodeDef.outputs.map((output) => ({
    name: output.name,
    type: output.type
  }))
})
</script>
