<template>
  <div
    class="option-container flex w-full cursor-pointer items-center justify-between overflow-hidden"
  >
    <div class="flex flex-col gap-0.5 overflow-hidden">
      <div class="font-semibold text-foreground flex items-center gap-2">
        <span v-if="isBookmarked && !hideBookmarkIcon">
          <i class="pi pi-bookmark-fill mr-1 text-sm" />
        </span>
        <span v-html="highlightQuery(nodeDef.display_name, currentQuery)" />
        <span v-if="showIdName">&nbsp;</span>
        <span
          v-if="showIdName"
          class="rounded bg-secondary-background px-1.5 py-0.5 text-xs text-muted-foreground"
          v-html="highlightQuery(nodeDef.name, currentQuery)"
        />

        <NodePricingBadge :node-def="nodeDef" />
        <NodeProviderBadge v-if="nodeDef.api_node" :node-def="nodeDef" />
      </div>
      <div
        v-if="showDescription"
        class="flex items-center gap-1 text-[11px] text-muted-foreground"
      >
        <span
          v-if="
            showSourceBadge &&
            nodeDef.nodeSource.type !== NodeSourceType.Core &&
            nodeDef.nodeSource.type !== NodeSourceType.Unknown
          "
          class="inline-flex shrink-0 rounded border border-border px-1.5 py-0.5 text-xs bg-base-foreground/5 text-base-foreground/70 mr-0.5"
        >
          {{ nodeDef.nodeSource.displayText }}
        </span>
        <TextTicker v-if="nodeDef.description">
          {{ nodeDef.description }}
        </TextTicker>
      </div>
      <div
        v-else-if="showCategory"
        class="option-category truncate text-sm font-light text-muted"
      >
        {{ nodeDef.category.replaceAll('/', ' > ') }}
      </div>
    </div>
    <div v-if="!showDescription" class="flex items-center gap-1">
      <span
        v-if="nodeDef.deprecated"
        class="rounded bg-red-500/20 px-1.5 py-0.5 text-xs text-red-400"
      >
        {{ $t('g.deprecated') }}
      </span>
      <span
        v-if="nodeDef.experimental"
        class="rounded bg-blue-500/20 px-1.5 py-0.5 text-xs text-blue-400"
      >
        {{ $t('g.experimental') }}
      </span>
      <span
        v-if="nodeDef.dev_only"
        class="rounded bg-cyan-500/20 px-1.5 py-0.5 text-xs text-cyan-400"
      >
        {{ $t('g.devOnly') }}
      </span>
      <span
        v-if="showNodeFrequency && nodeFrequency > 0"
        class="rounded bg-secondary-background px-1.5 py-0.5 text-xs text-muted-foreground"
      >
        {{ formatNumberWithSuffix(nodeFrequency, { roundToInt: true }) }}
      </span>
      <span
        v-if="nodeDef.nodeSource.type !== NodeSourceType.Unknown"
        class="rounded bg-secondary-background px-2 py-0.5 text-sm text-muted-foreground"
      >
        {{ nodeDef.nodeSource.displayText }}
      </span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import TextTicker from '@/components/common/TextTicker.vue'
import NodePricingBadge from '@/components/node/NodePricingBadge.vue'
import NodeProviderBadge from '@/components/node/NodeProviderBadge.vue'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useNodeBookmarkStore } from '@/stores/nodeBookmarkStore'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import { useNodeFrequencyStore } from '@/stores/nodeDefStore'
import { NodeSourceType } from '@/types/nodeSource'
import { formatNumberWithSuffix, highlightQuery } from '@/utils/formatUtil'

const {
  nodeDef,
  currentQuery,
  showDescription = false,
  showSourceBadge = false,
  hideBookmarkIcon = false
} = defineProps<{
  nodeDef: ComfyNodeDefImpl
  currentQuery: string
  showDescription?: boolean
  showSourceBadge?: boolean
  hideBookmarkIcon?: boolean
}>()

const settingStore = useSettingStore()
const showCategory = computed(() =>
  settingStore.get('Comfy.NodeSearchBoxImpl.ShowCategory')
)
const showIdName = computed(() =>
  settingStore.get('Comfy.NodeSearchBoxImpl.ShowIdName')
)
const showNodeFrequency = computed(() =>
  settingStore.get('Comfy.NodeSearchBoxImpl.ShowNodeFrequency')
)
const nodeFrequencyStore = useNodeFrequencyStore()
const nodeFrequency = computed(() =>
  nodeFrequencyStore.getNodeFrequency(nodeDef)
)

const nodeBookmarkStore = useNodeBookmarkStore()
const isBookmarked = computed(() => nodeBookmarkStore.isBookmarked(nodeDef))
</script>

<style scoped>
:deep(.highlight) {
  background-color: color-mix(in srgb, currentColor 20%, transparent);
  font-weight: 700;
  border-radius: 0.25rem;
  padding: 0 0.125rem;
  margin: -0.125rem 0.125rem;
}
</style>
