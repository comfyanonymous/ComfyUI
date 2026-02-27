<template>
  <div
    class="option-container flex w-full cursor-pointer items-center justify-between overflow-hidden px-2 py-0"
  >
    <div class="option-display-name flex flex-col font-semibold">
      <div>
        <span v-if="isBookmarked">
          <i class="pi pi-bookmark-fill mr-1 text-sm" />
        </span>
        <span v-html="highlightQuery(nodeDef.display_name, currentQuery)" />
        <span>&nbsp;</span>
        <Tag v-if="showIdName" severity="secondary">
          <span v-html="highlightQuery(nodeDef.name, currentQuery)" />
        </Tag>
      </div>
      <div
        v-if="showCategory"
        class="option-category truncate text-sm font-light text-muted"
      >
        {{ nodeDef.category.replaceAll('/', ' > ') }}
      </div>
    </div>
    <div class="option-badges">
      <Tag
        v-if="nodeDef.deprecated"
        :value="$t('g.deprecated')"
        severity="danger"
      />
      <Tag
        v-if="nodeDef.experimental"
        :value="$t('g.experimental')"
        severity="primary"
      />
      <Tag v-if="nodeDef.dev_only" :value="$t('g.devOnly')" severity="info" />
      <Tag
        v-if="showNodeFrequency && nodeFrequency > 0"
        :value="formatNumberWithSuffix(nodeFrequency, { roundToInt: true })"
        severity="secondary"
      />
      <Chip
        v-if="nodeDef.nodeSource.type !== NodeSourceType.Unknown"
        class="text-sm font-light"
      >
        {{ nodeDef.nodeSource.displayText }}
      </Chip>
    </div>
  </div>
</template>

<script setup lang="ts">
import Chip from 'primevue/chip'
import Tag from 'primevue/tag'
import { computed } from 'vue'

import { useSettingStore } from '@/platform/settings/settingStore'
import { useNodeBookmarkStore } from '@/stores/nodeBookmarkStore'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'
import { useNodeFrequencyStore } from '@/stores/nodeDefStore'
import { NodeSourceType } from '@/types/nodeSource'
import { formatNumberWithSuffix, highlightQuery } from '@/utils/formatUtil'

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
  nodeFrequencyStore.getNodeFrequency(props.nodeDef)
)

const nodeBookmarkStore = useNodeBookmarkStore()
const isBookmarked = computed(() =>
  nodeBookmarkStore.isBookmarked(props.nodeDef)
)

const props = defineProps<{
  nodeDef: ComfyNodeDefImpl
  currentQuery: string
}>()
</script>

<style scoped>
:deep(.highlight) {
  background-color: var(--p-primary-color);
  color: var(--p-primary-contrast-color);
  font-weight: 700;
  border-radius: 0.25rem;
  padding: 0 0.125rem;
  margin: -0.125rem 0.125rem;
}
</style>
