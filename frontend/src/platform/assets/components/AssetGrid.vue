<template>
  <div
    data-component-id="AssetGrid"
    class="h-full"
    role="grid"
    :aria-label="$t('assetBrowser.assetCollection')"
    :aria-rowcount="-1"
    :aria-colcount="-1"
    :aria-setsize="assets.length"
  >
    <div v-if="loading" class="flex h-full items-center justify-center py-20">
      <i
        class="icon-[lucide--loader] size-12 animate-spin text-muted-foreground"
      />
    </div>
    <div
      v-else-if="assets.length === 0"
      class="flex h-full select-none flex-col items-center justify-center py-16 text-muted-foreground"
    >
      <i class="mb-4 icon-[lucide--search] size-10" />
      <h3 class="mb-2 text-lg font-medium">
        {{ emptyTitle ?? $t('assetBrowser.noAssetsFound') }}
      </h3>
      <p class="text-sm whitespace-pre-wrap text-center">
        {{ emptyMessage ?? $t('assetBrowser.tryAdjustingFilters') }}
      </p>
    </div>
    <VirtualGrid
      v-else
      :items="assetsWithKey"
      :grid-style
      :default-item-height="320"
      :default-item-width="240"
      :max-columns
    >
      <template #item="{ item }">
        <AssetCard
          :asset="item"
          :interactive="true"
          :focused="item.id === focusedAssetId"
          @focus="$emit('assetFocus', $event)"
          @select="$emit('assetSelect', $event)"
          @deleted="$emit('assetDeleted', $event)"
          @show-info="$emit('assetShowInfo', $event)"
        />
      </template>
    </VirtualGrid>
  </div>
</template>

<script setup lang="ts">
import { breakpointsTailwind, useBreakpoints } from '@vueuse/core'
import type { CSSProperties } from 'vue'
import { computed } from 'vue'

import VirtualGrid from '@/components/common/VirtualGrid.vue'
import AssetCard from '@/platform/assets/components/AssetCard.vue'
import type { AssetDisplayItem } from '@/platform/assets/composables/useAssetBrowser'

const { assets, focusedAssetId, emptyTitle, emptyMessage } = defineProps<{
  assets: AssetDisplayItem[]
  loading?: boolean
  focusedAssetId?: string | null
  emptyTitle?: string
  emptyMessage?: string
}>()

defineEmits<{
  assetFocus: [asset: AssetDisplayItem]
  assetSelect: [asset: AssetDisplayItem]
  assetDeleted: [asset: AssetDisplayItem]
  assetShowInfo: [asset: AssetDisplayItem]
}>()

const assetsWithKey = computed(() =>
  assets.map((asset) => ({ ...asset, key: asset.id }))
)

const breakpoints = useBreakpoints(breakpointsTailwind)
const is2Xl = breakpoints.greaterOrEqual('2xl')
const isXl = breakpoints.greaterOrEqual('xl')
const isLg = breakpoints.greaterOrEqual('lg')
const isMd = breakpoints.greaterOrEqual('md')
const maxColumns = computed(() => {
  if (is2Xl.value) return 5
  if (isXl.value) return 3
  if (isLg.value) return 3
  if (isMd.value) return 2
  return 1
})

const gridStyle: CSSProperties = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fill, minmax(15rem, 1fr))',
  gap: '1rem',
  padding: '0.5rem'
}
</script>
