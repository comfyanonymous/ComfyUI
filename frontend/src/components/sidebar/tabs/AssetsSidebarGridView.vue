<template>
  <div class="flex h-full flex-col">
    <!-- Assets Header -->
    <div v-if="assets.length" class="px-2 2xl:px-4">
      <div
        class="flex items-center py-2 text-sm font-normal leading-normal text-muted-foreground font-inter"
      >
        {{
          t(
            assetType === 'input'
              ? 'sideToolbar.importedAssetsHeader'
              : 'sideToolbar.generatedAssetsHeader'
          )
        }}
      </div>
    </div>

    <!-- Assets Grid -->
    <VirtualGrid
      class="flex-1"
      :items="assetItems"
      :grid-style="gridStyle"
      @approach-end="emit('approach-end')"
    >
      <template #item="{ item }">
        <MediaAssetCard
          :asset="item.asset"
          :selected="isSelected(item.asset.id)"
          :show-output-count="showOutputCount(item.asset)"
          :output-count="getOutputCount(item.asset)"
          @click="emit('select-asset', item.asset)"
          @context-menu="emit('context-menu', $event, item.asset)"
          @zoom="emit('zoom', item.asset)"
          @output-count-click="emit('output-count-click', item.asset)"
        />
      </template>
    </VirtualGrid>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'

import VirtualGrid from '@/components/common/VirtualGrid.vue'
import MediaAssetCard from '@/platform/assets/components/MediaAssetCard.vue'
import type { AssetItem } from '@/platform/assets/schemas/assetSchema'

const {
  assets,
  isSelected,
  assetType = 'output',
  showOutputCount,
  getOutputCount
} = defineProps<{
  assets: AssetItem[]
  isSelected: (assetId: string) => boolean
  assetType?: 'input' | 'output'
  showOutputCount: (asset: AssetItem) => boolean
  getOutputCount: (asset: AssetItem) => number
}>()

const emit = defineEmits<{
  (e: 'select-asset', asset: AssetItem): void
  (e: 'context-menu', event: MouseEvent, asset: AssetItem): void
  (e: 'approach-end'): void
  (e: 'zoom', asset: AssetItem): void
  (e: 'output-count-click', asset: AssetItem): void
}>()

const { t } = useI18n()

type AssetGridItem = { key: string; asset: AssetItem }

const assetItems = computed<AssetGridItem[]>(() =>
  assets.map((asset) => ({
    key: `asset-${asset.id}`,
    asset
  }))
)

const gridStyle = {
  display: 'grid',
  gridTemplateColumns: 'repeat(auto-fill, minmax(200px, 1fr))',
  padding: '0 0.5rem',
  gap: '0.5rem'
}
</script>
