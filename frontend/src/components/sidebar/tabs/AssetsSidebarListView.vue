<template>
  <div class="flex h-full flex-col">
    <div v-if="assetItems.length" class="px-2">
      <div
        class="flex items-center p-2 text-sm font-normal leading-normal text-muted-foreground font-inter"
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

    <VirtualGrid
      class="flex-1"
      :items="assetItems"
      :grid-style="listGridStyle"
      @approach-end="emit('approach-end')"
    >
      <template #item="{ item }">
        <div class="relative">
          <LoadingOverlay
            :loading="assetsStore.isAssetDeleting(item.asset.id)"
            size="sm"
          >
            <i class="pi pi-trash text-xs" />
          </LoadingOverlay>
          <AssetsListItem
            role="button"
            tabindex="0"
            :aria-label="
              t('assetBrowser.ariaLabel.assetCard', {
                name: item.asset.name,
                type: getAssetMediaType(item.asset)
              })
            "
            :class="
              cn(
                getAssetCardClass(isSelected(item.asset.id)),
                item.isChild && 'pl-6'
              )
            "
            :preview-url="getAssetPreviewUrl(item.asset)"
            :preview-alt="item.asset.name"
            :icon-name="iconForMediaType(getAssetMediaType(item.asset))"
            :is-video-preview="isVideoAsset(item.asset)"
            :primary-text="getAssetPrimaryText(item.asset)"
            :secondary-text="getAssetSecondaryText(item.asset)"
            :stack-count="getStackCount(item.asset)"
            :stack-indicator-label="t('mediaAsset.actions.seeMoreOutputs')"
            :stack-expanded="isStackExpanded(item.asset)"
            @mouseenter="onAssetEnter(item.asset.id)"
            @mouseleave="onAssetLeave(item.asset.id)"
            @contextmenu.prevent.stop="emit('context-menu', $event, item.asset)"
            @click.stop="emit('select-asset', item.asset, selectableAssets)"
            @dblclick.stop="emit('preview-asset', item.asset)"
            @preview-click="emit('preview-asset', item.asset)"
            @stack-toggle="void toggleStack(item.asset)"
          >
            <template v-if="hoveredAssetId === item.asset.id" #actions>
              <Button
                variant="secondary"
                size="icon"
                :aria-label="t('mediaAsset.actions.moreOptions')"
                @click.stop="emit('context-menu', $event, item.asset)"
              >
                <i class="icon-[lucide--ellipsis] size-4" />
              </Button>
            </template>
          </AssetsListItem>
        </div>
      </template>
    </VirtualGrid>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'

import LoadingOverlay from '@/components/common/LoadingOverlay.vue'
import VirtualGrid from '@/components/common/VirtualGrid.vue'
import Button from '@/components/ui/button/Button.vue'
import AssetsListItem from '@/platform/assets/components/AssetsListItem.vue'
import type { OutputStackListItem } from '@/platform/assets/composables/useOutputStacks'
import { getOutputAssetMetadata } from '@/platform/assets/schemas/assetMetadataSchema'
import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import { iconForMediaType } from '@/platform/assets/utils/mediaIconUtil'
import { useAssetsStore } from '@/stores/assetsStore'
import {
  formatDuration,
  formatSize,
  getMediaTypeFromFilename,
  truncateFilename
} from '@/utils/formatUtil'
import { cn } from '@/utils/tailwindUtil'

const {
  assetItems,
  selectableAssets,
  isSelected,
  isStackExpanded,
  toggleStack,
  assetType = 'output'
} = defineProps<{
  assetItems: OutputStackListItem[]
  selectableAssets: AssetItem[]
  isSelected: (assetId: string) => boolean
  isStackExpanded: (asset: AssetItem) => boolean
  toggleStack: (asset: AssetItem) => Promise<void>
  assetType?: 'input' | 'output'
}>()

const assetsStore = useAssetsStore()

const emit = defineEmits<{
  (e: 'select-asset', asset: AssetItem, assets?: AssetItem[]): void
  (e: 'preview-asset', asset: AssetItem): void
  (e: 'context-menu', event: MouseEvent, asset: AssetItem): void
  (e: 'approach-end'): void
}>()

const { t } = useI18n()
const hoveredAssetId = ref<string | null>(null)

const listGridStyle = {
  display: 'grid',
  gridTemplateColumns: 'minmax(0, 1fr)',
  padding: '0 0.5rem',
  gap: '0.5rem'
}

function getAssetPrimaryText(asset: AssetItem): string {
  return truncateFilename(asset.name)
}

function getAssetMediaType(asset: AssetItem) {
  return getMediaTypeFromFilename(asset.name)
}

function isVideoAsset(asset: AssetItem): boolean {
  return getAssetMediaType(asset) === 'video'
}

function getAssetPreviewUrl(asset: AssetItem): string {
  const mediaType = getAssetMediaType(asset)
  if (mediaType === 'image' || mediaType === 'video') {
    return asset.preview_url || ''
  }
  return ''
}

function getAssetSecondaryText(asset: AssetItem): string {
  const metadata = getOutputAssetMetadata(asset.user_metadata)
  if (typeof metadata?.executionTimeInSeconds === 'number') {
    return `${metadata.executionTimeInSeconds.toFixed(2)}s`
  }

  const duration = asset.user_metadata?.duration
  if (typeof duration === 'number') {
    return formatDuration(duration)
  }

  if (typeof asset.size === 'number') {
    return formatSize(asset.size)
  }

  return ''
}

function getStackCount(asset: AssetItem): number | undefined {
  const metadata = getOutputAssetMetadata(asset.user_metadata)
  if (typeof metadata?.outputCount === 'number') {
    return metadata.outputCount
  }

  if (Array.isArray(metadata?.allOutputs)) {
    return metadata.allOutputs.length
  }

  return undefined
}

function getAssetCardClass(selected: boolean): string {
  return cn(
    'w-full text-text-primary transition-colors hover:bg-secondary-background-hover',
    'cursor-pointer',
    selected &&
      'bg-secondary-background-hover ring-1 ring-inset ring-modal-card-border-highlighted'
  )
}

function onAssetEnter(assetId: string) {
  hoveredAssetId.value = assetId
}

function onAssetLeave(assetId: string) {
  if (hoveredAssetId.value === assetId) {
    hoveredAssetId.value = null
  }
}
</script>
