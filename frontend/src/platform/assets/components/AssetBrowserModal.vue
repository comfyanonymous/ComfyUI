<template>
  <BaseModalLayout
    v-model:right-panel-open="isRightPanelOpen"
    data-component-id="AssetBrowserModal"
    class="size-full max-h-full max-w-full min-w-0"
    :content-title="displayTitle"
    :right-panel-title="$t('assetBrowser.modelInfo.title')"
    @close="handleClose"
  >
    <template v-if="shouldShowLeftPanel" #leftPanelHeaderTitle>
      <i class="icon-[comfy--ai-model] size-4" />
      <h2 class="flex-auto select-none text-base font-semibold text-nowrap">
        {{ displayTitle }}
      </h2>
    </template>
    <template v-if="shouldShowLeftPanel" #leftPanel>
      <LeftSidePanel
        v-model="selectedNavItem"
        data-component-id="AssetBrowserModal-LeftSidePanel"
        :nav-items
      />
    </template>

    <template #header>
      <div
        class="flex w-full items-center justify-between gap-2"
        @click.self="focusedAsset = null"
      >
        <SearchBox
          v-model="searchQuery"
          :autofocus="true"
          size="lg"
          :placeholder="$t('g.searchPlaceholder', { subject: '' })"
          class="max-w-96"
        />
        <Button
          v-if="isUploadButtonEnabled"
          variant="primary"
          :size="breakpoints.md ? 'lg' : 'icon'"
          data-attr="upload-model-button"
          @click="showUploadDialog"
        >
          <i class="icon-[lucide--folder-input]" />
          <span class="hidden md:inline">{{
            $t('assetBrowser.uploadModel')
          }}</span>
        </Button>
      </div>
    </template>

    <template #contentFilter>
      <AssetFilterBar
        :assets="categoryFilteredAssets"
        :show-ownership-filter
        @filter-change="updateFilters"
        @click.self="focusedAsset = null"
      />
    </template>

    <template #content>
      <AssetGrid
        :assets="filteredAssets"
        :loading="isLoading"
        :focused-asset-id="focusedAsset?.id"
        :empty-message
        @asset-focus="handleAssetFocus"
        @asset-select="handleAssetSelectAndEmit"
        @asset-deleted="refreshAssets"
        @asset-show-info="handleShowInfo"
        @click="focusedAsset = null"
      />
    </template>

    <template #rightPanel>
      <ModelInfoPanel v-if="focusedAsset" :asset="focusedAsset" :cache-key />
      <div
        v-else
        class="flex h-full items-center justify-center break-words p-6 text-center text-muted"
      >
        {{ $t('assetBrowser.modelInfo.selectModelPrompt') }}
      </div>
    </template>
  </BaseModalLayout>
</template>

<script setup lang="ts">
import { breakpointsTailwind, useBreakpoints } from '@vueuse/core'
import { computed, provide, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import SearchBox from '@/components/common/SearchBox.vue'
import Button from '@/components/ui/button/Button.vue'
import BaseModalLayout from '@/components/widget/layout/BaseModalLayout.vue'
import LeftSidePanel from '@/components/widget/panel/LeftSidePanel.vue'
import AssetFilterBar from '@/platform/assets/components/AssetFilterBar.vue'
import AssetGrid from '@/platform/assets/components/AssetGrid.vue'
import ModelInfoPanel from '@/platform/assets/components/modelInfo/ModelInfoPanel.vue'
import type { AssetDisplayItem } from '@/platform/assets/composables/useAssetBrowser'
import { useAssetBrowser } from '@/platform/assets/composables/useAssetBrowser'
import { useModelTypes } from '@/platform/assets/composables/useModelTypes'
import { useModelUpload } from '@/platform/assets/composables/useModelUpload'
import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import { formatCategoryLabel } from '@/platform/assets/utils/categoryLabel'
import { useAssetsStore } from '@/stores/assetsStore'
import { useModelToNodeStore } from '@/stores/modelToNodeStore'
import { OnCloseKey } from '@/types/widgetTypes'

const { t } = useI18n()
const assetStore = useAssetsStore()
const modelToNodeStore = useModelToNodeStore()
const breakpoints = useBreakpoints(breakpointsTailwind)

const props = defineProps<{
  nodeType?: string
  assetType?: string
  onSelect?: (asset: AssetItem) => void
  onClose?: () => void
  showLeftPanel?: boolean
  title?: string
}>()

const emit = defineEmits<{
  'asset-select': [asset: AssetDisplayItem]
  close: []
}>()

provide(OnCloseKey, props.onClose ?? (() => {}))

const cacheKey = computed(() => {
  if (props.nodeType) return props.nodeType
  if (props.assetType) return `tag:${props.assetType}`
  return ''
})

const fetchedAssets = computed(() => assetStore.getAssets(cacheKey.value))

const isStoreLoading = computed(() => assetStore.isModelLoading(cacheKey.value))

const isLoading = computed(
  () => isStoreLoading.value && fetchedAssets.value.length === 0
)

async function refreshAssets(): Promise<void> {
  if (props.nodeType) {
    await assetStore.updateModelsForNodeType(props.nodeType)
  } else if (props.assetType) {
    await assetStore.updateModelsForTag(props.assetType)
  }
}

void refreshAssets()

const { fetchModelTypes } = useModelTypes()
void fetchModelTypes()

const { isUploadButtonEnabled, showUploadDialog } =
  useModelUpload(refreshAssets)

const {
  searchQuery,
  selectedNavItem,
  selectedCategory,
  navItems,
  categoryFilteredAssets,
  filteredAssets,
  isImportedSelected,
  updateFilters
} = useAssetBrowser(fetchedAssets)

const focusedAsset = ref<AssetDisplayItem | null>(null)
const isRightPanelOpen = ref(false)

const primaryCategoryTag = computed(() => {
  const assets = fetchedAssets.value ?? []
  const tagFromAssets = assets
    .map((asset) => asset.tags?.find((tag) => tag !== 'models'))
    .find((tag): tag is string => typeof tag === 'string' && tag.length > 0)

  if (tagFromAssets) return tagFromAssets

  if (props.nodeType) {
    const mapped = modelToNodeStore.getCategoryForNodeType(props.nodeType)
    if (mapped) return mapped
  }

  if (props.assetType) return props.assetType

  return 'models'
})

const activeCategoryTag = computed(() => {
  if (selectedCategory.value !== 'all') {
    return selectedCategory.value
  }
  return primaryCategoryTag.value
})

const displayTitle = computed(() => {
  if (props.title) return props.title

  const label = formatCategoryLabel(activeCategoryTag.value)
  return t('assetBrowser.allCategory', { category: label })
})

const shouldShowLeftPanel = computed(() => {
  return props.showLeftPanel ?? true
})

const showOwnershipFilter = computed(
  () =>
    !shouldShowLeftPanel.value ||
    (selectedNavItem.value !== 'all' && selectedNavItem.value !== 'imported')
)

const emptyMessage = computed(() => {
  if (!isImportedSelected.value) {
    return isUploadButtonEnabled.value
      ? t('assetBrowser.noResultsCanImport')
      : undefined
  }

  return isUploadButtonEnabled.value
    ? t('assetBrowser.emptyImported.canImport')
    : t('assetBrowser.emptyImported.restricted')
})

function handleClose() {
  props.onClose?.()
  emit('close')
}

function handleAssetFocus(asset: AssetDisplayItem) {
  focusedAsset.value = asset
}

function handleShowInfo(asset: AssetDisplayItem) {
  focusedAsset.value = asset
  isRightPanelOpen.value = true
}

function handleAssetSelectAndEmit(asset: AssetDisplayItem) {
  emit('asset-select', asset)
  props.onSelect?.(asset)
}
</script>
