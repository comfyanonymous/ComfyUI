<template>
  <SidebarTabTemplate
    :title="isInFolderView ? '' : $t('sideToolbar.mediaAssets.title')"
    v-bind="$attrs"
  >
    <template #alt-title>
      <div
        v-if="isInFolderView"
        class="flex w-full items-center justify-between gap-2"
      >
        <div class="flex items-center gap-2">
          <span class="font-bold">{{ $t('assetBrowser.jobId') }}:</span>
          <span class="text-sm">{{ folderJobId?.substring(0, 8) }}</span>
          <button
            class="m-0 cursor-pointer border-0 bg-transparent p-0 outline-0"
            role="button"
            @click="copyJobId"
          >
            <i class="icon-[lucide--copy] text-sm"></i>
          </button>
        </div>
        <div>
          <span>{{ formattedExecutionTime }}</span>
        </div>
      </div>
    </template>
    <template #tool-buttons>
      <!-- Normal Tab View -->
      <TabList v-if="!isInFolderView" v-model="activeTab">
        <Tab class="font-inter" value="output">{{
          $t('sideToolbar.labels.generated')
        }}</Tab>
        <Tab class="font-inter" value="input">{{
          $t('sideToolbar.labels.imported')
        }}</Tab>
      </TabList>
    </template>
    <template #header>
      <!-- Job Detail View Header -->
      <div v-if="isInFolderView" class="px-2 2xl:px-4">
        <Button variant="secondary" size="lg" @click="exitFolderView">
          <i class="icon-[lucide--arrow-left] size-4" />
          <span>{{ $t('sideToolbar.backToAssets') }}</span>
        </Button>
      </div>

      <!-- Filter Bar -->
      <MediaAssetFilterBar
        v-model:search-query="searchQuery"
        v-model:sort-by="sortBy"
        v-model:view-mode="viewMode"
        v-model:media-type-filters="mediaTypeFilters"
        class="pb-1 px-2 2xl:px-4"
        :show-generation-time-sort="activeTab === 'output'"
      />
      <Divider type="dashed" class="my-2" />
    </template>
    <template #body>
      <div
        v-if="showLoadingState"
        class="grid grid-cols-[repeat(auto-fill,minmax(200px,1fr))] gap-2 px-2"
      >
        <div
          v-for="n in skeletonCount"
          :key="`skeleton-${n}`"
          class="flex flex-col gap-2 p-2"
        >
          <Skeleton class="aspect-square w-full rounded-lg" />
          <div class="flex flex-col gap-1">
            <Skeleton class="h-4 w-3/4" />
            <Skeleton class="h-3 w-1/2" />
          </div>
        </div>
      </div>
      <div v-else-if="showEmptyState">
        <NoResultsPlaceholder
          icon="pi pi-info-circle"
          :title="
            $t(
              activeTab === 'input'
                ? 'sideToolbar.noImportedFiles'
                : 'sideToolbar.noGeneratedFiles'
            )
          "
          :message="$t('sideToolbar.noFilesFoundMessage')"
        />
      </div>
      <div v-else class="relative size-full" @click="handleEmptySpaceClick">
        <AssetsSidebarListView
          v-if="isListView"
          :asset-items="listViewAssetItems"
          :is-selected="isSelected"
          :selectable-assets="listViewSelectableAssets"
          :is-stack-expanded="isListViewStackExpanded"
          :toggle-stack="toggleListViewStack"
          :asset-type="activeTab"
          @select-asset="handleAssetSelect"
          @preview-asset="handleZoomClick"
          @context-menu="handleAssetContextMenu"
          @approach-end="handleApproachEnd"
        />
        <AssetsSidebarGridView
          v-else
          :assets="displayAssets"
          :is-selected="isSelected"
          :asset-type="activeTab"
          :show-output-count="shouldShowOutputCount"
          :get-output-count="getOutputCount"
          @select-asset="handleAssetSelect"
          @context-menu="handleAssetContextMenu"
          @approach-end="handleApproachEnd"
          @zoom="handleZoomClick"
          @output-count-click="enterFolderView"
        />
      </div>
    </template>
    <template #footer>
      <div
        v-if="hasSelection"
        ref="footerRef"
        class="flex gap-1 h-18 w-full items-center justify-between"
      >
        <div class="flex-1 pl-4">
          <div ref="selectionCountButtonRef" class="inline-flex w-48">
            <Button
              variant="secondary"
              :class="cn(isCompact && 'text-left')"
              @click="handleDeselectAll"
            >
              {{
                isHoveringSelectionCount
                  ? $t('mediaAsset.selection.deselectAll')
                  : $t('mediaAsset.selection.selectedCount', {
                      count: totalOutputCount
                    })
              }}
            </Button>
          </div>
        </div>
        <div class="flex shrink gap-2 pr-4 items-center-safe justify-end-safe">
          <template v-if="isCompact">
            <!-- Compact mode: Icon only -->
            <Button
              v-if="shouldShowDeleteButton"
              size="icon"
              @click="handleDeleteSelected"
            >
              <i class="icon-[lucide--trash-2] size-4" />
            </Button>
            <Button size="icon" @click="handleDownloadSelected">
              <i class="icon-[lucide--download] size-4" />
            </Button>
          </template>
          <template v-else>
            <!-- Normal mode: Icon + Text -->
            <Button
              v-if="shouldShowDeleteButton"
              variant="secondary"
              @click="handleDeleteSelected"
            >
              <span>{{ $t('mediaAsset.selection.deleteSelected') }}</span>
              <i class="icon-[lucide--trash-2] size-4" />
            </Button>
            <Button variant="secondary" @click="handleDownloadSelected">
              <span>{{ $t('mediaAsset.selection.downloadSelected') }}</span>
              <i class="icon-[lucide--download] size-4" />
            </Button>
          </template>
        </div>
      </div>
    </template>
  </SidebarTabTemplate>
  <ResultGallery
    v-model:active-index="galleryActiveIndex"
    :all-gallery-items="galleryItems"
  />
  <MediaAssetContextMenu
    v-if="contextMenuAsset"
    ref="contextMenuRef"
    :asset="contextMenuAsset"
    :asset-type="contextMenuAssetType"
    :file-kind="contextMenuFileKind"
    :show-delete-button="shouldShowDeleteButton"
    :selected-assets="selectedAssets"
    :is-bulk-mode="isBulkMode"
    @zoom="handleZoomClick(contextMenuAsset)"
    @hide="handleContextMenuHide"
    @asset-deleted="refreshAssets"
    @bulk-download="handleBulkDownload"
    @bulk-delete="handleBulkDelete"
    @bulk-add-to-workflow="handleBulkAddToWorkflow"
    @bulk-open-workflow="handleBulkOpenWorkflow"
    @bulk-export-workflow="handleBulkExportWorkflow"
  />
</template>

<script setup lang="ts">
import {
  useAsyncState,
  useDebounceFn,
  useElementHover,
  useResizeObserver,
  useStorage,
  useTimeoutFn
} from '@vueuse/core'
import Divider from 'primevue/divider'
import { useToast } from 'primevue/usetoast'
import {
  computed,
  defineAsyncComponent,
  nextTick,
  onMounted,
  onUnmounted,
  ref,
  watch
} from 'vue'
import { useI18n } from 'vue-i18n'

import NoResultsPlaceholder from '@/components/common/NoResultsPlaceholder.vue'
import AssetsSidebarGridView from '@/components/sidebar/tabs/AssetsSidebarGridView.vue'
import AssetsSidebarListView from '@/components/sidebar/tabs/AssetsSidebarListView.vue'
import SidebarTabTemplate from '@/components/sidebar/tabs/SidebarTabTemplate.vue'
import Skeleton from '@/components/ui/skeleton/Skeleton.vue'
import ResultGallery from '@/components/sidebar/tabs/queue/ResultGallery.vue'
import Tab from '@/components/tab/Tab.vue'
import TabList from '@/components/tab/TabList.vue'
import Button from '@/components/ui/button/Button.vue'
import MediaAssetContextMenu from '@/platform/assets/components/MediaAssetContextMenu.vue'
import MediaAssetFilterBar from '@/platform/assets/components/MediaAssetFilterBar.vue'
import { getAssetType } from '@/platform/assets/composables/media/assetMappers'
import { useMediaAssets } from '@/platform/assets/composables/media/useMediaAssets'
import { useAssetSelection } from '@/platform/assets/composables/useAssetSelection'
import { useMediaAssetActions } from '@/platform/assets/composables/useMediaAssetActions'
import { useMediaAssetFiltering } from '@/platform/assets/composables/useMediaAssetFiltering'
import { useOutputStacks } from '@/platform/assets/composables/useOutputStacks'
import type { OutputAssetMetadata } from '@/platform/assets/schemas/assetMetadataSchema'
import { getOutputAssetMetadata } from '@/platform/assets/schemas/assetMetadataSchema'
import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import type { MediaKind } from '@/platform/assets/schemas/mediaAssetSchema'
import { resolveOutputAssetItems } from '@/platform/assets/utils/outputAssetUtil'
import { isCloud } from '@/platform/distribution/types'
import { useDialogStore } from '@/stores/dialogStore'
import { ResultItemImpl } from '@/stores/queueStore'
import {
  formatDuration,
  getMediaTypeFromFilename,
  isPreviewableMediaType
} from '@/utils/formatUtil'
import { cn } from '@/utils/tailwindUtil'

const Load3dViewerContent = defineAsyncComponent(
  () => import('@/components/load3d/Load3dViewerContent.vue')
)

const { t } = useI18n()

const emit = defineEmits<{ assetSelected: [asset: AssetItem] }>()

const activeTab = ref<'input' | 'output'>('output')
const folderJobId = ref<string | null>(null)
const folderExecutionTime = ref<number | undefined>(undefined)
const expectedFolderCount = ref(0)
const isInFolderView = computed(() => folderJobId.value !== null)
const viewMode = useStorage<'list' | 'grid'>(
  'Comfy.Assets.Sidebar.ViewMode',
  'grid'
)
const isListView = computed(() => viewMode.value === 'list')

const contextMenuRef = ref<InstanceType<typeof MediaAssetContextMenu>>()
const contextMenuAsset = ref<AssetItem | null>(null)

// Determine if delete button should be shown
// Hide delete button when in input tab and not in cloud (OSS mode - files are from local folders)
const shouldShowDeleteButton = computed(() => {
  if (activeTab.value === 'input' && !isCloud) return false
  return true
})

const contextMenuAssetType = computed(() =>
  contextMenuAsset.value ? getAssetType(contextMenuAsset.value.tags) : 'input'
)

const contextMenuFileKind = computed<MediaKind>(() =>
  getMediaTypeFromFilename(contextMenuAsset.value?.name ?? '')
)

const shouldShowOutputCount = (item: AssetItem): boolean => {
  if (activeTab.value !== 'output' || isInFolderView.value) {
    return false
  }
  return getOutputCount(item) > 1
}

const formattedExecutionTime = computed(() => {
  if (!folderExecutionTime.value) return ''
  return formatDuration(folderExecutionTime.value * 1000)
})

const toast = useToast()

const inputAssets = useMediaAssets('input')
const outputAssets = useMediaAssets('output')

// Asset selection
const {
  isSelected,
  handleAssetClick,
  hasSelection,
  clearSelection,
  getSelectedAssets,
  reconcileSelection,
  getOutputCount,
  getTotalOutputCount,
  activate: activateSelection,
  deactivate: deactivateSelection
} = useAssetSelection()

const {
  downloadMultipleAssets,
  deleteAssets,
  addMultipleToWorkflow,
  openMultipleWorkflows,
  exportMultipleWorkflows
} = useMediaAssetActions()

// Footer responsive behavior
const footerRef = ref<HTMLElement | null>(null)
const footerWidth = ref(0)

// Track footer width changes
useResizeObserver(footerRef, (entries) => {
  const entry = entries[0]
  footerWidth.value = entry.contentRect.width
})

// Determine if we should show compact mode (icon only)
// Threshold matches when grid switches from 2 columns to 1 column
// 2 columns need about ~430px
const COMPACT_MODE_THRESHOLD_PX = 430
const isCompact = computed(
  () => footerWidth.value > 0 && footerWidth.value <= COMPACT_MODE_THRESHOLD_PX
)

// Hover state for selection count button
const selectionCountButtonRef = ref<HTMLElement | null>(null)
const isHoveringSelectionCount = useElementHover(selectionCountButtonRef)

// Total output count for all selected assets
const totalOutputCount = computed(() => {
  return getTotalOutputCount(selectedAssets.value)
})

const currentAssets = computed(() =>
  activeTab.value === 'input' ? inputAssets : outputAssets
)
const loading = computed(() => currentAssets.value.loading.value)
const error = computed(() => currentAssets.value.error.value)
const mediaAssets = computed(() => currentAssets.value.media.value)

const galleryActiveIndex = ref(-1)
const currentGalleryAssetId = ref<string | null>(null)

const DEFAULT_SKELETON_COUNT = 6
const skeletonCount = computed(() =>
  expectedFolderCount.value > 0
    ? expectedFolderCount.value
    : DEFAULT_SKELETON_COUNT
)

const {
  state: folderAssets,
  isLoading: folderLoading,
  error: folderError,
  execute: loadFolderAssets
} = useAsyncState(
  (metadata: OutputAssetMetadata, options: { createdAt?: string } = {}) =>
    resolveOutputAssetItems(metadata, options),
  [] as AssetItem[],
  { immediate: false, resetOnExecute: true }
)

// Base assets before search filtering
const baseAssets = computed(() => {
  if (isInFolderView.value) {
    return folderAssets.value
  }
  return mediaAssets.value
})

// Use media asset filtering composable
const { searchQuery, sortBy, mediaTypeFilters, filteredAssets } =
  useMediaAssetFiltering(baseAssets)

const displayAssets = computed(() => {
  return filteredAssets.value
})

const {
  assetItems: listViewAssetItems,
  selectableAssets: listViewSelectableAssets,
  isStackExpanded: isListViewStackExpanded,
  toggleStack: toggleListViewStack
} = useOutputStacks({
  assets: computed(() => displayAssets.value)
})

const visibleAssets = computed(() => {
  if (!isListView.value) return displayAssets.value
  return listViewSelectableAssets.value
})

const previewableVisibleAssets = computed(() =>
  visibleAssets.value.filter((asset) =>
    isPreviewableMediaType(getMediaTypeFromFilename(asset.name))
  )
)

const selectedAssets = computed(() => getSelectedAssets(visibleAssets.value))

const isBulkMode = computed(
  () => hasSelection.value && selectedAssets.value.length > 1
)

const isFolderLoading = computed(
  () => isInFolderView.value && folderLoading.value
)

const showLoadingState = computed(
  () =>
    (loading.value || isFolderLoading.value) && displayAssets.value.length === 0
)

const showEmptyState = computed(
  () =>
    !loading.value && !isFolderLoading.value && displayAssets.value.length === 0
)

watch(visibleAssets, (newAssets) => {
  // Alternative: keep hidden selections and surface them in UI; for now prune
  // so selection stays consistent with what this view can act on.
  reconcileSelection(newAssets)
  if (currentGalleryAssetId.value && galleryActiveIndex.value !== -1) {
    const newIndex = previewableVisibleAssets.value.findIndex(
      (asset) => asset.id === currentGalleryAssetId.value
    )
    galleryActiveIndex.value = newIndex
  }
})

watch(galleryActiveIndex, (index) => {
  if (index === -1) {
    currentGalleryAssetId.value = null
  }
})

const galleryItems = computed(() => {
  return previewableVisibleAssets.value.map((asset) => {
    const mediaType = getMediaTypeFromFilename(asset.name)
    const resultItem = new ResultItemImpl({
      filename: asset.name,
      subfolder: '',
      type: 'output',
      nodeId: '0',
      mediaType: mediaType === 'image' ? 'images' : mediaType
    })

    Object.defineProperty(resultItem, 'url', {
      get() {
        return asset.preview_url || ''
      },
      configurable: true
    })

    return resultItem
  })
})

const refreshAssets = async () => {
  await currentAssets.value.fetchMediaList()
  if (error.value) {
    console.error('Failed to refresh assets:', error.value)
  }
}

watch(
  activeTab,
  () => {
    clearSelection()
    // Clear search when switching tabs
    searchQuery.value = ''
    // Reset pagination state when tab changes
    void refreshAssets()
  },
  { immediate: true }
)

function handleAssetSelect(asset: AssetItem, assets?: AssetItem[]) {
  const assetList = assets ?? visibleAssets.value
  const index = assetList.findIndex((a) => a.id === asset.id)
  emit('assetSelected', asset)
  handleAssetClick(asset, index, assetList)
}

const { start: scheduleCleanup, stop: cancelCleanup } = useTimeoutFn(
  () => {
    contextMenuAsset.value = null
  },
  0,
  { immediate: false }
)

function handleAssetContextMenu(event: MouseEvent, asset: AssetItem) {
  cancelCleanup()
  contextMenuAsset.value = asset
  void nextTick(() => {
    contextMenuRef.value?.show(event)
  })
}

function handleContextMenuHide() {
  scheduleCleanup()
}

const handleBulkDownload = (assets: AssetItem[]) => {
  downloadMultipleAssets(assets)
  clearSelection()
}

const handleBulkDelete = async (assets: AssetItem[]) => {
  if (await deleteAssets(assets)) {
    clearSelection()
  }
}

const handleBulkAddToWorkflow = async (assets: AssetItem[]) => {
  await addMultipleToWorkflow(assets)
  clearSelection()
}

const handleBulkOpenWorkflow = async (assets: AssetItem[]) => {
  await openMultipleWorkflows(assets)
  clearSelection()
}

const handleBulkExportWorkflow = async (assets: AssetItem[]) => {
  await exportMultipleWorkflows(assets)
  clearSelection()
}

const handleDownloadSelected = () => {
  downloadMultipleAssets(selectedAssets.value)
  clearSelection()
}

const handleDeleteSelected = async () => {
  if (await deleteAssets(selectedAssets.value)) {
    clearSelection()
  }
}

const handleZoomClick = (asset: AssetItem) => {
  const mediaType = getMediaTypeFromFilename(asset.name)
  if (!isPreviewableMediaType(mediaType)) {
    return
  }

  if (mediaType === '3D') {
    const dialogStore = useDialogStore()
    dialogStore.showDialog({
      key: 'asset-3d-viewer',
      title: asset.name,
      component: Load3dViewerContent,
      props: {
        modelUrl: asset.preview_url || ''
      },
      dialogComponentProps: {
        style: 'width: 80vw; height: 80vh;',
        maximizable: true
      }
    })
    return
  }

  currentGalleryAssetId.value = asset.id
  const index = previewableVisibleAssets.value.findIndex(
    (a) => a.id === asset.id
  )
  if (index !== -1) {
    galleryActiveIndex.value = index
  }
}

const enterFolderView = async (asset: AssetItem) => {
  const metadata = getOutputAssetMetadata(asset.user_metadata)
  if (!metadata) {
    console.warn('Invalid output asset metadata')
    return
  }

  const { jobId, executionTimeInSeconds } = metadata

  if (!jobId) {
    console.warn('Missing required folder view data')
    return
  }

  folderJobId.value = jobId
  folderExecutionTime.value = executionTimeInSeconds
  expectedFolderCount.value = metadata.outputCount ?? 0

  await loadFolderAssets(0, metadata, { createdAt: asset.created_at })

  if (folderError.value) {
    toast.add({
      severity: 'error',
      summary: t('sideToolbar.folderView.errorSummary'),
      detail: t('sideToolbar.folderView.errorDetail'),
      life: 5000
    })
    exitFolderView()
  }
}

const exitFolderView = () => {
  folderJobId.value = null
  folderExecutionTime.value = undefined
  expectedFolderCount.value = 0
  folderAssets.value = []
  searchQuery.value = ''
}

onMounted(() => {
  activateSelection()
})

onUnmounted(() => {
  deactivateSelection()
})

const handleDeselectAll = () => {
  clearSelection()
}

const handleEmptySpaceClick = () => {
  if (hasSelection) {
    clearSelection()
  }
}

const copyJobId = async () => {
  if (folderJobId.value) {
    try {
      await navigator.clipboard.writeText(folderJobId.value)
      toast.add({
        severity: 'success',
        summary: t('mediaAsset.jobIdToast.copied'),
        detail: t('mediaAsset.jobIdToast.jobIdCopied'),
        life: 2000
      })
    } catch (error) {
      toast.add({
        severity: 'error',
        summary: t('mediaAsset.jobIdToast.error'),
        detail: t('mediaAsset.jobIdToast.jobIdCopyFailed'),
        life: 3000
      })
    }
  }
}

const handleApproachEnd = useDebounceFn(async () => {
  if (
    activeTab.value === 'output' &&
    !isInFolderView.value &&
    outputAssets.hasMore.value &&
    !outputAssets.isLoadingMore.value
  ) {
    await outputAssets.loadMore()
  }
}, 300)
</script>
