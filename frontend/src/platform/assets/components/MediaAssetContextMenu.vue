<template>
  <ContextMenu
    ref="contextMenu"
    :model="contextMenuItems"
    :pt="{
      root: {
        class: cn(
          'rounded-lg',
          'bg-secondary-background text-base-foreground',
          'shadow-lg'
        )
      }
    }"
    @hide="onMenuHide"
  >
    <template #item="{ item, props }">
      <Button
        variant="secondary"
        class="w-full justify-start"
        v-bind="props.action"
      >
        <i v-if="item.icon" :class="item.icon" class="size-4" />
        <span>{{
          typeof item.label === 'function' ? item.label() : (item.label ?? '')
        }}</span>
      </Button>
    </template>
  </ContextMenu>
</template>

<script setup lang="ts">
import { useEventListener } from '@vueuse/core'
import ContextMenu from 'primevue/contextmenu'
import type { MenuItem } from 'primevue/menuitem'
import type { ComponentPublicInstance } from 'vue'
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import { isCloud } from '@/platform/distribution/types'
import { supportsWorkflowMetadata } from '@/platform/workflow/utils/workflowExtractionUtil'
import { isPreviewableMediaType } from '@/utils/formatUtil'
import { detectNodeTypeFromFilename } from '@/utils/loaderNodeUtil'
import { cn } from '@/utils/tailwindUtil'

import { useMediaAssetActions } from '../composables/useMediaAssetActions'
import type { AssetItem } from '../schemas/assetSchema'
import type { AssetContext, MediaKind } from '../schemas/mediaAssetSchema'

const {
  asset,
  assetType,
  fileKind,
  showDeleteButton,
  selectedAssets,
  isBulkMode
} = defineProps<{
  asset: AssetItem
  assetType: AssetContext['type']
  fileKind: MediaKind
  showDeleteButton?: boolean
  selectedAssets?: AssetItem[]
  isBulkMode?: boolean
}>()

const emit = defineEmits<{
  zoom: []
  hide: []
  'asset-deleted': []
  'bulk-download': [assets: AssetItem[]]
  'bulk-delete': [assets: AssetItem[]]
  'bulk-add-to-workflow': [assets: AssetItem[]]
  'bulk-open-workflow': [assets: AssetItem[]]
  'bulk-export-workflow': [assets: AssetItem[]]
}>()

type ContextMenuInstance = ComponentPublicInstance & {
  show: (event: MouseEvent) => void
  hide: () => void
  container?: HTMLElement
  $el?: HTMLElement
}

const contextMenu = ref<ContextMenuInstance | null>(null)
const isVisible = ref(false)
const actions = useMediaAssetActions()
const { t } = useI18n()

function getOverlayEl(): HTMLElement | null {
  return contextMenu.value?.container ?? contextMenu.value?.$el ?? null
}

function dismissIfOutside(event: Event) {
  if (!isVisible.value) return
  const overlay = getOverlayEl()
  if (!overlay) return
  if (overlay.contains(event.target as Node)) return
  hide()
}

useEventListener(window, 'pointerdown', dismissIfOutside, { capture: true })
useEventListener(window, 'scroll', dismissIfOutside, {
  capture: true,
  passive: true
})

const showAddToWorkflow = computed(() => {
  // Output assets can always be added
  if (assetType === 'output') return true

  // Input assets: check if file type is supported by loader nodes
  if (assetType === 'input' && asset?.name) {
    const { nodeType } = detectNodeTypeFromFilename(asset.name)
    return nodeType !== null
  }

  return false
})

const showWorkflowActions = computed(() => {
  // Output assets always have workflow metadata
  if (assetType === 'output') return true

  // Input assets: only formats that support workflow metadata
  if (assetType === 'input' && asset?.name) {
    return supportsWorkflowMetadata(asset.name)
  }

  return false
})

const showCopyJobId = computed(() => {
  return assetType !== 'input'
})

const shouldShowDeleteButton = computed(() => {
  const propAllows = showDeleteButton ?? true
  const typeAllows =
    assetType === 'output' || (assetType === 'input' && isCloud)

  return propAllows && typeAllows
})

// Context menu items
const contextMenuItems = computed<MenuItem[]>(() => {
  if (!asset) return []

  const items: MenuItem[] = []

  // Check if current asset is part of the selection
  const isCurrentAssetSelected = selectedAssets?.some(
    (selectedAsset) => selectedAsset.id === asset.id
  )

  // Bulk mode: Show selected count and bulk actions only if current asset is selected
  if (
    isBulkMode &&
    selectedAssets &&
    selectedAssets.length > 0 &&
    isCurrentAssetSelected
  ) {
    // Header item showing selected count
    items.push({
      label: t('mediaAsset.selection.multipleSelectedAssets'),
      disabled: true
    })

    // Bulk Add to Workflow
    items.push({
      label: t('mediaAsset.selection.insertAllAssetsAsNodes'),
      icon: 'icon-[comfy--node]',
      command: () => emit('bulk-add-to-workflow', selectedAssets)
    })

    // Bulk Open Workflow
    items.push({
      label: t('mediaAsset.selection.openWorkflowAll'),
      icon: 'icon-[comfy--workflow]',
      command: () => emit('bulk-open-workflow', selectedAssets)
    })

    // Bulk Export Workflow
    items.push({
      label: t('mediaAsset.selection.exportWorkflowAll'),
      icon: 'icon-[lucide--file-output]',
      command: () => emit('bulk-export-workflow', selectedAssets)
    })

    // Bulk Download
    items.push({
      label: t('mediaAsset.selection.downloadSelectedAll'),
      icon: 'icon-[lucide--download]',
      command: () => emit('bulk-download', selectedAssets)
    })

    // Bulk Delete (if allowed)
    if (shouldShowDeleteButton.value) {
      items.push({
        label: t('mediaAsset.selection.deleteSelectedAll'),
        icon: 'icon-[lucide--trash-2]',
        command: () => emit('bulk-delete', selectedAssets)
      })
    }

    return items
  }

  // Individual mode: Show all menu options

  // Inspect
  if (isPreviewableMediaType(fileKind)) {
    items.push({
      label: t('mediaAsset.actions.inspect'),
      icon: 'icon-[lucide--zoom-in]',
      command: () => emit('zoom')
    })
  }

  // Add to workflow (conditional)
  if (showAddToWorkflow.value) {
    items.push({
      label: t('mediaAsset.actions.insertAsNodeInWorkflow'),
      icon: 'icon-[comfy--node]',
      command: () => actions.addWorkflow(asset)
    })
  }

  // Download
  items.push({
    label: t('mediaAsset.actions.download'),
    icon: 'icon-[lucide--download]',
    command: () => actions.downloadAsset(asset)
  })

  // Separator before workflow actions (only if there are workflow actions)
  if (showWorkflowActions.value) {
    items.push({ separator: true })
    items.push({
      label: t('mediaAsset.actions.openWorkflow'),
      icon: 'icon-[comfy--workflow]',
      command: () => actions.openWorkflow(asset)
    })
    items.push({
      label: t('mediaAsset.actions.exportWorkflow'),
      icon: 'icon-[lucide--file-output]',
      command: () => actions.exportWorkflow(asset)
    })
  }

  // Copy job ID
  if (showCopyJobId.value) {
    items.push({ separator: true })
    items.push({
      label: t('mediaAsset.actions.copyJobId'),
      icon: 'icon-[lucide--copy]',
      command: async () => {
        await actions.copyJobId(asset)
      }
    })
  }

  // Delete
  if (shouldShowDeleteButton.value) {
    items.push({ separator: true })
    items.push({
      label: t('mediaAsset.actions.delete'),
      icon: 'icon-[lucide--trash-2]',
      command: async () => {
        if (asset) {
          const confirmed = await actions.deleteAssets(asset)
          if (confirmed) {
            emit('asset-deleted')
          }
        }
      }
    })
  }

  return items
})

function onMenuHide() {
  isVisible.value = false
  emit('hide')
}

function show(event: MouseEvent) {
  isVisible.value = true
  contextMenu.value?.show(event)
}

function hide() {
  isVisible.value = false
  contextMenu.value?.hide()
}

defineExpose({ show, hide })
</script>
