import { useToast } from 'primevue/usetoast'
import { inject } from 'vue'
import { useI18n } from 'vue-i18n'

import ConfirmationDialogContent from '@/components/dialog/content/ConfirmationDialogContent.vue'
import { downloadFile } from '@/base/common/downloadUtil'
import { useCopyToClipboard } from '@/composables/useCopyToClipboard'
import { isCloud } from '@/platform/distribution/types'
import { useWorkflowActionsService } from '@/platform/workflow/core/services/workflowActionsService'
import { extractWorkflowFromAsset } from '@/platform/workflow/utils/workflowExtractionUtil'
import { api } from '@/scripts/api'
import { useLitegraphService } from '@/services/litegraphService'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import { getOutputAssetMetadata } from '../schemas/assetMetadataSchema'
import { useAssetsStore } from '@/stores/assetsStore'
import { useDialogStore } from '@/stores/dialogStore'
import { getAssetType } from '../utils/assetTypeUtil'
import { getAssetUrl } from '../utils/assetUrlUtil'
import { createAnnotatedPath } from '@/utils/createAnnotatedPath'
import { detectNodeTypeFromFilename } from '@/utils/loaderNodeUtil'
import { isResultItemType } from '@/utils/typeGuardUtil'

import { useAssetExportStore } from '@/stores/assetExportStore'

import type { AssetItem } from '../schemas/assetSchema'
import { MediaAssetKey } from '../schemas/mediaAssetSchema'
import { assetService } from '../services/assetService'

const EXCLUDED_TAGS = new Set(['models', 'input', 'output'])

export function useMediaAssetActions() {
  const { t } = useI18n()
  const toast = useToast()
  const dialogStore = useDialogStore()
  const mediaContext = inject(MediaAssetKey, null)
  const { copyToClipboard } = useCopyToClipboard()
  const workflowActions = useWorkflowActionsService()
  const litegraphService = useLitegraphService()
  const nodeDefStore = useNodeDefStore()

  /**
   * Internal helper to perform the API deletion for a single asset
   * Handles both output assets (via history API) and input assets (via asset service)
   * @throws Error if deletion fails or is not allowed
   */
  const deleteAssetApi = async (
    asset: AssetItem,
    assetType: string
  ): Promise<void> => {
    if (assetType === 'output') {
      const jobId =
        getOutputAssetMetadata(asset.user_metadata)?.jobId || asset.id
      if (!jobId) {
        throw new Error('Unable to extract job ID from asset')
      }
      await api.deleteItem('history', jobId)
    } else {
      // Input assets can only be deleted in cloud environment
      if (!isCloud) {
        throw new Error(t('mediaAsset.deletingImportedFilesCloudOnly'))
      }
      await assetService.deleteAsset(asset.id)
    }
  }

  const downloadAsset = (asset?: AssetItem) => {
    const targetAsset = asset ?? mediaContext?.asset.value
    if (!targetAsset) return

    try {
      const filename = targetAsset.name
      // Prefer preview_url (already includes subfolder) with getAssetUrl as fallback
      const downloadUrl = targetAsset.preview_url || getAssetUrl(targetAsset)

      downloadFile(downloadUrl, filename)

      toast.add({
        severity: 'success',
        summary: t('g.success'),
        detail: t('mediaAsset.selection.downloadsStarted', 1),
        life: 2000
      })
    } catch (error) {
      toast.add({
        severity: 'error',
        summary: t('g.error'),
        detail: t('g.failedToDownloadImage'),
        life: 3000
      })
    }
  }

  /**
   * Download multiple assets at once.
   * In cloud mode with 2+ assets, creates a ZIP export via the backend.
   * Falls back to individual downloads in OSS mode or for single assets.
   */
  const downloadMultipleAssets = (assets: AssetItem[]) => {
    if (!assets || assets.length === 0) return

    const hasMultiOutputJobs = assets.some((a) => {
      const count = getOutputAssetMetadata(a.user_metadata)?.outputCount
      return typeof count === 'number' && count > 1
    })

    if (isCloud && (assets.length > 1 || hasMultiOutputJobs)) {
      void downloadMultipleAssetsAsZip(assets)
      return
    }

    try {
      assets.forEach((asset) => {
        const filename = asset.name
        const downloadUrl = asset.preview_url || getAssetUrl(asset)
        downloadFile(downloadUrl, filename)
      })

      toast.add({
        severity: 'success',
        summary: t('g.success'),
        detail: t('mediaAsset.selection.downloadsStarted', assets.length),
        life: 2000
      })
    } catch (error) {
      console.error('Failed to download assets:', error)
      toast.add({
        severity: 'error',
        summary: t('g.error'),
        detail: t('g.failedToDownloadImage'),
        life: 3000
      })
    }
  }

  async function downloadMultipleAssetsAsZip(assets: AssetItem[]) {
    const assetExportStore = useAssetExportStore()

    try {
      const jobIds: string[] = []
      const assetIds: string[] = []
      const jobAssetNameFilters: Record<string, string[]> = {}

      for (const asset of assets) {
        if (getAssetType(asset) === 'output') {
          const metadata = getOutputAssetMetadata(asset.user_metadata)
          const jobId = metadata?.jobId || asset.id
          if (!jobIds.includes(jobId)) {
            jobIds.push(jobId)
          }
          if (metadata?.jobId && asset.name) {
            if (!jobAssetNameFilters[metadata.jobId]) {
              jobAssetNameFilters[metadata.jobId] = []
            }
            if (!jobAssetNameFilters[metadata.jobId].includes(asset.name)) {
              jobAssetNameFilters[metadata.jobId].push(asset.name)
            }
          }
        } else {
          assetIds.push(asset.id)
        }
      }

      const result = await assetService.createAssetExport({
        ...(jobIds.length > 0 ? { job_ids: jobIds } : {}),
        ...(assetIds.length > 0 ? { asset_ids: assetIds } : {}),
        ...(Object.keys(jobAssetNameFilters).length > 0
          ? { job_asset_name_filters: jobAssetNameFilters }
          : {}),
        naming_strategy: 'preserve'
      })

      assetExportStore.trackExport(result.task_id)

      toast.add({
        severity: 'info',
        summary: t('exportToast.exportStarted'),
        detail: t('mediaAsset.selection.exportStarted', assets.length),
        life: 3000
      })
    } catch (error) {
      console.error('Failed to create asset export:', error)
      toast.add({
        severity: 'error',
        summary: t('g.error'),
        detail: t('exportToast.exportFailedSingle'),
        life: 3000
      })
    }
  }

  const copyJobId = async (asset?: AssetItem) => {
    const targetAsset = asset ?? mediaContext?.asset.value
    if (!targetAsset) return

    const metadata = getOutputAssetMetadata(targetAsset.user_metadata)
    const jobId =
      metadata?.jobId ||
      (getAssetType(targetAsset) === 'output' ? targetAsset.id : undefined)

    if (!jobId) {
      toast.add({
        severity: 'warn',
        summary: t('g.warning'),
        detail: t('mediaAsset.noJobIdFound'),
        life: 2000
      })
      return
    }

    await copyToClipboard(jobId)
  }

  /**
   * Add a loader node to the current workflow for this asset
   * Uses shared utility to detect appropriate node type based on file extension
   */
  const addWorkflow = async (asset?: AssetItem) => {
    const targetAsset = asset ?? mediaContext?.asset.value
    if (!targetAsset) return

    // Detect node type using shared utility
    const { nodeType, widgetName } = detectNodeTypeFromFilename(
      targetAsset.name
    )

    if (!nodeType || !widgetName) {
      toast.add({
        severity: 'warn',
        summary: t('g.warning'),
        detail: t('mediaAsset.unsupportedFileType'),
        life: 2000
      })
      return
    }

    const nodeDef = nodeDefStore.nodeDefsByName[nodeType]
    if (!nodeDef) {
      toast.add({
        severity: 'error',
        summary: t('g.error'),
        detail: t('mediaAsset.nodeTypeNotFound', { nodeType }),
        life: 3000
      })
      return
    }

    const node = litegraphService.addNodeOnGraph(nodeDef, {
      pos: litegraphService.getCanvasCenter()
    })

    if (!node) {
      toast.add({
        severity: 'error',
        summary: t('g.error'),
        detail: t('mediaAsset.failedToCreateNode'),
        life: 3000
      })
      return
    }

    // Get metadata to construct the annotated path
    const metadata = getOutputAssetMetadata(targetAsset.user_metadata)
    const assetType = getAssetType(targetAsset, 'input')

    // In Cloud mode, use asset_hash (the actual stored filename)
    // In OSS mode, use the original name
    const filename =
      isCloud && targetAsset.asset_hash
        ? targetAsset.asset_hash
        : targetAsset.name

    // Create annotated path for the asset
    const annotated = createAnnotatedPath(
      {
        filename,
        subfolder: metadata?.subfolder || '',
        type: isResultItemType(assetType) ? assetType : undefined
      },
      {
        rootFolder: isResultItemType(assetType) ? assetType : undefined
      }
    )

    const widget = node.widgets?.find((w) => w.name === widgetName)
    if (widget) {
      widget.value = annotated
      widget.callback?.(annotated)
    }
    node.graph?.setDirtyCanvas(true, true)

    toast.add({
      severity: 'success',
      summary: t('g.success'),
      detail: t('mediaAsset.nodeAddedToWorkflow', { nodeType }),
      life: 2000
    })
  }

  /**
   * Open the workflow from this asset in a new tab
   * Uses shared workflow extraction and action service
   */
  const openWorkflow = async (asset?: AssetItem) => {
    const targetAsset = asset ?? mediaContext?.asset.value
    if (!targetAsset) return

    // Extract workflow using shared utility
    const { workflow, filename } = await extractWorkflowFromAsset(targetAsset)

    // Use shared action service
    const result = await workflowActions.openWorkflowAction(workflow, filename)

    if (!result.success) {
      toast.add({
        severity: 'warn',
        summary: t('g.warning'),
        detail: result.error || t('mediaAsset.noWorkflowDataFound'),
        life: 2000
      })
    } else {
      toast.add({
        severity: 'success',
        summary: t('g.success'),
        detail: t('mediaAsset.workflowOpenedInNewTab'),
        life: 2000
      })
    }
  }

  /**
   * Export the workflow from this asset as a JSON file
   * Uses shared workflow extraction and action service
   */
  const exportWorkflow = async (asset?: AssetItem) => {
    const targetAsset = asset ?? mediaContext?.asset.value
    if (!targetAsset) return

    // Extract workflow using shared utility
    const { workflow, filename } = await extractWorkflowFromAsset(targetAsset)

    // Use shared action service
    const result = await workflowActions.exportWorkflowAction(
      workflow,
      filename
    )

    if (!result.success) {
      const isNoWorkflow = result.error?.includes('No workflow')
      toast.add({
        severity: isNoWorkflow ? 'warn' : 'error',
        summary: isNoWorkflow ? t('g.warning') : t('g.error'),
        detail: result.error || t('mediaAsset.failedToExportWorkflow'),
        life: 3000
      })
    } else {
      toast.add({
        severity: 'success',
        summary: t('g.success'),
        detail: t('mediaAsset.workflowExportedSuccessfully'),
        life: 2000
      })
    }
  }

  /**
   * Add multiple assets to the current workflow
   * Creates loader nodes for each asset
   */
  const addMultipleToWorkflow = async (assets: AssetItem[]) => {
    if (!assets || assets.length === 0) return

    const NODE_OFFSET = 50
    let nodeIndex = 0
    let succeeded = 0
    let failed = 0

    for (const asset of assets) {
      const { nodeType, widgetName } = detectNodeTypeFromFilename(asset.name)

      if (!nodeType || !widgetName) {
        failed++
        continue
      }

      const nodeDef = nodeDefStore.nodeDefsByName[nodeType]
      if (!nodeDef) {
        failed++
        continue
      }

      const center = litegraphService.getCanvasCenter()
      const node = litegraphService.addNodeOnGraph(nodeDef, {
        pos: [
          center[0] + nodeIndex * NODE_OFFSET,
          center[1] + nodeIndex * NODE_OFFSET
        ]
      })

      if (!node) {
        failed++
        continue
      }

      const metadata = getOutputAssetMetadata(asset.user_metadata)
      const assetType = getAssetType(asset, 'input')

      // In Cloud mode, use asset_hash (the actual stored filename)
      // In OSS mode, use the original name
      const filename =
        isCloud && asset.asset_hash ? asset.asset_hash : asset.name

      const annotated = createAnnotatedPath(
        {
          filename,
          subfolder: metadata?.subfolder || '',
          type: isResultItemType(assetType) ? assetType : undefined
        },
        {
          rootFolder: isResultItemType(assetType) ? assetType : undefined
        }
      )

      const widget = node.widgets?.find((w) => w.name === widgetName)
      if (widget) {
        widget.value = annotated
        widget.callback?.(annotated)
      }
      node.graph?.setDirtyCanvas(true, true)
      succeeded++
      nodeIndex++
    }

    if (failed === 0) {
      toast.add({
        severity: 'success',
        summary: t('g.success'),
        detail: t('mediaAsset.selection.nodesAddedToWorkflow', {
          count: succeeded
        }),
        life: 2000
      })
    } else if (succeeded === 0) {
      toast.add({
        severity: 'error',
        summary: t('g.error'),
        detail: t('mediaAsset.selection.failedToAddNodes'),
        life: 3000
      })
    } else {
      toast.add({
        severity: 'warn',
        summary: t('g.warning'),
        detail: t('mediaAsset.selection.partialAddNodesSuccess', {
          succeeded,
          failed
        }),
        life: 3000
      })
    }
  }

  /**
   * Open workflows from multiple assets in new tabs
   */
  const openMultipleWorkflows = async (assets: AssetItem[]) => {
    if (!assets || assets.length === 0) return

    let succeeded = 0
    let failed = 0

    for (const asset of assets) {
      try {
        const { workflow, filename } = await extractWorkflowFromAsset(asset)
        const result = await workflowActions.openWorkflowAction(
          workflow,
          filename
        )

        if (result.success) {
          succeeded++
        } else {
          failed++
        }
      } catch {
        failed++
      }
    }

    if (failed === 0) {
      toast.add({
        severity: 'success',
        summary: t('g.success'),
        detail: t('mediaAsset.selection.workflowsOpened', { count: succeeded }),
        life: 2000
      })
    } else if (succeeded === 0) {
      toast.add({
        severity: 'warn',
        summary: t('g.warning'),
        detail: t('mediaAsset.selection.noWorkflowsFound'),
        life: 3000
      })
    } else {
      toast.add({
        severity: 'warn',
        summary: t('g.warning'),
        detail: t('mediaAsset.selection.partialWorkflowsOpened', {
          succeeded,
          failed
        }),
        life: 3000
      })
    }
  }

  /**
   * Export workflows from multiple assets as JSON files
   */
  const exportMultipleWorkflows = async (assets: AssetItem[]) => {
    if (!assets || assets.length === 0) return

    let succeeded = 0
    let failed = 0

    for (const asset of assets) {
      try {
        const { workflow, filename } = await extractWorkflowFromAsset(asset)
        const result = await workflowActions.exportWorkflowAction(
          workflow,
          filename
        )

        if (result.success) {
          succeeded++
        } else {
          failed++
        }
      } catch {
        failed++
      }
    }

    if (failed === 0) {
      toast.add({
        severity: 'success',
        summary: t('g.success'),
        detail: t('mediaAsset.selection.workflowsExported', {
          count: succeeded
        }),
        life: 2000
      })
    } else if (succeeded === 0) {
      toast.add({
        severity: 'warn',
        summary: t('g.warning'),
        detail: t('mediaAsset.selection.noWorkflowsToExport'),
        life: 3000
      })
    } else {
      toast.add({
        severity: 'warn',
        summary: t('g.warning'),
        detail: t('mediaAsset.selection.partialWorkflowsExported', {
          succeeded,
          failed
        }),
        life: 3000
      })
    }
  }

  /**
   * Show confirmation dialog and delete asset(s) if confirmed
   * @param assets Single asset or array of assets to delete
   * @returns true if user confirmed and deletion was attempted, false if cancelled
   */
  const deleteAssets = async (
    assets: AssetItem | AssetItem[]
  ): Promise<boolean> => {
    const assetArray = Array.isArray(assets) ? assets : [assets]
    if (assetArray.length === 0) return false

    const assetsStore = useAssetsStore()
    const isSingle = assetArray.length === 1

    return new Promise((resolve) => {
      dialogStore.showDialog({
        key: 'delete-assets-confirmation',
        title: isSingle
          ? t('mediaAsset.deleteAssetTitle')
          : t('mediaAsset.deleteSelectedTitle'),
        component: ConfirmationDialogContent,
        props: {
          message: isSingle
            ? t('mediaAsset.deleteAssetDescription')
            : t('mediaAsset.deleteSelectedDescription', {
                count: assetArray.length
              }),
          type: 'delete',
          itemList: assetArray.map((asset) => asset.name),
          onConfirm: async () => {
            // Show loading overlay for all assets being deleted
            assetArray.forEach((asset) =>
              assetsStore.setAssetDeleting(asset.id, true)
            )

            try {
              // Delete all assets using Promise.allSettled to track individual results
              const results = await Promise.allSettled(
                assetArray.map((asset) =>
                  deleteAssetApi(asset, getAssetType(asset))
                )
              )

              // Count successes and failures
              const succeeded = results.filter(
                (r) => r.status === 'fulfilled'
              ).length
              const failed = results.filter((r) => r.status === 'rejected')

              // Log failed deletions for debugging
              failed.forEach((result, index) => {
                console.warn(
                  `Failed to delete asset ${assetArray[index].name}:`,
                  result.reason
                )
              })

              // Update stores after deletions
              const hasOutputAssets = assetArray.some(
                (a) => getAssetType(a) === 'output'
              )
              const hasInputAssets = assetArray.some(
                (a) => getAssetType(a) === 'input'
              )

              if (hasOutputAssets) {
                await assetsStore.updateHistory()
              }
              if (hasInputAssets) {
                await assetsStore.updateInputs()
              }

              // Invalidate model caches for affected categories
              const modelCategories = new Set<string>()

              for (const asset of assetArray) {
                for (const tag of asset.tags ?? []) {
                  if (EXCLUDED_TAGS.has(tag)) continue
                  if (assetsStore.hasCategory(tag)) {
                    modelCategories.add(tag)
                  }
                }
              }

              for (const category of modelCategories) {
                assetsStore.invalidateModelsForCategory(category)
              }

              // Show appropriate feedback based on results
              if (failed.length === 0) {
                toast.add({
                  severity: 'success',
                  summary: t('g.success'),
                  detail: isSingle
                    ? t('mediaAsset.assetDeletedSuccessfully')
                    : t(
                        'mediaAsset.selection.assetsDeletedSuccessfully',
                        succeeded
                      ),
                  life: 2000
                })
              } else if (succeeded === 0) {
                toast.add({
                  severity: 'error',
                  summary: t('g.error'),
                  detail: isSingle
                    ? t('mediaAsset.failedToDeleteAsset')
                    : t('mediaAsset.selection.failedToDeleteAssets'),
                  life: 3000
                })
              } else {
                // Partial success (only possible with multiple assets)
                toast.add({
                  severity: 'warn',
                  summary: t('g.warning'),
                  detail: t('mediaAsset.selection.partialDeleteSuccess', {
                    succeeded,
                    failed: failed.length
                  }),
                  life: 3000
                })
              }
            } catch (error) {
              console.error('Failed to delete assets:', error)
              toast.add({
                severity: 'error',
                summary: t('g.error'),
                detail: isSingle
                  ? t('mediaAsset.failedToDeleteAsset')
                  : t('mediaAsset.selection.failedToDeleteAssets'),
                life: 3000
              })
            } finally {
              // Hide loading overlay for all assets
              assetArray.forEach((asset) =>
                assetsStore.setAssetDeleting(asset.id, false)
              )
            }

            resolve(true)
          },
          onCancel: () => {
            resolve(false)
          }
        }
      })
    })
  }

  return {
    downloadAsset,
    downloadMultipleAssets,
    deleteAssets,
    copyJobId,
    addWorkflow,
    addMultipleToWorkflow,
    openWorkflow,
    openMultipleWorkflows,
    exportWorkflow,
    exportMultipleWorkflows
  }
}
