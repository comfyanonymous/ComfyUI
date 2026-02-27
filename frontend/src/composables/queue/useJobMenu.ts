import { computed } from 'vue'

import { downloadFile } from '@/base/common/downloadUtil'
import type { JobListItem } from '@/composables/queue/useJobList'
import { useCopyToClipboard } from '@/composables/useCopyToClipboard'
import { st, t } from '@/i18n'
import { mapTaskOutputToAssetItem } from '@/platform/assets/composables/media/assetMappers'
import { useMediaAssetActions } from '@/platform/assets/composables/useMediaAssetActions'
import { isCloud } from '@/platform/distribution/types'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useWorkflowService } from '@/platform/workflow/core/services/workflowService'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import type { ResultItem, ResultItemType } from '@/schemas/apiSchema'
import { api } from '@/scripts/api'
import { downloadBlob } from '@/scripts/utils'
import { useDialogService } from '@/services/dialogService'
import { getJobWorkflow } from '@/services/jobOutputCache'
import { useLitegraphService } from '@/services/litegraphService'
import { useExecutionStore } from '@/stores/executionStore'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import { useQueueStore } from '@/stores/queueStore'
import type { ResultItemImpl, TaskItemImpl } from '@/stores/queueStore'
import { createAnnotatedPath } from '@/utils/createAnnotatedPath'
import { appendJsonExt } from '@/utils/formatUtil'

export type MenuEntry =
  | {
      kind?: 'item'
      key: string
      label: string
      icon?: string
      disabled?: boolean
      onClick?: () => void | Promise<void>
    }
  | { kind: 'divider'; key: string }

/**
 * Provides job context menu entries and actions.
 *
 * @param currentMenuItem Getter for the currently targeted job list item
 * @param onInspectAsset Callback to trigger when inspecting a completed job's asset
 */
export function useJobMenu(
  currentMenuItem: () => JobListItem | null = () => null,
  onInspectAsset?: (item: JobListItem) => void
) {
  const workflowStore = useWorkflowStore()
  const workflowService = useWorkflowService()
  const queueStore = useQueueStore()
  const executionStore = useExecutionStore()
  const { copyToClipboard } = useCopyToClipboard()
  const litegraphService = useLitegraphService()
  const nodeDefStore = useNodeDefStore()
  const mediaAssetActions = useMediaAssetActions()

  const resolveItem = (item?: JobListItem | null): JobListItem | null =>
    item ?? currentMenuItem()

  const openJobWorkflow = async (item?: JobListItem | null) => {
    const target = resolveItem(item)
    if (!target) return
    const data = await getJobWorkflow(target.id)
    if (!data) return
    const filename = `Job ${target.id}.json`
    const temp = workflowStore.createTemporary(filename, data)
    await workflowService.openWorkflow(temp)
  }

  const copyJobId = async (item?: JobListItem | null) => {
    const target = resolveItem(item)
    if (!target) return
    await copyToClipboard(target.id)
  }

  const cancelJob = async (item?: JobListItem | null) => {
    const target = resolveItem(item)
    if (!target) return
    if (target.state === 'running' || target.state === 'initialization') {
      if (isCloud) {
        await api.deleteItem('queue', target.id)
      } else {
        await api.interrupt(target.id)
      }
    } else if (target.state === 'pending') {
      await api.deleteItem('queue', target.id)
    }
    executionStore.clearInitializationByJobId(target.id)
    await queueStore.update()
  }

  const copyErrorMessage = async (item?: JobListItem | null) => {
    const target = resolveItem(item)
    const message = target?.taskRef?.errorMessage
    if (message) await copyToClipboard(message)
  }

  const reportError = (item?: JobListItem | null) => {
    const target = resolveItem(item)
    if (!target) return

    // Use execution_error from list response if available
    const executionError = target.taskRef?.executionError

    if (executionError) {
      useDialogService().showExecutionErrorDialog(executionError)
      return
    }

    // Fall back to simple error dialog
    const message = target.taskRef?.errorMessage
    if (message) {
      useDialogService().showErrorDialog(new Error(message), {
        reportType: 'queueJobError'
      })
    }
  }

  // This is very magical only because it matches the respective backend implementation
  // There is or will be a better way to do this
  const addOutputLoaderNode = async () => {
    const item = currentMenuItem()
    if (!item) return
    const result: ResultItemImpl | undefined = item.taskRef?.previewOutput
    if (!result) return

    let nodeType: 'LoadImage' | 'LoadVideo' | 'LoadAudio' | null = null
    let widgetName: 'image' | 'file' | 'audio' | null = null
    if (result.isImage) {
      nodeType = 'LoadImage'
      widgetName = 'image'
    } else if (result.isVideo) {
      nodeType = 'LoadVideo'
      widgetName = 'file'
    } else if (result.isAudio) {
      nodeType = 'LoadAudio'
      widgetName = 'audio'
    }
    if (!nodeType || !widgetName) return

    const nodeDef = nodeDefStore.nodeDefsByName[nodeType]
    if (!nodeDef) return
    const node = litegraphService.addNodeOnGraph(nodeDef, {
      pos: litegraphService.getCanvasCenter()
    })

    if (!node) return

    const isResultItemType = (v: string | undefined): v is ResultItemType =>
      v === 'input' || v === 'output' || v === 'temp'

    const apiItem: ResultItem = {
      filename: result.filename,
      subfolder: result.subfolder,
      type: isResultItemType(result.type) ? result.type : undefined
    }

    const annotated = createAnnotatedPath(apiItem, {
      rootFolder: apiItem.type
    })
    const widget = node.widgets?.find((w) => w.name === widgetName)
    if (widget) {
      widget.value = annotated
      widget.callback?.(annotated)
    }
    node.graph?.setDirtyCanvas(true, true)
  }

  /**
   * Trigger a download of the job's previewable output asset.
   */
  const downloadPreviewAsset = () => {
    const item = currentMenuItem()
    if (!item) return
    const result: ResultItemImpl | undefined = item.taskRef?.previewOutput
    if (!result) return
    downloadFile(result.url)
  }

  /**
   * Export the workflow JSON attached to the job.
   */
  const exportJobWorkflow = async () => {
    const item = currentMenuItem()
    if (!item) return
    const data = await getJobWorkflow(item.id)
    if (!data) return

    const settingStore = useSettingStore()
    let filename = `Job ${item.id}.json`

    if (settingStore.get('Comfy.PromptFilename')) {
      const input = await useDialogService().prompt({
        title: t('workflowService.exportWorkflow'),
        message: t('workflowService.enterFilenamePrompt'),
        defaultValue: filename
      })
      if (!input) return
      filename = appendJsonExt(input)
    }

    const json = JSON.stringify(data, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    downloadBlob(filename, blob)
  }

  const deleteJobAsset = async () => {
    const item = currentMenuItem()
    if (!item) return
    const task = item.taskRef as TaskItemImpl | undefined
    const preview = task?.previewOutput
    if (!task || !preview) return

    const asset = mapTaskOutputToAssetItem(task, preview)
    const confirmed = await mediaAssetActions.deleteAssets(asset)
    if (confirmed) {
      await queueStore.update()
    }
  }

  const removeFailedJob = async (task?: TaskItemImpl | null) => {
    const target =
      task ?? (currentMenuItem()?.taskRef as TaskItemImpl | undefined)
    if (!target) return
    await queueStore.delete(target)
  }

  const jobMenuOpenWorkflowLabel = computed(() =>
    st('queue.jobMenu.openAsWorkflowNewTab', 'Open as workflow in new tab')
  )
  const jobMenuOpenWorkflowFailedLabel = computed(() =>
    st('queue.jobMenu.openWorkflowNewTab', 'Open workflow in new tab')
  )
  const jobMenuCopyJobIdLabel = computed(() =>
    st('queue.jobMenu.copyJobId', 'Copy job ID')
  )
  const jobMenuCancelLabel = computed(() =>
    st('queue.jobMenu.cancelJob', 'Cancel job')
  )

  const jobMenuEntries = computed<MenuEntry[]>(() => {
    const item = currentMenuItem()
    const state = item?.state
    if (!state) return []
    const hasPreviewAsset = !!item?.taskRef?.previewOutput
    if (state === 'completed') {
      return [
        {
          key: 'inspect-asset',
          label: st('queue.jobMenu.inspectAsset', 'Inspect asset'),
          icon: 'icon-[lucide--zoom-in]',
          disabled: !hasPreviewAsset || !onInspectAsset,
          onClick: onInspectAsset
            ? () => {
                const item = currentMenuItem()
                if (item) onInspectAsset(item)
              }
            : undefined
        },
        {
          key: 'add-to-current',
          label: st(
            'queue.jobMenu.addToCurrentWorkflow',
            'Add to current workflow'
          ),
          icon: 'icon-[comfy--node]',
          disabled: !hasPreviewAsset,
          onClick: addOutputLoaderNode
        },
        {
          key: 'download',
          label: st('queue.jobMenu.download', 'Download'),
          icon: 'icon-[lucide--download]',
          disabled: !hasPreviewAsset,
          onClick: downloadPreviewAsset
        },
        { kind: 'divider', key: 'd1' },
        {
          key: 'open-workflow',
          label: jobMenuOpenWorkflowLabel.value,
          icon: 'icon-[comfy--workflow]',
          onClick: openJobWorkflow
        },
        {
          key: 'export-workflow',
          label: st('queue.jobMenu.exportWorkflow', 'Export workflow'),
          icon: 'icon-[comfy--file-output]',
          onClick: exportJobWorkflow
        },
        { kind: 'divider', key: 'd2' },
        {
          key: 'copy-id',
          label: jobMenuCopyJobIdLabel.value,
          icon: 'icon-[lucide--copy]',
          onClick: copyJobId
        },
        { kind: 'divider', key: 'd3' },
        ...(hasPreviewAsset
          ? [
              {
                key: 'delete',
                label: st('queue.jobMenu.deleteAsset', 'Delete asset'),
                icon: 'icon-[lucide--trash-2]',
                onClick: deleteJobAsset
              }
            ]
          : [])
      ]
    }
    if (state === 'failed') {
      return [
        {
          key: 'open-workflow',
          label: jobMenuOpenWorkflowFailedLabel.value,
          icon: 'icon-[comfy--workflow]',
          onClick: openJobWorkflow
        },
        { kind: 'divider', key: 'd1' },
        {
          key: 'copy-id',
          label: jobMenuCopyJobIdLabel.value,
          icon: 'icon-[lucide--copy]',
          onClick: copyJobId
        },
        {
          key: 'copy-error',
          label: st('queue.jobMenu.copyErrorMessage', 'Copy error message'),
          icon: 'icon-[lucide--copy]',
          onClick: copyErrorMessage
        },
        {
          key: 'report-error',
          label: st('queue.jobMenu.reportError', 'Report error'),
          icon: 'icon-[lucide--message-circle-warning]',
          onClick: reportError
        },
        { kind: 'divider', key: 'd2' },
        {
          key: 'delete',
          label: st('queue.jobMenu.removeJob', 'Remove job'),
          icon: 'icon-[lucide--circle-minus]',
          onClick: removeFailedJob
        }
      ]
    }
    return [
      {
        key: 'open-workflow',
        label: jobMenuOpenWorkflowLabel.value,
        icon: 'icon-[comfy--workflow]',
        onClick: openJobWorkflow
      },
      { kind: 'divider', key: 'd1' },
      {
        key: 'copy-id',
        label: jobMenuCopyJobIdLabel.value,
        icon: 'icon-[lucide--copy]',
        onClick: copyJobId
      },
      { kind: 'divider', key: 'd2' },
      {
        key: 'cancel-job',
        label: jobMenuCancelLabel.value,
        icon: 'icon-[lucide--x]',
        onClick: cancelJob
      }
    ]
  })

  return {
    jobMenuEntries,
    openJobWorkflow,
    copyJobId,
    cancelJob,
    removeFailedJob
  }
}
