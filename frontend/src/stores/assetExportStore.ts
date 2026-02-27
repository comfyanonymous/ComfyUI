import { useIntervalFn } from '@vueuse/core'
import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'

import { assetService } from '@/platform/assets/services/assetService'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { taskService } from '@/platform/tasks/services/taskService'
import type { AssetExportWsMessage } from '@/schemas/apiSchema'
import { api } from '@/scripts/api'
import { t } from '@/i18n'

export interface AssetExport {
  taskId: string
  exportName: string
  assetsTotal: number
  assetsAttempted: number
  assetsFailed: number
  bytesTotal: number
  bytesProcessed: number
  progress: number
  status: 'created' | 'running' | 'completed' | 'failed'
  error?: string
  downloadError?: string
  lastUpdate: number
  downloadTriggered: boolean
}

const STALE_THRESHOLD_MS = 10_000
const POLL_INTERVAL_MS = 10_000

export const useAssetExportStore = defineStore('assetExport', () => {
  const exports = ref<Map<string, AssetExport>>(new Map())

  const exportList = computed(() => Array.from(exports.value.values()))
  const activeExports = computed(() =>
    exportList.value.filter(
      (e) => e.status === 'created' || e.status === 'running'
    )
  )
  const finishedExports = computed(() =>
    exportList.value.filter(
      (e) => e.status === 'completed' || e.status === 'failed'
    )
  )
  const hasActiveExports = computed(() => activeExports.value.length > 0)
  const hasExports = computed(() => exports.value.size > 0)

  function trackExport(taskId: string) {
    if (exports.value.has(taskId)) return

    exports.value.set(taskId, {
      taskId,
      exportName: '',
      assetsTotal: 0,
      assetsAttempted: 0,
      assetsFailed: 0,
      bytesTotal: 0,
      bytesProcessed: 0,
      progress: 0,
      status: 'created',
      lastUpdate: Date.now(),
      downloadTriggered: false
    })
  }

  async function triggerDownload(exp: AssetExport, force = false) {
    if (!force && (exp.downloadTriggered || !exp.exportName)) return
    exp.downloadTriggered = true

    try {
      exp.downloadError = undefined
      const { url } = await assetService.getExportDownloadUrl(exp.exportName)
      const link = document.createElement('a')
      link.href = url
      link.download = exp.exportName
      link.style.display = 'none'
      link.target = '_blank'
      link.rel = 'noopener noreferrer'
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      exp.downloadError = message
      exp.downloadTriggered = false

      useToastStore().add({
        severity: 'error',
        summary: t('exportToast.downloadFailed', {
          name: exp.exportName
        }),
        detail: message
      })
    }
  }

  function handleAssetExport(data: AssetExportWsMessage) {
    const existing = exports.value.get(data.task_id)

    if (
      (existing?.status === 'completed' || existing?.status === 'failed') &&
      existing?.downloadTriggered
    ) {
      return
    }

    const exp: AssetExport = {
      taskId: data.task_id,
      exportName: data.export_name ?? existing?.exportName ?? '',
      assetsTotal: data.assets_total,
      assetsAttempted: data.assets_attempted,
      assetsFailed: data.assets_failed,
      bytesTotal: data.bytes_total,
      bytesProcessed: data.bytes_processed,
      progress: data.progress,
      status: data.status,
      error: data.error,
      lastUpdate: Date.now(),
      downloadTriggered: existing?.downloadTriggered ?? false
    }

    exports.value.set(data.task_id, exp)

    if (data.status === 'completed') {
      void triggerDownload(exp)
    }
  }

  async function pollStaleExports() {
    const now = Date.now()
    const staleExports = activeExports.value.filter(
      (e) => now - e.lastUpdate >= STALE_THRESHOLD_MS
    )

    if (staleExports.length === 0) return

    async function pollSingleExport(exp: AssetExport) {
      try {
        const task = await taskService.getTask(exp.taskId)

        if (task.status === 'completed' || task.status === 'failed') {
          const result = task.result as Record<string, unknown> | undefined
          handleAssetExport({
            task_id: exp.taskId,
            export_name: (result?.export_name as string) ?? exp.exportName,
            assets_total: (result?.assets_total as number) ?? exp.assetsTotal,
            assets_attempted:
              (result?.assets_attempted as number) ?? exp.assetsAttempted,
            assets_failed:
              (result?.assets_failed as number) ?? exp.assetsFailed,
            bytes_total: exp.bytesTotal,
            bytes_processed: exp.bytesTotal,
            progress: task.status === 'completed' ? 1 : exp.progress,
            status: task.status as 'completed' | 'failed',
            error: task.error_message ?? (result?.error as string)
          })
        }
      } catch {
        // Task not ready or not found
      }
    }

    await Promise.all(staleExports.map(pollSingleExport))
  }

  const { pause, resume } = useIntervalFn(
    () => void pollStaleExports(),
    POLL_INTERVAL_MS,
    { immediate: false }
  )

  watch(
    hasActiveExports,
    (hasActive) => {
      if (hasActive) resume()
      else pause()
    },
    { immediate: true }
  )

  api.addEventListener('asset_export', (e) => handleAssetExport(e.detail))

  function clearFinishedExports() {
    for (const exp of finishedExports.value) {
      exports.value.delete(exp.taskId)
    }
  }

  return {
    activeExports,
    finishedExports,
    hasActiveExports,
    hasExports,
    exportList,
    trackExport,
    triggerDownload,
    clearFinishedExports
  }
})
