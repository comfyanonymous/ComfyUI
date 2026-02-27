import { useIntervalFn } from '@vueuse/core'
import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'

import { taskService } from '@/platform/tasks/services/taskService'
import type { AssetDownloadWsMessage } from '@/schemas/apiSchema'
import { api } from '@/scripts/api'

export interface AssetDownload {
  taskId: string
  assetName: string
  bytesTotal: number
  bytesDownloaded: number
  progress: number
  status: 'created' | 'running' | 'completed' | 'failed'
  lastUpdate: number
  assetId?: string
  error?: string
  modelType?: string
  acknowledged?: boolean
}

interface CompletedDownload {
  taskId: string
  modelType: string
  timestamp: number
}
const STALE_THRESHOLD_MS = 10_000
const POLL_INTERVAL_MS = 10_000

function generateDownloadTrackingPlaceholder(
  taskId: string,
  modelType: string,
  assetName: string
): AssetDownload {
  return {
    taskId,
    modelType,
    assetName,
    bytesTotal: 0,
    bytesDownloaded: 0,
    progress: 0,
    status: 'created',
    lastUpdate: Date.now()
  }
}

export const useAssetDownloadStore = defineStore('assetDownload', () => {
  const downloads = ref<Map<string, AssetDownload>>(new Map())
  const lastCompletedDownload = ref<CompletedDownload | null>(null)

  const downloadList = computed(() => Array.from(downloads.value.values()))
  const activeDownloads = computed(() =>
    downloadList.value.filter(
      (d) => d.status === 'created' || d.status === 'running'
    )
  )
  const finishedDownloads = computed(() =>
    downloadList.value.filter(
      (d) => d.status === 'completed' || d.status === 'failed'
    )
  )
  const unacknowledgedDownloads = computed(() =>
    finishedDownloads.value.filter(
      (d) => d.status === 'completed' && d.assetId && !d.acknowledged
    )
  )
  const sessionDownloadCount = computed(
    () => unacknowledgedDownloads.value.length
  )
  const hasActiveDownloads = computed(() => activeDownloads.value.length > 0)
  const hasDownloads = computed(() => downloads.value.size > 0)

  function isDownloadedThisSession(assetId: string): boolean {
    return unacknowledgedDownloads.value.some((d) => d.assetId === assetId)
  }

  function acknowledgeAsset(assetId: string) {
    for (const download of downloads.value.values()) {
      if (download.assetId === assetId) {
        download.acknowledged = true
      }
    }
  }

  function trackDownload(taskId: string, modelType: string, assetName: string) {
    if (downloads.value.has(taskId)) return

    downloads.value.set(
      taskId,
      generateDownloadTrackingPlaceholder(taskId, modelType, assetName)
    )
  }

  function handleAssetDownload(e: CustomEvent<AssetDownloadWsMessage>) {
    const data = e.detail
    const existing = downloads.value.get(data.task_id)

    // Skip if already in terminal state
    if (existing?.status === 'completed' || existing?.status === 'failed') {
      return
    }

    const download: AssetDownload = {
      taskId: data.task_id,
      assetId: data.asset_id,
      assetName: data.asset_name,
      bytesTotal: data.bytes_total,
      bytesDownloaded: data.bytes_downloaded,
      progress: data.progress,
      status: data.status,
      error: data.error,
      lastUpdate: Date.now(),
      modelType: existing?.modelType
    }

    downloads.value.set(data.task_id, download)

    if (data.status === 'completed' && download.modelType) {
      lastCompletedDownload.value = {
        taskId: data.task_id,
        modelType: download.modelType,
        timestamp: Date.now()
      }
    }
  }

  async function pollStaleDownloads() {
    const now = Date.now()
    const staleDownloads = activeDownloads.value.filter(
      (d) => now - d.lastUpdate >= STALE_THRESHOLD_MS
    )

    if (staleDownloads.length === 0) return

    async function pollSingleDownload(download: AssetDownload) {
      try {
        const task = await taskService.getTask(download.taskId)

        if (task.status === 'completed' || task.status === 'failed') {
          const result = task.result
          handleAssetDownload(
            new CustomEvent('asset_download', {
              detail: {
                task_id: download.taskId,
                asset_id: result?.asset_id ?? download.assetId,
                asset_name: result?.filename ?? download.assetName,
                bytes_total: download.bytesTotal,
                bytes_downloaded:
                  result?.bytes_downloaded ?? download.bytesTotal,
                progress: task.status === 'completed' ? 100 : download.progress,
                status: task.status,
                error: task.error_message ?? result?.error
              }
            })
          )
        }
      } catch {
        // Task not ready or not found
      }
    }

    await Promise.all(staleDownloads.map(pollSingleDownload))
  }

  const { pause, resume } = useIntervalFn(
    () => void pollStaleDownloads(),
    POLL_INTERVAL_MS,
    { immediate: false }
  )

  watch(
    hasActiveDownloads,
    (hasActive) => {
      if (hasActive) resume()
      else pause()
    },
    { immediate: true }
  )

  api.addEventListener('asset_download', handleAssetDownload)

  function clearFinishedDownloads() {
    for (const download of finishedDownloads.value) {
      downloads.value.delete(download.taskId)
    }
  }

  return {
    activeDownloads,
    finishedDownloads,
    hasActiveDownloads,
    hasDownloads,
    downloadList,
    lastCompletedDownload,
    sessionDownloadCount,
    trackDownload,
    clearFinishedDownloads,
    isDownloadedThisSession,
    acknowledgeAsset
  }
})
