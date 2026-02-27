import { computed, nextTick, onUnmounted, ref, watch } from 'vue'

import { api } from '@/scripts/api'
import type {
  PromptQueuedEventPayload,
  PromptQueueingEventPayload
} from '@/scripts/api'
import { useExecutionStore } from '@/stores/executionStore'
import { useQueueStore } from '@/stores/queueStore'
import { jobStateFromTask } from '@/utils/queueUtil'

const BANNER_DISMISS_DELAY_MS = 4000
const MAX_COMPLETION_THUMBNAILS = 2

type QueueQueuedNotificationType = 'queuedPending' | 'queued'

type QueueQueuedNotification = {
  type: QueueQueuedNotificationType
  count: number
  requestId?: number
}

type QueueCompletedNotification = {
  type: 'completed'
  count: number
  thumbnailUrls?: string[]
}

type QueueFailedNotification = {
  type: 'failed'
  count: number
}

export type QueueNotificationBanner =
  | QueueQueuedNotification
  | QueueCompletedNotification
  | QueueFailedNotification

const sanitizeCount = (value: number | undefined) => {
  if (!(typeof value === 'number' && value > 0)) {
    return 1
  }
  return Math.floor(value)
}

export const useQueueNotificationBanners = () => {
  const queueStore = useQueueStore()
  const executionStore = useExecutionStore()

  const pendingNotifications = ref<QueueNotificationBanner[]>([])
  const activeNotification = ref<QueueNotificationBanner | null>(null)
  const dismissTimer = ref<number | null>(null)
  const lastActiveStartTs = ref<number | null>(null)
  let stopIdleHistoryWatch: (() => void) | null = null
  let idleCompletionScheduleToken = 0
  const isQueueActive = computed(
    () =>
      queueStore.pendingTasks.length > 0 ||
      queueStore.runningTasks.length > 0 ||
      !executionStore.isIdle
  )

  const clearIdleCompletionHooks = () => {
    idleCompletionScheduleToken++
    if (!stopIdleHistoryWatch) {
      return
    }
    stopIdleHistoryWatch()
    stopIdleHistoryWatch = null
  }

  const clearDismissTimer = () => {
    if (dismissTimer.value === null) {
      return
    }
    clearTimeout(dismissTimer.value)
    dismissTimer.value = null
  }

  const dismissActiveNotification = () => {
    activeNotification.value = null
    dismissTimer.value = null
    showNextNotification()
  }

  const showNextNotification = () => {
    if (activeNotification.value !== null) {
      return
    }
    const [nextNotification, ...rest] = pendingNotifications.value
    pendingNotifications.value = rest
    if (!nextNotification) {
      return
    }

    activeNotification.value = nextNotification
    clearDismissTimer()
    dismissTimer.value = window.setTimeout(
      dismissActiveNotification,
      BANNER_DISMISS_DELAY_MS
    )
  }

  const queueNotification = (notification: QueueNotificationBanner) => {
    pendingNotifications.value = [...pendingNotifications.value, notification]
    showNextNotification()
  }

  const toQueueLifecycleNotification = (
    type: QueueQueuedNotificationType,
    count: number,
    requestId?: number
  ): QueueQueuedNotification => {
    if (requestId === undefined) {
      return {
        type,
        count
      }
    }
    return {
      type,
      count,
      requestId
    }
  }

  const toCompletedNotification = (
    count: number,
    thumbnailUrls: string[]
  ): QueueCompletedNotification => ({
    type: 'completed',
    count,
    thumbnailUrls: thumbnailUrls.slice(0, MAX_COMPLETION_THUMBNAILS)
  })

  const toFailedNotification = (count: number): QueueFailedNotification => ({
    type: 'failed',
    count
  })

  const convertQueuedPendingToQueued = (
    requestId: number | undefined,
    count: number
  ) => {
    if (
      activeNotification.value?.type === 'queuedPending' &&
      (requestId === undefined ||
        activeNotification.value.requestId === requestId)
    ) {
      activeNotification.value = toQueueLifecycleNotification(
        'queued',
        count,
        requestId
      )
      return true
    }

    const pendingIndex = pendingNotifications.value.findIndex(
      (notification) =>
        notification.type === 'queuedPending' &&
        (requestId === undefined || notification.requestId === requestId)
    )

    if (pendingIndex === -1) {
      return false
    }

    const queuedPendingNotification = pendingNotifications.value[pendingIndex]
    if (
      queuedPendingNotification === undefined ||
      queuedPendingNotification.type !== 'queuedPending'
    ) {
      return false
    }

    pendingNotifications.value = [
      ...pendingNotifications.value.slice(0, pendingIndex),
      toQueueLifecycleNotification(
        'queued',
        count,
        queuedPendingNotification.requestId
      ),
      ...pendingNotifications.value.slice(pendingIndex + 1)
    ]

    return true
  }

  const handlePromptQueueing = (
    event: CustomEvent<PromptQueueingEventPayload>
  ) => {
    const payload = event.detail
    const count = sanitizeCount(payload?.batchCount)
    queueNotification(
      toQueueLifecycleNotification('queuedPending', count, payload?.requestId)
    )
  }

  const handlePromptQueued = (event: CustomEvent<PromptQueuedEventPayload>) => {
    const payload = event.detail
    const count = sanitizeCount(payload?.batchCount)
    const handled = convertQueuedPendingToQueued(payload?.requestId, count)
    if (!handled) {
      queueNotification(
        toQueueLifecycleNotification('queued', count, payload?.requestId)
      )
    }
  }

  api.addEventListener('promptQueueing', handlePromptQueueing)
  api.addEventListener('promptQueued', handlePromptQueued)

  const queueCompletionBatchNotifications = () => {
    const startTs = lastActiveStartTs.value ?? 0
    const finishedTasks = queueStore.historyTasks.filter((task) => {
      const ts = task.executionEndTimestamp
      return typeof ts === 'number' && ts >= startTs
    })

    if (!finishedTasks.length) {
      return false
    }

    let completedCount = 0
    let failedCount = 0
    const imagePreviews: string[] = []

    for (const task of finishedTasks) {
      const state = jobStateFromTask(task, false)
      if (state === 'completed') {
        completedCount++
        const preview = task.previewOutput
        if (preview?.isImage) {
          imagePreviews.push(preview.urlWithTimestamp)
        }
      } else if (state === 'failed') {
        failedCount++
      }
    }

    if (completedCount > 0) {
      queueNotification(toCompletedNotification(completedCount, imagePreviews))
    }

    if (failedCount > 0) {
      queueNotification(toFailedNotification(failedCount))
    }

    return completedCount > 0 || failedCount > 0
  }

  const scheduleIdleCompletionBatchNotifications = () => {
    clearIdleCompletionHooks()
    const scheduleToken = idleCompletionScheduleToken
    const startTsSnapshot = lastActiveStartTs.value

    const isStillSameIdleWindow = () =>
      scheduleToken === idleCompletionScheduleToken &&
      !isQueueActive.value &&
      lastActiveStartTs.value === startTsSnapshot

    stopIdleHistoryWatch = watch(
      () => queueStore.historyTasks,
      () => {
        if (!isStillSameIdleWindow()) {
          clearIdleCompletionHooks()
          return
        }
        queueCompletionBatchNotifications()
        clearIdleCompletionHooks()
      }
    )

    void nextTick(() => {
      if (!isStillSameIdleWindow()) {
        clearIdleCompletionHooks()
        return
      }

      const hasShownNotifications = queueCompletionBatchNotifications()
      if (hasShownNotifications) {
        clearIdleCompletionHooks()
      }
    })
  }

  watch(
    isQueueActive,
    (active, prev) => {
      if (!prev && active) {
        clearIdleCompletionHooks()
        lastActiveStartTs.value = Date.now()
        return
      }
      if (prev && !active) {
        scheduleIdleCompletionBatchNotifications()
      }
    },
    { immediate: true }
  )

  onUnmounted(() => {
    api.removeEventListener('promptQueueing', handlePromptQueueing)
    api.removeEventListener('promptQueued', handlePromptQueued)
    clearIdleCompletionHooks()
    clearDismissTimer()
    pendingNotifications.value = []
    activeNotification.value = null
    lastActiveStartTs.value = null
  })

  const currentNotification = computed(() => activeNotification.value)

  return {
    currentNotification
  }
}
