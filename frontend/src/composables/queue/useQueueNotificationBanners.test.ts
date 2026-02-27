import { mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick, reactive } from 'vue'

import { useQueueNotificationBanners } from '@/composables/queue/useQueueNotificationBanners'
import { useExecutionStore } from '@/stores/executionStore'
import { useQueueStore } from '@/stores/queueStore'

const mockApi = vi.hoisted(() => new EventTarget())

vi.mock('@/scripts/api', () => ({
  api: mockApi
}))

type MockTask = {
  displayStatus: 'Completed' | 'Failed' | 'Cancelled' | 'Running' | 'Pending'
  executionEndTimestamp?: number
  previewOutput?: {
    isImage: boolean
    urlWithTimestamp: string
  }
}

vi.mock('@/stores/queueStore', () => {
  const state = reactive({
    pendingTasks: [] as MockTask[],
    runningTasks: [] as MockTask[],
    historyTasks: [] as MockTask[]
  })

  return {
    useQueueStore: () => state
  }
})

vi.mock('@/stores/executionStore', () => {
  const state = reactive({
    isIdle: true
  })

  return {
    useExecutionStore: () => state
  }
})

const mountComposable = () => {
  let composable: ReturnType<typeof useQueueNotificationBanners>
  const wrapper = mount({
    template: '<div />',
    setup() {
      composable = useQueueNotificationBanners()
      return {}
    }
  })
  return { wrapper, composable: composable! }
}

describe(useQueueNotificationBanners, () => {
  const queueStore = () =>
    useQueueStore() as {
      pendingTasks: MockTask[]
      runningTasks: MockTask[]
      historyTasks: MockTask[]
    }
  const executionStore = () => useExecutionStore() as { isIdle: boolean }

  const resetState = () => {
    queueStore().pendingTasks = []
    queueStore().runningTasks = []
    queueStore().historyTasks = []
    executionStore().isIdle = true
  }

  const createTask = (
    options: {
      state?: MockTask['displayStatus']
      ts?: number
      previewUrl?: string
      isImage?: boolean
    } = {}
  ): MockTask => {
    const {
      state = 'Completed',
      ts = Date.now(),
      previewUrl,
      isImage = true
    } = options

    const task: MockTask = {
      displayStatus: state,
      executionEndTimestamp: ts
    }

    if (previewUrl) {
      task.previewOutput = {
        isImage,
        urlWithTimestamp: previewUrl
      }
    }

    return task
  }

  const runBatch = async (options: {
    start: number
    finish: number
    tasks: MockTask[]
  }) => {
    const { start, finish, tasks } = options

    vi.setSystemTime(start)
    executionStore().isIdle = false
    await nextTick()

    vi.setSystemTime(finish)
    queueStore().historyTasks = tasks
    executionStore().isIdle = true
    await nextTick()
  }

  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(0)
    resetState()
  })

  afterEach(() => {
    vi.runOnlyPendingTimers()
    vi.useRealTimers()
    resetState()
  })

  it('shows queued notifications from promptQueued events', async () => {
    const { wrapper, composable } = mountComposable()

    try {
      mockApi.dispatchEvent(
        new CustomEvent('promptQueued', { detail: { batchCount: 4 } })
      )
      await nextTick()

      expect(composable.currentNotification.value).toEqual({
        type: 'queued',
        count: 4
      })

      await vi.advanceTimersByTimeAsync(4000)
      await nextTick()
      expect(composable.currentNotification.value).toBeNull()
    } finally {
      wrapper.unmount()
    }
  })

  it('shows queued pending then queued confirmation', async () => {
    const { wrapper, composable } = mountComposable()

    try {
      mockApi.dispatchEvent(
        new CustomEvent('promptQueueing', {
          detail: { requestId: 1, batchCount: 2 }
        })
      )
      await nextTick()

      expect(composable.currentNotification.value).toEqual({
        type: 'queuedPending',
        count: 2,
        requestId: 1
      })

      mockApi.dispatchEvent(
        new CustomEvent('promptQueued', {
          detail: { requestId: 1, batchCount: 2 }
        })
      )
      await nextTick()

      expect(composable.currentNotification.value).toEqual({
        type: 'queued',
        count: 2,
        requestId: 1
      })
    } finally {
      wrapper.unmount()
    }
  })

  it('falls back to 1 when queued batch count is invalid', async () => {
    const { wrapper, composable } = mountComposable()

    try {
      mockApi.dispatchEvent(
        new CustomEvent('promptQueued', { detail: { batchCount: 0 } })
      )
      await nextTick()

      expect(composable.currentNotification.value).toEqual({
        type: 'queued',
        count: 1
      })
    } finally {
      wrapper.unmount()
    }
  })

  it('shows a completed notification from a finished batch', async () => {
    const { wrapper, composable } = mountComposable()

    try {
      await runBatch({
        start: 1_000,
        finish: 1_200,
        tasks: [
          createTask({
            ts: 1_050,
            previewUrl: 'https://example.com/preview.png'
          })
        ]
      })

      expect(composable.currentNotification.value).toEqual({
        type: 'completed',
        count: 1,
        thumbnailUrls: ['https://example.com/preview.png']
      })
    } finally {
      wrapper.unmount()
    }
  })

  it('shows one completion notification when history updates after queue becomes idle', async () => {
    const { wrapper, composable } = mountComposable()

    try {
      vi.setSystemTime(4_000)
      executionStore().isIdle = false
      await nextTick()

      vi.setSystemTime(4_100)
      executionStore().isIdle = true
      queueStore().historyTasks = []
      await nextTick()

      expect(composable.currentNotification.value).toBeNull()

      queueStore().historyTasks = [
        createTask({
          ts: 4_050,
          previewUrl: 'https://example.com/race-preview.png'
        })
      ]
      await nextTick()

      expect(composable.currentNotification.value).toEqual({
        type: 'completed',
        count: 1,
        thumbnailUrls: ['https://example.com/race-preview.png']
      })

      await vi.advanceTimersByTimeAsync(4000)
      await nextTick()
      expect(composable.currentNotification.value).toBeNull()

      await vi.advanceTimersByTimeAsync(4000)
      await nextTick()
      expect(composable.currentNotification.value).toBeNull()
    } finally {
      wrapper.unmount()
    }
  })

  it('queues both completed and failed notifications for mixed batches', async () => {
    const { wrapper, composable } = mountComposable()

    try {
      await runBatch({
        start: 2_000,
        finish: 2_200,
        tasks: [
          createTask({
            ts: 2_050,
            previewUrl: 'https://example.com/result.png'
          }),
          createTask({ ts: 2_060 }),
          createTask({ ts: 2_070 }),
          createTask({ state: 'Failed', ts: 2_080 })
        ]
      })

      expect(composable.currentNotification.value).toEqual({
        type: 'completed',
        count: 3,
        thumbnailUrls: ['https://example.com/result.png']
      })

      await vi.advanceTimersByTimeAsync(4000)
      await nextTick()

      expect(composable.currentNotification.value).toEqual({
        type: 'failed',
        count: 1
      })
    } finally {
      wrapper.unmount()
    }
  })

  it('uses up to two completion thumbnails for notification icon previews', async () => {
    const { wrapper, composable } = mountComposable()

    try {
      await runBatch({
        start: 3_000,
        finish: 3_300,
        tasks: [
          createTask({
            ts: 3_050,
            previewUrl: 'https://example.com/preview-1.png'
          }),
          createTask({
            ts: 3_060,
            previewUrl: 'https://example.com/preview-2.png'
          }),
          createTask({
            ts: 3_070,
            previewUrl: 'https://example.com/preview-3.png'
          }),
          createTask({
            ts: 3_080,
            previewUrl: 'https://example.com/preview-4.png'
          })
        ]
      })

      expect(composable.currentNotification.value).toEqual({
        type: 'completed',
        count: 4,
        thumbnailUrls: [
          'https://example.com/preview-1.png',
          'https://example.com/preview-2.png'
        ]
      })
    } finally {
      wrapper.unmount()
    }
  })
})
