import { mount } from '@vue/test-utils'
import type { VueWrapper } from '@vue/test-utils'
import { nextTick, reactive } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useQueuePolling } from '@/platform/remote/comfyui/useQueuePolling'

vi.mock('@/stores/queueStore', () => {
  const state = reactive({
    activeJobsCount: 0,
    isLoading: false,
    update: vi.fn()
  })

  return {
    useQueueStore: () => state
  }
})

// Re-import to get the mock instance for assertions
import { useQueueStore } from '@/stores/queueStore'

function mountUseQueuePolling() {
  return mount({
    template: '<div />',
    setup() {
      useQueuePolling()
      return {}
    }
  })
}

describe('useQueuePolling', () => {
  let wrapper: VueWrapper
  const store = useQueueStore() as Partial<
    ReturnType<typeof useQueueStore>
  > as {
    activeJobsCount: number
    isLoading: boolean
    update: ReturnType<typeof vi.fn>
  }

  beforeEach(() => {
    vi.useFakeTimers()
    store.activeJobsCount = 0
    store.isLoading = false
    store.update.mockReset()
  })

  afterEach(() => {
    wrapper?.unmount()
    vi.useRealTimers()
  })

  it('does not call update on creation', () => {
    wrapper = mountUseQueuePolling()
    expect(store.update).not.toHaveBeenCalled()
  })

  it('polls when activeJobsCount is exactly 1', async () => {
    wrapper = mountUseQueuePolling()

    store.activeJobsCount = 1
    await vi.advanceTimersByTimeAsync(8_000)

    expect(store.update).toHaveBeenCalledOnce()
  })

  it('does not poll when activeJobsCount > 1', async () => {
    wrapper = mountUseQueuePolling()

    store.activeJobsCount = 2
    await vi.advanceTimersByTimeAsync(16_000)

    expect(store.update).not.toHaveBeenCalled()
  })

  it('stops polling when activeJobsCount drops to 0', async () => {
    store.activeJobsCount = 1
    wrapper = mountUseQueuePolling()

    store.activeJobsCount = 0
    await vi.advanceTimersByTimeAsync(16_000)

    expect(store.update).not.toHaveBeenCalled()
  })

  it('resets timer when loading completes', async () => {
    store.activeJobsCount = 1
    wrapper = mountUseQueuePolling()

    // Advance 5s toward the 8s timeout
    await vi.advanceTimersByTimeAsync(5_000)
    expect(store.update).not.toHaveBeenCalled()

    // Simulate an external update (e.g. WebSocket-triggered) completing
    store.isLoading = true
    await nextTick()
    store.isLoading = false
    await nextTick()

    // Timer should now be reset — should not fire 3s later (original 8s mark)
    await vi.advanceTimersByTimeAsync(3_000)
    expect(store.update).not.toHaveBeenCalled()

    // Should fire 8s after the reset (5s more)
    await vi.advanceTimersByTimeAsync(5_000)
    expect(store.update).toHaveBeenCalledOnce()
  })

  it('applies exponential backoff on successive polls', async () => {
    store.activeJobsCount = 1
    wrapper = mountUseQueuePolling()

    // First poll at 8s
    await vi.advanceTimersByTimeAsync(8_000)
    expect(store.update).toHaveBeenCalledTimes(1)

    // Simulate update completing to reschedule
    store.isLoading = true
    await nextTick()
    store.isLoading = false
    await nextTick()

    // Second poll at 12s (8 * 1.5)
    await vi.advanceTimersByTimeAsync(11_000)
    expect(store.update).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(1_000)
    expect(store.update).toHaveBeenCalledTimes(2)
  })

  it('skips poll when an update is already in-flight', async () => {
    store.activeJobsCount = 1
    wrapper = mountUseQueuePolling()

    // Simulate an external update starting before the timer fires
    store.isLoading = true

    await vi.advanceTimersByTimeAsync(8_000)
    expect(store.update).not.toHaveBeenCalled()

    // Once the in-flight update completes, polling resumes
    store.isLoading = false

    await vi.advanceTimersByTimeAsync(8_000)
    expect(store.update).toHaveBeenCalledOnce()
  })

  it('resets backoff when activeJobsCount changes', async () => {
    store.activeJobsCount = 1
    wrapper = mountUseQueuePolling()

    // First poll at 8s (backs off delay to 12s)
    await vi.advanceTimersByTimeAsync(8_000)
    expect(store.update).toHaveBeenCalledTimes(1)

    // Simulate update completing
    store.isLoading = true
    await nextTick()
    store.isLoading = false
    await nextTick()

    // Count changes — backoff should reset to 8s
    store.activeJobsCount = 0
    await nextTick()
    store.activeJobsCount = 1
    await nextTick()

    store.update.mockClear()
    await vi.advanceTimersByTimeAsync(8_000)
    expect(store.update).toHaveBeenCalledOnce()
  })
})
