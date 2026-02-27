import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { computed, effectScope, ref } from 'vue'
import type { EffectScope } from 'vue'

import type { CloudSubscriptionStatusResponse } from '@/platform/cloud/subscription/composables/useSubscription'
import { useSubscriptionCancellationWatcher } from '@/platform/cloud/subscription/composables/useSubscriptionCancellationWatcher'
import type { TelemetryDispatcher } from '@/platform/telemetry/types'

describe('useSubscriptionCancellationWatcher', () => {
  const trackMonthlySubscriptionCancelled = vi.fn()
  const telemetryMock: Pick<
    TelemetryDispatcher,
    'trackMonthlySubscriptionCancelled'
  > = {
    trackMonthlySubscriptionCancelled
  }

  const baseStatus: CloudSubscriptionStatusResponse = {
    is_active: true,
    subscription_id: 'sub_123',
    renewal_date: '2025-11-16'
  }

  const subscriptionStatus = ref<CloudSubscriptionStatusResponse | null>(
    baseStatus
  )
  const isActive = ref(true)
  const isActiveSubscription = computed(() => isActive.value)

  let shouldWatch = true
  const shouldWatchCancellation = () => shouldWatch

  const activeScopes: EffectScope[] = []

  const initWatcher = (
    options: Parameters<typeof useSubscriptionCancellationWatcher>[0]
  ): ReturnType<typeof useSubscriptionCancellationWatcher> => {
    const scope = effectScope()
    let result: ReturnType<typeof useSubscriptionCancellationWatcher> | null =
      null
    scope.run(() => {
      result = useSubscriptionCancellationWatcher(options)
    })
    if (!result) {
      throw new Error('Failed to initialize cancellation watcher')
    }
    activeScopes.push(scope)
    return result
  }

  beforeEach(() => {
    vi.useFakeTimers()
    trackMonthlySubscriptionCancelled.mockReset()
    subscriptionStatus.value = { ...baseStatus }
    isActive.value = true
    shouldWatch = true
  })

  afterEach(() => {
    activeScopes.forEach((scope) => scope.stop())
    activeScopes.length = 0
    vi.useRealTimers()
  })

  it('polls with exponential backoff and fires telemetry once cancellation detected', async () => {
    const fetchStatus = vi.fn(async () => {
      if (fetchStatus.mock.calls.length === 2) {
        isActive.value = false
        subscriptionStatus.value = {
          is_active: false,
          subscription_id: 'sub_cancelled',
          renewal_date: '2025-11-16',
          end_date: '2025-12-01'
        }
      }
    })

    const { startCancellationWatcher } = initWatcher({
      fetchStatus,
      isActiveSubscription,
      subscriptionStatus,
      telemetry: telemetryMock,
      shouldWatchCancellation
    })

    startCancellationWatcher()

    await vi.advanceTimersByTimeAsync(5000)
    expect(fetchStatus).toHaveBeenCalledTimes(1)

    await vi.advanceTimersByTimeAsync(15000)
    expect(fetchStatus).toHaveBeenCalledTimes(2)
    expect(
      telemetryMock.trackMonthlySubscriptionCancelled
    ).toHaveBeenCalledTimes(1)
  })

  it('triggers a check immediately when window regains focus', async () => {
    const fetchStatus = vi.fn(async () => {
      isActive.value = false
      subscriptionStatus.value = {
        ...baseStatus,
        is_active: false,
        end_date: '2025-12-01'
      }
    })

    const { startCancellationWatcher } = initWatcher({
      fetchStatus,
      isActiveSubscription,
      subscriptionStatus,
      telemetry: telemetryMock,
      shouldWatchCancellation
    })

    startCancellationWatcher()

    window.dispatchEvent(new Event('focus'))
    await Promise.resolve()

    expect(fetchStatus).toHaveBeenCalledTimes(1)
    expect(
      telemetryMock.trackMonthlySubscriptionCancelled
    ).toHaveBeenCalledTimes(1)
  })

  it('stops after max attempts when subscription stays active', async () => {
    const fetchStatus = vi.fn(async () => {})

    const { startCancellationWatcher } = initWatcher({
      fetchStatus,
      isActiveSubscription,
      subscriptionStatus,
      telemetry: telemetryMock,
      shouldWatchCancellation
    })

    startCancellationWatcher()

    const delays = [5000, 15000, 45000, 135000]
    for (const delay of delays) {
      await vi.advanceTimersByTimeAsync(delay)
    }

    expect(fetchStatus).toHaveBeenCalledTimes(4)
    expect(trackMonthlySubscriptionCancelled).not.toHaveBeenCalled()

    await vi.advanceTimersByTimeAsync(200000)
    expect(fetchStatus).toHaveBeenCalledTimes(4)
  })

  it('does not start watcher when guard fails or subscription inactive', async () => {
    const fetchStatus = vi.fn()

    const { startCancellationWatcher } = initWatcher({
      fetchStatus,
      isActiveSubscription,
      subscriptionStatus,
      telemetry: telemetryMock,
      shouldWatchCancellation
    })

    shouldWatch = false
    startCancellationWatcher()
    await vi.advanceTimersByTimeAsync(60000)
    expect(fetchStatus).not.toHaveBeenCalled()

    shouldWatch = true
    isActive.value = false
    subscriptionStatus.value = {
      ...baseStatus,
      is_active: false
    }
    startCancellationWatcher()
    await vi.advanceTimersByTimeAsync(60000)
    expect(fetchStatus).not.toHaveBeenCalled()
  })
})
