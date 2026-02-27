import { onScopeDispose, ref } from 'vue'
import type { ComputedRef, Ref } from 'vue'
import { defaultWindow, useEventListener, useTimeoutFn } from '@vueuse/core'

import type { TelemetryDispatcher } from '@/platform/telemetry/types'

import type { CloudSubscriptionStatusResponse } from './useSubscription'

const MAX_CANCELLATION_ATTEMPTS = 4
const CANCELLATION_BASE_DELAY_MS = 5000
const CANCELLATION_BACKOFF_MULTIPLIER = 3 // 5s, 15s, 45s, 135s intervals

type CancellationWatcherOptions = {
  fetchStatus: () => Promise<CloudSubscriptionStatusResponse | null | void>
  isActiveSubscription: ComputedRef<boolean>
  subscriptionStatus: Ref<CloudSubscriptionStatusResponse | null>
  telemetry: Pick<
    TelemetryDispatcher,
    'trackMonthlySubscriptionCancelled'
  > | null
  shouldWatchCancellation: () => boolean
}

export function useSubscriptionCancellationWatcher({
  fetchStatus,
  isActiveSubscription,
  subscriptionStatus,
  telemetry,
  shouldWatchCancellation
}: CancellationWatcherOptions) {
  const watcherActive = ref(false)
  const cancellationAttempts = ref(0)
  const cancellationTracked = ref(false)
  const cancellationCheckInFlight = ref(false)
  const nextDelay = ref(CANCELLATION_BASE_DELAY_MS)
  let detachFocusListener: (() => void) | null = null

  const { start: startTimer, stop: stopTimer } = useTimeoutFn(
    () => {
      void checkForCancellation()
    },
    nextDelay,
    { immediate: false }
  )

  const stopCancellationWatcher = () => {
    watcherActive.value = false
    stopTimer()
    cancellationAttempts.value = 0
    cancellationCheckInFlight.value = false
    if (detachFocusListener) {
      detachFocusListener()
      detachFocusListener = null
    }
  }

  const scheduleNextCancellationCheck = () => {
    if (!watcherActive.value) return

    if (cancellationAttempts.value >= MAX_CANCELLATION_ATTEMPTS) {
      stopCancellationWatcher()
      return
    }

    nextDelay.value =
      CANCELLATION_BASE_DELAY_MS *
      CANCELLATION_BACKOFF_MULTIPLIER ** cancellationAttempts.value
    cancellationAttempts.value += 1
    startTimer()
  }

  const checkForCancellation = async (triggeredFromFocus = false) => {
    if (!watcherActive.value || cancellationCheckInFlight.value) return

    cancellationCheckInFlight.value = true
    try {
      await fetchStatus()

      if (!isActiveSubscription.value) {
        if (!cancellationTracked.value) {
          cancellationTracked.value = true
          try {
            telemetry?.trackMonthlySubscriptionCancelled()
          } catch (telemetryError) {
            console.error(
              '[Subscription] Failed to track cancellation telemetry:',
              telemetryError
            )
          }
        }
        stopCancellationWatcher()
        return
      }

      if (!triggeredFromFocus) {
        scheduleNextCancellationCheck()
      }
    } catch (error) {
      console.error('[Subscription] Error checking cancellation status:', error)
      scheduleNextCancellationCheck()
    } finally {
      cancellationCheckInFlight.value = false
    }
  }

  const startCancellationWatcher = () => {
    if (!shouldWatchCancellation() || !subscriptionStatus.value?.is_active) {
      return
    }

    stopCancellationWatcher()
    watcherActive.value = true
    cancellationTracked.value = false
    cancellationAttempts.value = 0
    if (!detachFocusListener && defaultWindow) {
      detachFocusListener = useEventListener(defaultWindow, 'focus', () => {
        if (!watcherActive.value) return
        void checkForCancellation(true)
      })
    }
    scheduleNextCancellationCheck()
  }

  onScopeDispose(() => {
    stopCancellationWatcher()
  })

  return {
    startCancellationWatcher,
    stopCancellationWatcher
  }
}
