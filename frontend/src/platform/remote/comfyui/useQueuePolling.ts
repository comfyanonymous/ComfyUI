import { useTimeoutFn } from '@vueuse/core'
import { ref, watch } from 'vue'

import { useQueueStore } from '@/stores/queueStore'

const BASE_INTERVAL_MS = 8_000
const MAX_INTERVAL_MS = 32_000
const BACKOFF_MULTIPLIER = 1.5

export function useQueuePolling() {
  const queueStore = useQueueStore()
  const delay = ref(BASE_INTERVAL_MS)

  const { start, stop } = useTimeoutFn(
    () => {
      if (queueStore.activeJobsCount !== 1 || queueStore.isLoading) return
      delay.value = Math.min(delay.value * BACKOFF_MULTIPLIER, MAX_INTERVAL_MS)
      void queueStore.update()
    },
    delay,
    { immediate: false }
  )

  function scheduleNextPoll() {
    if (queueStore.activeJobsCount === 1 && !queueStore.isLoading) start()
    else stop()
  }

  watch(
    () => queueStore.activeJobsCount,
    () => {
      delay.value = BASE_INTERVAL_MS
      scheduleNextPoll()
    },
    { immediate: true }
  )

  // Reschedule after any update completes (whether from polling or
  // WebSocket events) to avoid redundant requests.
  watch(
    () => queueStore.isLoading,
    (loading, wasLoading) => {
      if (wasLoading && !loading) scheduleNextPoll()
    },
    { flush: 'sync' }
  )
}
