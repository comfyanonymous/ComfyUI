import { api } from '@/scripts/api'
import { app } from '@/scripts/app'
import {
  isInstantRunningMode,
  useQueuePendingTaskCountStore,
  useQueueSettingsStore
} from '@/stores/queueStore'

export function setupAutoQueueHandler() {
  const queueCountStore = useQueuePendingTaskCountStore()
  const queueSettingsStore = useQueueSettingsStore()

  let graphHasChanged = false
  let internalCount = 0 // Use an internal counter here so it is instantly updated when re-queuing
  api.addEventListener('graphChanged', () => {
    if (queueSettingsStore.mode === 'change') {
      if (internalCount) {
        graphHasChanged = true
      } else {
        graphHasChanged = false
        // Queue the prompt in the background
        void app.queuePrompt(0, queueSettingsStore.batchCount)
        internalCount++
      }
    }
  })

  queueCountStore.$subscribe(
    async () => {
      internalCount = queueCountStore.count
      if (!internalCount && !app.lastExecutionError) {
        if (
          isInstantRunningMode(queueSettingsStore.mode) ||
          (queueSettingsStore.mode === 'change' && graphHasChanged)
        ) {
          graphHasChanged = false
          await app.queuePrompt(0, queueSettingsStore.batchCount)
        }
      }
    },
    { detached: true }
  )
}
