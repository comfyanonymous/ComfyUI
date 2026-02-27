import { useStorage } from '@vueuse/core'
import { computed } from 'vue'

interface FeatureUsage {
  useCount: number
  firstUsed: number
  lastUsed: number
}

type FeatureUsageRecord = Record<string, FeatureUsage>

const STORAGE_KEY = 'Comfy.FeatureUsage'

/**
 * Tracks feature usage for survey eligibility.
 * Persists to localStorage.
 */
export function useFeatureUsageTracker(featureId: string) {
  const usageData = useStorage<FeatureUsageRecord>(STORAGE_KEY, {})

  const usage = computed(() => usageData.value[featureId])

  const useCount = computed(() => usage.value?.useCount ?? 0)

  function trackUsage() {
    const now = Date.now()
    const existing = usageData.value[featureId]

    usageData.value[featureId] = {
      useCount: (existing?.useCount ?? 0) + 1,
      firstUsed: existing?.firstUsed ?? now,
      lastUsed: now
    }
  }

  function reset() {
    delete usageData.value[featureId]
  }

  return {
    usage,
    useCount,
    trackUsage,
    reset
  }
}
