import { useStorage } from '@vueuse/core'
import type { MaybeRefOrGetter } from 'vue'
import { computed, toValue } from 'vue'

import { isCloud, isDesktop, isNightly } from '@/platform/distribution/types'

import { useFeatureUsageTracker } from './useFeatureUsageTracker'

export interface FeatureSurveyConfig {
  /** Feature identifier. Must remain static after initialization. */
  featureId: string
  typeformId: string
  triggerThreshold?: number
  delayMs?: number
  enabled?: boolean
}

interface SurveyState {
  optedOut: boolean
  seenSurveys: Record<string, number>
}

const STORAGE_KEY = 'Comfy.SurveyState'
const GLOBAL_COOLDOWN_MS = 4 * 24 * 60 * 60 * 1000 // 4 days
const DEFAULT_THRESHOLD = 3
const DEFAULT_DELAY_MS = 5000

export function useSurveyEligibility(
  config: MaybeRefOrGetter<FeatureSurveyConfig>
) {
  const state = useStorage<SurveyState>(STORAGE_KEY, {
    optedOut: false,
    seenSurveys: {}
  })
  const resolvedConfig = computed(() => toValue(config))

  const { useCount } = useFeatureUsageTracker(resolvedConfig.value.featureId)

  const threshold = computed(
    () => resolvedConfig.value.triggerThreshold ?? DEFAULT_THRESHOLD
  )
  const delayMs = computed(
    () => resolvedConfig.value.delayMs ?? DEFAULT_DELAY_MS
  )
  const isSurveyEnabled = computed(() => resolvedConfig.value.enabled ?? true)

  const isNightlyLocalhost = computed(() => isNightly && !isCloud && !isDesktop)

  const hasReachedThreshold = computed(() => useCount.value >= threshold.value)

  const hasSeenSurvey = computed(
    () => !!state.value.seenSurveys[resolvedConfig.value.featureId]
  )

  const isInGlobalCooldown = computed(() => {
    const timestamps = Object.values(state.value.seenSurveys)
    if (timestamps.length === 0) return false
    const lastShown = Math.max(...timestamps)
    return Date.now() - lastShown < GLOBAL_COOLDOWN_MS
  })

  const hasOptedOut = computed(() => state.value.optedOut)

  const isEligible = computed(() => {
    if (!isSurveyEnabled.value) return false
    if (!isNightlyLocalhost.value) return false
    if (!hasReachedThreshold.value) return false
    if (hasSeenSurvey.value) return false
    if (isInGlobalCooldown.value) return false
    if (hasOptedOut.value) return false

    return true
  })

  function markSurveyShown() {
    state.value.seenSurveys[resolvedConfig.value.featureId] = Date.now()
  }

  function optOut() {
    state.value.optedOut = true
  }

  function resetState() {
    state.value = {
      optedOut: false,
      seenSurveys: {}
    }
  }

  return {
    isEligible,
    delayMs,
    markSurveyShown,
    optOut,
    resetState
  }
}
