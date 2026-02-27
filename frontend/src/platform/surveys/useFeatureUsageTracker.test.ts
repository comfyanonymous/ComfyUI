import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useFeatureUsageTracker } from './useFeatureUsageTracker'

const STORAGE_KEY = 'Comfy.FeatureUsage'

describe('useFeatureUsageTracker', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.clearAllMocks()
  })

  afterEach(() => {
    localStorage.clear()
  })

  it('initializes with zero count for new feature', () => {
    const { useCount } = useFeatureUsageTracker('test-feature-1')

    expect(useCount.value).toBe(0)
  })

  it('increments count on trackUsage', () => {
    const { useCount, trackUsage } = useFeatureUsageTracker('test-feature-2')

    expect(useCount.value).toBe(0)

    trackUsage()
    expect(useCount.value).toBe(1)

    trackUsage()
    expect(useCount.value).toBe(2)
  })

  it('sets firstUsed only on first use', () => {
    vi.useFakeTimers()
    const firstTs = 1000000
    vi.setSystemTime(firstTs)
    try {
      const { usage, trackUsage } = useFeatureUsageTracker('test-feature-3')

      trackUsage()
      expect(usage.value?.firstUsed).toBe(firstTs)

      vi.setSystemTime(firstTs + 5000)
      trackUsage()
      expect(usage.value?.firstUsed).toBe(firstTs)
    } finally {
      vi.useRealTimers()
    }
  })

  it('updates lastUsed on each use', () => {
    vi.useFakeTimers()
    try {
      const { usage, trackUsage } = useFeatureUsageTracker('test-feature-4')

      trackUsage()
      const firstLastUsed = usage.value?.lastUsed ?? 0

      vi.advanceTimersByTime(10)
      trackUsage()

      expect(usage.value?.lastUsed).toBeGreaterThan(firstLastUsed)
    } finally {
      vi.useRealTimers()
    }
  })

  it('reset clears feature data', () => {
    const { useCount, trackUsage, reset } =
      useFeatureUsageTracker('test-feature-5')

    trackUsage()
    trackUsage()
    expect(useCount.value).toBe(2)

    reset()
    expect(useCount.value).toBe(0)
  })

  it('tracks multiple features independently', () => {
    const featureA = useFeatureUsageTracker('feature-a')
    const featureB = useFeatureUsageTracker('feature-b')

    featureA.trackUsage()
    featureA.trackUsage()
    featureB.trackUsage()

    expect(featureA.useCount.value).toBe(2)
    expect(featureB.useCount.value).toBe(1)
  })

  it('persists to localStorage', async () => {
    vi.useFakeTimers()
    try {
      const { trackUsage } = useFeatureUsageTracker('persisted-feature')

      trackUsage()
      await vi.runAllTimersAsync()

      const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? '{}')
      expect(stored['persisted-feature']?.useCount).toBe(1)
    } finally {
      vi.useRealTimers()
    }
  })

  it('loads existing data from localStorage', () => {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        'existing-feature': { useCount: 5, firstUsed: 1000, lastUsed: 2000 }
      })
    )

    const { useCount } = useFeatureUsageTracker('existing-feature')

    expect(useCount.value).toBe(5)
  })
})
