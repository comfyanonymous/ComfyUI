import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { useSurveyEligibility } from './useSurveyEligibility'

const SURVEY_STATE_KEY = 'Comfy.SurveyState'
const FEATURE_USAGE_KEY = 'Comfy.FeatureUsage'

const mockDistribution = vi.hoisted(() => ({
  isNightly: true,
  isCloud: false,
  isDesktop: false
}))

vi.mock('@/platform/distribution/types', () => ({
  get isNightly() {
    return mockDistribution.isNightly
  },
  get isCloud() {
    return mockDistribution.isCloud
  },
  get isDesktop() {
    return mockDistribution.isDesktop
  }
}))

describe('useSurveyEligibility', () => {
  const defaultConfig = {
    featureId: 'test-feature',
    typeformId: 'abc123'
  }

  beforeEach(() => {
    localStorage.clear()
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2024-06-15T12:00:00Z'))

    mockDistribution.isNightly = true
    mockDistribution.isCloud = false
    mockDistribution.isDesktop = false
  })

  afterEach(() => {
    localStorage.clear()
    vi.useRealTimers()
  })

  function setFeatureUsage(featureId: string, useCount: number) {
    const existing = JSON.parse(localStorage.getItem(FEATURE_USAGE_KEY) ?? '{}')
    existing[featureId] = {
      useCount,
      firstUsed: Date.now() - 1000,
      lastUsed: Date.now()
    }
    localStorage.setItem(FEATURE_USAGE_KEY, JSON.stringify(existing))
  }

  describe('eligibility checks', () => {
    it('is not eligible when not nightly', () => {
      mockDistribution.isNightly = false
      setFeatureUsage('test-feature', 5)

      const { isEligible } = useSurveyEligibility(defaultConfig)

      expect(isEligible.value).toBe(false)
    })

    it('is not eligible on cloud', () => {
      mockDistribution.isCloud = true
      setFeatureUsage('test-feature', 5)

      const { isEligible } = useSurveyEligibility(defaultConfig)

      expect(isEligible.value).toBe(false)
    })

    it('is not eligible on desktop', () => {
      mockDistribution.isDesktop = true
      setFeatureUsage('test-feature', 5)

      const { isEligible } = useSurveyEligibility(defaultConfig)

      expect(isEligible.value).toBe(false)
    })

    it('is not eligible below threshold', () => {
      setFeatureUsage('test-feature', 2)

      const { isEligible } = useSurveyEligibility(defaultConfig)

      expect(isEligible.value).toBe(false)
    })

    it('is eligible when all conditions met', () => {
      setFeatureUsage('test-feature', 3)

      const { isEligible } = useSurveyEligibility(defaultConfig)

      expect(isEligible.value).toBe(true)
    })

    it('respects custom threshold', () => {
      setFeatureUsage('test-feature', 5)

      const { isEligible } = useSurveyEligibility({
        ...defaultConfig,
        triggerThreshold: 10
      })

      expect(isEligible.value).toBe(false)
    })

    it('is not eligible when survey already seen', () => {
      setFeatureUsage('test-feature', 5)
      localStorage.setItem(
        SURVEY_STATE_KEY,
        JSON.stringify({
          optedOut: false,
          seenSurveys: { 'test-feature': Date.now() }
        })
      )

      const { isEligible } = useSurveyEligibility(defaultConfig)

      expect(isEligible.value).toBe(false)
    })

    it('is not eligible during global cooldown', () => {
      setFeatureUsage('test-feature', 5)
      const threeDaysAgo = Date.now() - 3 * 24 * 60 * 60 * 1000
      localStorage.setItem(
        SURVEY_STATE_KEY,
        JSON.stringify({
          optedOut: false,
          seenSurveys: { 'other-feature': threeDaysAgo }
        })
      )

      const { isEligible } = useSurveyEligibility(defaultConfig)

      expect(isEligible.value).toBe(false)
    })

    it('is eligible after global cooldown expires', () => {
      setFeatureUsage('test-feature', 5)
      const fiveDaysAgo = Date.now() - 5 * 24 * 60 * 60 * 1000
      localStorage.setItem(
        SURVEY_STATE_KEY,
        JSON.stringify({
          optedOut: false,
          seenSurveys: { 'other-feature': fiveDaysAgo }
        })
      )

      const { isEligible } = useSurveyEligibility(defaultConfig)

      expect(isEligible.value).toBe(true)
    })

    it('is not eligible when opted out', () => {
      setFeatureUsage('test-feature', 5)
      localStorage.setItem(
        SURVEY_STATE_KEY,
        JSON.stringify({
          optedOut: true,
          seenSurveys: {}
        })
      )

      const { isEligible } = useSurveyEligibility(defaultConfig)

      expect(isEligible.value).toBe(false)
    })

    it('is not eligible when config disabled', () => {
      setFeatureUsage('test-feature', 5)

      const { isEligible } = useSurveyEligibility({
        ...defaultConfig,
        enabled: false
      })

      expect(isEligible.value).toBe(false)
    })
  })

  describe('actions', () => {
    it('markSurveyShown makes user ineligible', () => {
      setFeatureUsage('test-feature', 5)

      const { isEligible, markSurveyShown } =
        useSurveyEligibility(defaultConfig)

      expect(isEligible.value).toBe(true)

      markSurveyShown()

      expect(isEligible.value).toBe(false)
    })

    it('optOut prevents all future surveys', () => {
      setFeatureUsage('test-feature', 5)

      const { isEligible, optOut } = useSurveyEligibility(defaultConfig)

      expect(isEligible.value).toBe(true)

      optOut()

      expect(isEligible.value).toBe(false)
    })

    it('resetState restores eligibility', () => {
      setFeatureUsage('test-feature', 5)
      localStorage.setItem(
        SURVEY_STATE_KEY,
        JSON.stringify({
          optedOut: true,
          seenSurveys: { 'test-feature': Date.now() }
        })
      )

      const { isEligible, resetState } = useSurveyEligibility(defaultConfig)

      expect(isEligible.value).toBe(false)

      resetState()

      expect(isEligible.value).toBe(true)
    })
  })

  describe('config values', () => {
    it('exposes delayMs from config', () => {
      const { delayMs } = useSurveyEligibility({
        ...defaultConfig,
        delayMs: 10000
      })

      expect(delayMs.value).toBe(10000)
    })
  })

  describe('persistence', () => {
    it('loads existing state from localStorage', () => {
      setFeatureUsage('test-feature', 5)
      localStorage.setItem(
        SURVEY_STATE_KEY,
        JSON.stringify({
          optedOut: false,
          seenSurveys: { 'test-feature': 1000 }
        })
      )

      const { isEligible } = useSurveyEligibility(defaultConfig)

      expect(isEligible.value).toBe(false)
    })
  })
})
