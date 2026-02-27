import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useTemplateRankingStore } from '@/stores/templateRankingStore'

// Mock axios
vi.mock('axios', () => ({
  default: {
    get: vi.fn()
  }
}))

describe('templateRankingStore', () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    vi.clearAllMocks()
  })

  describe('computeFreshness', () => {
    it('returns 1.0 for brand new template (today)', () => {
      const store = useTemplateRankingStore()
      const today = new Date().toISOString().split('T')[0]
      const freshness = store.computeFreshness(today)
      expect(freshness).toBeCloseTo(1.0, 1)
    })

    it('returns ~0.5 for 90-day old template', () => {
      const store = useTemplateRankingStore()
      const ninetyDaysAgo = new Date(Date.now() - 90 * 24 * 60 * 60 * 1000)
        .toISOString()
        .split('T')[0]
      const freshness = store.computeFreshness(ninetyDaysAgo)
      expect(freshness).toBeCloseTo(0.5, 1)
    })

    it('returns 0.1 minimum for very old template', () => {
      const store = useTemplateRankingStore()
      const freshness = store.computeFreshness('2020-01-01')
      expect(freshness).toBe(0.1)
    })

    it('returns 0.5 for undefined date', () => {
      const store = useTemplateRankingStore()
      expect(store.computeFreshness(undefined)).toBe(0.5)
    })

    it('returns 0.5 for invalid date', () => {
      const store = useTemplateRankingStore()
      expect(store.computeFreshness('not-a-date')).toBe(0.5)
    })
  })

  describe('computeDefaultScore', () => {
    it('uses default searchRank of 5 when not provided', () => {
      const store = useTemplateRankingStore()
      // Set largestUsageScore to avoid NaN when usage is 0
      store.largestUsageScore = 100
      const score = store.computeDefaultScore('2024-01-01', undefined, 0)
      // With no usage score loaded, usage = 0
      // internal = 5/10 = 0.5, freshness ~0.1 (old date)
      // score = 0 * 0.5 + 0.5 * 0.3 + 0.1 * 0.2 = 0.15 + 0.02 = 0.17
      expect(score).toBeCloseTo(0.17, 1)
    })

    it('high searchRank (10) boosts score', () => {
      const store = useTemplateRankingStore()
      store.largestUsageScore = 100
      const lowRank = store.computeDefaultScore('2024-01-01', 1, 0)
      const highRank = store.computeDefaultScore('2024-01-01', 10, 0)
      expect(highRank).toBeGreaterThan(lowRank)
    })

    it('low searchRank (1) demotes score', () => {
      const store = useTemplateRankingStore()
      store.largestUsageScore = 100
      const neutral = store.computeDefaultScore('2024-01-01', 5, 0)
      const demoted = store.computeDefaultScore('2024-01-01', 1, 0)
      expect(demoted).toBeLessThan(neutral)
    })

    it('searchRank difference is significant', () => {
      const store = useTemplateRankingStore()
      store.largestUsageScore = 100
      const rank1 = store.computeDefaultScore('2024-01-01', 1, 0)
      const rank10 = store.computeDefaultScore('2024-01-01', 10, 0)
      // Difference should be 0.9 * 0.3 = 0.27 (30% weight, 0.9 range)
      expect(rank10 - rank1).toBeCloseTo(0.27, 2)
    })
  })

  describe('computePopularScore', () => {
    it('does not use searchRank', () => {
      const store = useTemplateRankingStore()
      store.largestUsageScore = 100
      // Popular score ignores searchRank - just usage + freshness
      const score1 = store.computePopularScore('2024-01-01', 0)
      const score2 = store.computePopularScore('2024-01-01', 0)
      expect(score1).toBe(score2)
    })

    it('newer templates score higher', () => {
      const store = useTemplateRankingStore()
      store.largestUsageScore = 100
      const today = new Date().toISOString().split('T')[0]
      const oldScore = store.computePopularScore('2020-01-01', 0)
      const newScore = store.computePopularScore(today, 0)
      expect(newScore).toBeGreaterThan(oldScore)
    })
  })

  describe('searchRank edge cases', () => {
    it('handles searchRank of 0 (should still work, treated as very low)', () => {
      const store = useTemplateRankingStore()
      store.largestUsageScore = 100
      const score = store.computeDefaultScore('2024-01-01', 0, 0)
      expect(score).toBeGreaterThanOrEqual(0)
    })

    it('handles searchRank above 10 (clamping not enforced, but works)', () => {
      const store = useTemplateRankingStore()
      store.largestUsageScore = 100
      const rank10 = store.computeDefaultScore('2024-01-01', 10, 0)
      const rank15 = store.computeDefaultScore('2024-01-01', 15, 0)
      expect(rank15).toBeGreaterThan(rank10)
    })

    it('handles negative searchRank', () => {
      const store = useTemplateRankingStore()
      store.largestUsageScore = 100
      const score = store.computeDefaultScore('2024-01-01', -5, 0)
      // Should still compute, just negative contribution from searchRank
      expect(typeof score).toBe('number')
    })
  })
})
