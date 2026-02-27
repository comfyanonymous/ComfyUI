import { beforeEach, describe, expect, it, vi } from 'vitest'
import { computed } from 'vue'
import type * as VueI18nModule from 'vue-i18n'

import * as comfyCredits from '@/base/credits/comfyCredits'
import { useSubscriptionCredits } from '@/platform/cloud/subscription/composables/useSubscriptionCredits'

// Shared mock state (reset in beforeEach)
let mockBillingBalance: {
  amountMicros: number
  cloudCreditBalanceMicros?: number
  prepaidBalanceMicros?: number
} | null = null
let mockBillingIsLoading = false

vi.mock(
  'vue-i18n',
  async (importOriginal: () => Promise<typeof VueI18nModule>) => {
    const actual = await importOriginal()
    return {
      ...actual,
      useI18n: () => ({
        t: () => 'Credits',
        locale: { value: 'en-US' }
      })
    }
  }
)

// Mock useBillingContext - returns computed refs that read from module-level state
vi.mock('@/composables/billing/useBillingContext', () => ({
  useBillingContext: () => ({
    balance: computed(() => mockBillingBalance),
    isLoading: computed(() => mockBillingIsLoading)
  })
}))

describe('useSubscriptionCredits', () => {
  beforeEach(() => {
    mockBillingBalance = null
    mockBillingIsLoading = false
    vi.clearAllMocks()
  })

  describe('totalCredits', () => {
    it('should return "0" when balance is null', () => {
      mockBillingBalance = null
      const { totalCredits } = useSubscriptionCredits()
      expect(totalCredits.value).toBe('0')
    })

    it('should format amountMicros correctly', () => {
      mockBillingBalance = { amountMicros: 100 }
      const { totalCredits } = useSubscriptionCredits()
      expect(totalCredits.value).toBe('211')
    })

    it('should handle formatting errors by throwing', () => {
      const formatSpy = vi.spyOn(comfyCredits, 'formatCreditsFromCents')
      formatSpy.mockImplementationOnce(() => {
        throw new Error('Formatting error')
      })

      mockBillingBalance = { amountMicros: 100 }
      const { totalCredits } = useSubscriptionCredits()
      expect(() => totalCredits.value).toThrow('Formatting error')
      formatSpy.mockRestore()
    })
  })

  describe('monthlyBonusCredits', () => {
    it('should return "0" when cloudCreditBalanceMicros is missing', () => {
      mockBillingBalance = { amountMicros: 100 }
      const { monthlyBonusCredits } = useSubscriptionCredits()
      expect(monthlyBonusCredits.value).toBe('0')
    })

    it('should format cloudCreditBalanceMicros correctly', () => {
      mockBillingBalance = {
        amountMicros: 300,
        cloudCreditBalanceMicros: 200
      }
      const { monthlyBonusCredits } = useSubscriptionCredits()
      expect(monthlyBonusCredits.value).toBe('422')
    })
  })

  describe('prepaidCredits', () => {
    it('should return "0" when prepaidBalanceMicros is missing', () => {
      mockBillingBalance = { amountMicros: 100 }
      const { prepaidCredits } = useSubscriptionCredits()
      expect(prepaidCredits.value).toBe('0')
    })

    it('should format prepaidBalanceMicros correctly', () => {
      mockBillingBalance = {
        amountMicros: 500,
        prepaidBalanceMicros: 300
      }
      const { prepaidCredits } = useSubscriptionCredits()
      expect(prepaidCredits.value).toBe('633')
    })
  })

  describe('isLoadingBalance', () => {
    it('should reflect billingContext.isLoading', () => {
      mockBillingIsLoading = true
      const { isLoadingBalance } = useSubscriptionCredits()
      expect(isLoadingBalance.value).toBe(true)

      mockBillingIsLoading = false
      // Need to re-get the composable since computed caches the value
      const { isLoadingBalance: reloaded } = useSubscriptionCredits()
      expect(reloaded.value).toBe(false)
    })
  })
})
