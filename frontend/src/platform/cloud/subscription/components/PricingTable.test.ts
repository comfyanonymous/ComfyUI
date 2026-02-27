import { createTestingPinia } from '@pinia/testing'
import { flushPromises, mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { computed, reactive, ref } from 'vue'
import { createI18n } from 'vue-i18n'

import PricingTable from '@/platform/cloud/subscription/components/PricingTable.vue'
import Button from '@/components/ui/button/Button.vue'

const mockIsActiveSubscription = ref(false)
const mockSubscriptionTier = ref<
  'STANDARD' | 'CREATOR' | 'PRO' | 'FOUNDERS_EDITION' | null
>(null)
const mockIsYearlySubscription = ref(false)
const mockAccessBillingPortal = vi.fn()
const mockReportError = vi.fn()
const mockTrackBeginCheckout = vi.fn()
const mockUserId = ref<string | undefined>('user-123')
const mockGetAuthHeader = vi.fn(() =>
  Promise.resolve({ Authorization: 'Bearer test-token' })
)
const mockGetCheckoutAttribution = vi.hoisted(() => vi.fn(() => ({})))

vi.mock('@/platform/cloud/subscription/composables/useSubscription', () => ({
  useSubscription: () => ({
    isActiveSubscription: computed(() => mockIsActiveSubscription.value),
    isFreeTier: computed(() => false),
    subscriptionTier: computed(() => mockSubscriptionTier.value),
    isYearlySubscription: computed(() => mockIsYearlySubscription.value),
    subscriptionStatus: ref(null)
  })
}))

vi.mock('@/composables/auth/useFirebaseAuthActions', () => ({
  useFirebaseAuthActions: () => ({
    accessBillingPortal: mockAccessBillingPortal,
    reportError: mockReportError
  })
}))

vi.mock('@/composables/useErrorHandling', () => ({
  useErrorHandling: () => ({
    wrapWithErrorHandlingAsync: vi.fn(
      (fn, errorHandler) =>
        async (...args: unknown[]) => {
          try {
            return await fn(...args)
          } catch (error) {
            if (errorHandler) {
              errorHandler(error)
            }
            throw error
          }
        }
    )
  })
}))

vi.mock('@/stores/firebaseAuthStore', () => ({
  useFirebaseAuthStore: () =>
    reactive({
      getAuthHeader: mockGetAuthHeader,
      userId: computed(() => mockUserId.value)
    }),
  FirebaseAuthStoreError: class extends Error {}
}))

vi.mock('@/platform/telemetry', () => ({
  useTelemetry: () => ({
    trackBeginCheckout: mockTrackBeginCheckout
  })
}))

vi.mock('@/platform/telemetry/utils/checkoutAttribution', () => ({
  getCheckoutAttribution: mockGetCheckoutAttribution
}))

vi.mock('@/platform/distribution/types', () => ({
  isCloud: true
}))

global.fetch = vi.fn()

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {
      subscription: {
        yearly: 'Yearly',
        monthly: 'Monthly',
        mostPopular: 'Most Popular',
        usdPerMonth: '/ month',
        billedYearly: 'Billed yearly ({total})',
        billedMonthly: 'Billed monthly',
        currentPlan: 'Current Plan',
        subscribeTo: 'Subscribe to {plan}',
        changeTo: 'Change to {plan}',
        tierNameYearly: '{name} Yearly',
        yearlyCreditsLabel: 'Yearly credits',
        monthlyCreditsLabel: 'Monthly credits',
        maxDurationLabel: 'Max duration',
        gpuLabel: 'GPU',
        addCreditsLabel: 'Add more credits',
        customLoRAsLabel: 'Custom LoRAs',
        videoEstimateLabel: 'Video estimate',
        videoEstimateHelp: 'How is this calculated?',
        videoEstimateExplanation: 'Based on average usage.',
        videoEstimateTryTemplate: 'Try template',
        maxDuration: {
          standard: '30 min',
          creator: '30 min',
          pro: '1 hr'
        },
        tiers: {
          standard: { name: 'Standard' },
          creator: { name: 'Creator' },
          pro: { name: 'Pro' }
        },
        benefits: {
          monthlyCredits: '{credits} monthly credits',
          maxDuration: '{duration} max duration',
          gpu: 'RTX 6000 Pro GPU',
          addCredits: 'Add more credits anytime',
          customLoRAs: 'Import custom LoRAs'
        }
      }
    }
  }
})

function createWrapper() {
  return mount(PricingTable, {
    global: {
      plugins: [createTestingPinia({ createSpy: vi.fn }), i18n],
      components: {
        Button
      },
      stubs: {
        SelectButton: {
          template: '<div><slot /></div>',
          props: ['modelValue', 'options'],
          emits: ['update:modelValue']
        },
        Popover: { template: '<div><slot /></div>' }
      }
    }
  })
}

describe('PricingTable', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockIsActiveSubscription.value = false
    mockSubscriptionTier.value = null
    mockIsYearlySubscription.value = false
    mockUserId.value = 'user-123'
    mockTrackBeginCheckout.mockReset()
    vi.mocked(global.fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ checkout_url: 'https://checkout.stripe.com/test' })
    } as Response)
  })

  describe('billing portal deep linking', () => {
    it('should call accessBillingPortal with yearly tier suffix when billing cycle is yearly (default)', async () => {
      mockIsActiveSubscription.value = true
      mockSubscriptionTier.value = 'STANDARD'

      const wrapper = createWrapper()
      await flushPromises()

      const creatorButton = wrapper
        .findAll('button')
        .find((btn) => btn.text().includes('Creator'))

      expect(creatorButton).toBeDefined()
      await creatorButton?.trigger('click')
      await flushPromises()

      expect(mockTrackBeginCheckout).toHaveBeenCalledWith({
        user_id: 'user-123',
        tier: 'creator',
        cycle: 'yearly',
        checkout_type: 'change',
        previous_tier: 'standard'
      })
      expect(mockAccessBillingPortal).toHaveBeenCalledWith('creator-yearly')
    })

    it('should call accessBillingPortal with different tiers correctly', async () => {
      mockIsActiveSubscription.value = true
      mockSubscriptionTier.value = 'STANDARD'

      const wrapper = createWrapper()
      await flushPromises()

      const proButton = wrapper
        .findAll('button')
        .find((btn) => btn.text().includes('Pro'))

      await proButton?.trigger('click')
      await flushPromises()

      expect(mockAccessBillingPortal).toHaveBeenCalledWith('pro-yearly')
    })

    it('should use the latest userId value when it changes after mount', async () => {
      mockIsActiveSubscription.value = true
      mockSubscriptionTier.value = 'STANDARD'
      mockUserId.value = 'user-early'

      const wrapper = createWrapper()
      await flushPromises()

      mockUserId.value = 'user-late'

      const creatorButton = wrapper
        .findAll('button')
        .find((btn) => btn.text().includes('Creator'))

      await creatorButton?.trigger('click')
      await flushPromises()

      expect(mockTrackBeginCheckout).toHaveBeenCalledTimes(1)
      expect(mockTrackBeginCheckout).toHaveBeenCalledWith({
        user_id: 'user-late',
        tier: 'creator',
        cycle: 'yearly',
        checkout_type: 'change',
        previous_tier: 'standard'
      })
    })

    it('should not call accessBillingPortal when clicking current plan', async () => {
      mockIsActiveSubscription.value = true
      mockSubscriptionTier.value = 'CREATOR'

      const wrapper = createWrapper()
      await flushPromises()

      const currentPlanButton = wrapper
        .findAll('button')
        .find((btn) => btn.text().includes('Current Plan'))

      await currentPlanButton?.trigger('click')
      await flushPromises()

      expect(mockAccessBillingPortal).not.toHaveBeenCalled()
    })

    it('should initiate checkout instead of billing portal for new subscribers', async () => {
      mockIsActiveSubscription.value = false

      const windowOpenSpy = vi
        .spyOn(window, 'open')
        .mockImplementation(() => null)

      const wrapper = createWrapper()
      await flushPromises()

      const subscribeButton = wrapper
        .findAll('button')
        .find((btn) => btn.text().includes('Subscribe'))

      await subscribeButton?.trigger('click')
      await flushPromises()

      expect(mockAccessBillingPortal).not.toHaveBeenCalled()
      expect(global.fetch).toHaveBeenCalledWith(
        expect.stringContaining('/customers/cloud-subscription-checkout/'),
        expect.any(Object)
      )
      expect(windowOpenSpy).toHaveBeenCalledWith(
        'https://checkout.stripe.com/test',
        '_blank'
      )

      windowOpenSpy.mockRestore()
    })

    it('should pass correct tier for each subscription level', async () => {
      mockIsActiveSubscription.value = true
      mockSubscriptionTier.value = 'PRO'

      const wrapper = createWrapper()
      await flushPromises()

      const standardButton = wrapper
        .findAll('button')
        .find((btn) => btn.text().includes('Standard'))

      await standardButton?.trigger('click')
      await flushPromises()

      expect(mockAccessBillingPortal).toHaveBeenCalledWith('standard-yearly')
    })
  })
})
