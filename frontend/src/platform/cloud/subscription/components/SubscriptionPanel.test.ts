import { createTestingPinia } from '@pinia/testing'
import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { computed, ref } from 'vue'
import { createI18n } from 'vue-i18n'

import SubscriptionPanel from '@/platform/cloud/subscription/components/SubscriptionPanel.vue'

// Mock state refs that can be modified between tests
const mockIsActiveSubscription = ref(false)
const mockIsCancelled = ref(false)
const mockSubscriptionTier = ref<
  'STANDARD' | 'CREATOR' | 'PRO' | 'FOUNDERS_EDITION' | null
>('CREATOR')
const mockIsYearlySubscription = ref(false)

const TIER_TO_NAME: Record<string, string> = {
  STANDARD: 'Standard',
  CREATOR: 'Creator',
  PRO: 'Pro',
  FOUNDERS_EDITION: "Founder's Edition"
}

// Mock composables - using computed to match composable return types
const mockSubscriptionData = {
  isActiveSubscription: computed(() => mockIsActiveSubscription.value),
  isCancelled: computed(() => mockIsCancelled.value),
  formattedRenewalDate: computed(() => '2024-12-31'),
  formattedEndDate: computed(() => '2024-12-31'),
  subscriptionTier: computed(() => mockSubscriptionTier.value),
  subscriptionTierName: computed(() => {
    if (!mockSubscriptionTier.value) return ''
    const baseName = TIER_TO_NAME[mockSubscriptionTier.value]
    return mockIsYearlySubscription.value ? `${baseName} Yearly` : baseName
  }),
  subscriptionStatus: computed(() => ({
    renewal_date: '2024-12-31T00:00:00Z'
  })),
  isYearlySubscription: computed(() => mockIsYearlySubscription.value),
  handleInvoiceHistory: vi.fn()
}

const mockCreditsData = {
  totalCredits: '10.00 Credits',
  monthlyBonusCredits: '5.00 Credits',
  prepaidCredits: '5.00 Credits',
  isLoadingBalance: false
}

const mockActionsData = {
  isLoadingSupport: false,
  handleAddApiCredits: vi.fn(),
  handleMessageSupport: vi.fn(),
  handleRefresh: vi.fn(),
  handleLearnMoreClick: vi.fn()
}

vi.mock('@/platform/cloud/subscription/composables/useSubscription', () => ({
  useSubscription: () => mockSubscriptionData
}))

vi.mock(
  '@/platform/cloud/subscription/composables/useSubscriptionCredits',
  () => ({
    useSubscriptionCredits: () => mockCreditsData
  })
)

vi.mock(
  '@/platform/cloud/subscription/composables/useSubscriptionActions',
  () => ({
    useSubscriptionActions: () => mockActionsData
  })
)

vi.mock(
  '@/platform/cloud/subscription/composables/useSubscriptionDialog',
  () => ({
    useSubscriptionDialog: () => ({
      show: vi.fn()
    })
  })
)

vi.mock('@/composables/auth/useFirebaseAuthActions', () => ({
  useFirebaseAuthActions: vi.fn(() => ({
    authActions: vi.fn(() => ({
      accessBillingPortal: vi.fn()
    }))
  }))
}))

vi.mock('@/composables/billing/useBillingContext', () => ({
  useBillingContext: () => ({
    isActiveSubscription: computed(() => mockIsActiveSubscription.value),
    manageSubscription: vi.fn()
  })
}))

// Create i18n instance for testing
const i18n = createI18n({
  legacy: false,
  locale: 'en',
  escapeParameter: true,
  messages: {
    en: {
      subscription: {
        title: 'Subscription',
        titleUnsubscribed: 'Subscribe',
        perMonth: '/ month',
        subscribeNow: 'Subscribe Now',
        manageSubscription: 'Manage Subscription',
        partnerNodesBalance: 'Partner Nodes Balance',
        partnerNodesDescription: 'Credits for partner nodes',
        totalCredits: 'Total Credits',
        creditsRemainingThisMonth: 'Included (Refills {date})',
        creditsRemainingThisYear: 'Included (Refills {date})',
        creditsYouveAdded: "Credits you've added",
        monthlyBonusDescription: 'Monthly bonus',
        prepaidDescription: 'Prepaid credits',
        monthlyCreditsRollover: 'Monthly credits rollover info',
        prepaidCreditsInfo: 'Prepaid credits info',
        viewUsageHistory: 'View Usage History',
        addCredits: 'Add Credits',
        yourPlanIncludes: 'Your plan includes',
        viewMoreDetailsPlans: 'View more details about plans & pricing',
        learnMore: 'Learn More',
        messageSupport: 'Message Support',
        invoiceHistory: 'Invoice History',
        partnerNodesCredits: 'Partner nodes pricing',
        renewsDate: 'Renews {date}',
        expiresDate: 'Expires {date}',
        tiers: {
          founder: {
            name: "Founder's Edition",
            price: '20.00',
            benefits: {
              monthlyCredits: '5,460',
              monthlyCreditsLabel: 'monthly credits',
              maxDuration: '30 min',
              maxDurationLabel: 'max duration of each workflow run',
              gpuLabel: 'RTX 6000 Pro (96GB VRAM)',
              addCreditsLabel: 'Add more credits whenever',
              customLoRAsLabel: 'Import your own LoRAs'
            }
          },
          standard: {
            name: 'Standard',
            price: '20.00',
            benefits: {
              monthlyCredits: '4,200',
              monthlyCreditsLabel: 'monthly credits',
              maxDuration: '30 min',
              maxDurationLabel: 'max duration of each workflow run',
              gpuLabel: 'RTX 6000 Pro (96GB VRAM)',
              addCreditsLabel: 'Add more credits whenever',
              customLoRAsLabel: 'Import your own LoRAs'
            }
          },
          creator: {
            name: 'Creator',
            price: '35.00',
            benefits: {
              monthlyCredits: '7,400',
              monthlyCreditsLabel: 'monthly credits',
              maxDuration: '30 min',
              maxDurationLabel: 'max duration of each workflow run',
              gpuLabel: 'RTX 6000 Pro (96GB VRAM)',
              addCreditsLabel: 'Add more credits whenever',
              customLoRAsLabel: 'Import your own LoRAs'
            }
          },
          pro: {
            name: 'Pro',
            price: '100.00',
            benefits: {
              monthlyCredits: '21,100',
              monthlyCreditsLabel: 'monthly credits',
              maxDuration: '1 hr',
              maxDurationLabel: 'max duration of each workflow run',
              gpuLabel: 'RTX 6000 Pro (96GB VRAM)',
              addCreditsLabel: 'Add more credits whenever',
              customLoRAsLabel: 'Import your own LoRAs'
            }
          }
        }
      }
    }
  }
})

function createWrapper(overrides = {}) {
  return mount(SubscriptionPanel, {
    global: {
      plugins: [createTestingPinia({ createSpy: vi.fn }), i18n],

      stubs: {
        CloudBadge: true,
        SubscribeButton: true,
        SubscriptionBenefits: true,
        Button: {
          template:
            '<button @click="$emit(\'click\')" :disabled="loading" :data-testid="label" :data-icon="icon"><slot/></button>',
          props: ['variant', 'size'],
          emits: ['click']
        },
        Skeleton: {
          template: '<div class="skeleton"></div>'
        }
      }
    },
    ...overrides
  })
}

describe('SubscriptionPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    // Reset mock state
    mockIsActiveSubscription.value = false
    mockIsCancelled.value = false
    mockSubscriptionTier.value = 'CREATOR'
    mockIsYearlySubscription.value = false
  })

  describe('subscription state functionality', () => {
    it('shows correct UI for active subscription', () => {
      mockIsActiveSubscription.value = true
      const wrapper = createWrapper()
      expect(wrapper.text()).toContain('Manage Subscription')
      expect(wrapper.text()).toContain('Add Credits')
    })

    it('shows correct UI for inactive subscription', () => {
      mockIsActiveSubscription.value = false
      const wrapper = createWrapper()
      expect(wrapper.findComponent({ name: 'SubscribeButton' }).exists()).toBe(
        true
      )
      expect(wrapper.text()).not.toContain('Manage Subscription')
      expect(wrapper.text()).not.toContain('Add Credits')
    })

    it('shows renewal date for active non-cancelled subscription', () => {
      mockIsActiveSubscription.value = true
      mockIsCancelled.value = false
      const wrapper = createWrapper()
      expect(wrapper.text()).toContain('Renews 2024-12-31')
    })

    it('shows expiry date for cancelled subscription', () => {
      mockIsActiveSubscription.value = true
      mockIsCancelled.value = true
      const wrapper = createWrapper()
      expect(wrapper.text()).toContain('Expires 2024-12-31')
    })

    it('displays FOUNDERS_EDITION tier correctly', () => {
      mockSubscriptionTier.value = 'FOUNDERS_EDITION'
      const wrapper = createWrapper()
      expect(wrapper.text()).toContain("Founder's Edition")
      expect(wrapper.text()).toContain('5,460')
    })

    it('displays CREATOR tier correctly', () => {
      mockSubscriptionTier.value = 'CREATOR'
      const wrapper = createWrapper()
      expect(wrapper.text()).toContain('Creator')
      expect(wrapper.text()).toContain('7,400')
    })
  })

  describe('credit display functionality', () => {
    it('displays dynamic credit values correctly', () => {
      const wrapper = createWrapper()
      expect(wrapper.text()).toContain('10.00 Credits')
      expect(wrapper.text()).toContain('5.00 Credits')
    })

    it('shows loading skeleton when fetching balance', () => {
      mockCreditsData.isLoadingBalance = true
      const wrapper = createWrapper()
      expect(wrapper.findAll('.skeleton').length).toBeGreaterThan(0)
    })

    it('hides skeleton when balance loaded', () => {
      mockCreditsData.isLoadingBalance = false
      const wrapper = createWrapper()
      expect(wrapper.findAll('.skeleton').length).toBe(0)
    })

    it('renders refill date with literal slashes', () => {
      mockIsActiveSubscription.value = true
      const wrapper = createWrapper()
      expect(wrapper.text()).toContain('Included (Refills 12/31/24)')
      expect(wrapper.text()).not.toContain('&#x2F;')
    })
  })

  // TODO: Re-enable when migrating to VTL so we can find by user visible content.
  describe.skip('action buttons', () => {
    it('should call handleLearnMoreClick when learn more is clicked', async () => {
      const wrapper = createWrapper()
      const learnMoreButton = wrapper.find('[data-testid="Learn More"]')
      await learnMoreButton.trigger('click')
      expect(mockActionsData.handleLearnMoreClick).toHaveBeenCalledOnce()
    })

    it('should call handleMessageSupport when message support is clicked', async () => {
      const wrapper = createWrapper()
      const supportButton = wrapper.find('[data-testid="Message Support"]')
      await supportButton.trigger('click')
      expect(mockActionsData.handleMessageSupport).toHaveBeenCalledOnce()
    })

    it('should call handleRefresh when refresh button is clicked', async () => {
      const wrapper = createWrapper()
      // Find the refresh button by icon
      const refreshButton = wrapper.find('[data-icon="pi pi-sync"]')
      await refreshButton.trigger('click')
      expect(mockActionsData.handleRefresh).toHaveBeenCalledOnce()
    })
  })

  describe.skip('loading states', () => {
    it('should show loading state on support button when loading', () => {
      mockActionsData.isLoadingSupport = true
      const wrapper = createWrapper()
      const supportButton = wrapper.find('[data-testid="Message Support"]')
      expect(supportButton.attributes('disabled')).toBeDefined()
    })

    it('should show loading state on refresh button when loading balance', () => {
      mockCreditsData.isLoadingBalance = true
      const wrapper = createWrapper()
      const refreshButton = wrapper.find('[data-icon="pi pi-sync"]')
      expect(refreshButton.attributes('disabled')).toBeDefined()
    })
  })
})
