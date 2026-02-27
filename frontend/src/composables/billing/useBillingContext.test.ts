import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { Plan } from '@/platform/workspace/api/workspaceApi'

import { useBillingContext } from './useBillingContext'

const { mockTeamWorkspacesEnabled, mockIsPersonal, mockPlans } = vi.hoisted(
  () => ({
    mockTeamWorkspacesEnabled: { value: false },
    mockIsPersonal: { value: true },
    mockPlans: { value: [] as Plan[] }
  })
)

vi.mock('@vueuse/core', async (importOriginal) => {
  const original = await importOriginal()
  return {
    ...(original as Record<string, unknown>),
    createSharedComposable: (fn: (...args: unknown[]) => unknown) => fn
  }
})

vi.mock('@/composables/useFeatureFlags', () => ({
  useFeatureFlags: () => ({
    flags: {
      get teamWorkspacesEnabled() {
        return mockTeamWorkspacesEnabled.value
      }
    }
  })
}))

vi.mock('@/platform/workspace/stores/teamWorkspaceStore', () => ({
  useTeamWorkspaceStore: () => ({
    get isInPersonalWorkspace() {
      return mockIsPersonal.value
    },
    get activeWorkspace() {
      return mockIsPersonal.value
        ? { id: 'personal-123', type: 'personal' }
        : { id: 'team-456', type: 'team' }
    },
    updateActiveWorkspace: vi.fn()
  })
}))

vi.mock('@/platform/cloud/subscription/composables/useSubscription', () => ({
  useSubscription: () => ({
    isActiveSubscription: { value: true },
    subscriptionTier: { value: 'PRO' },
    subscriptionDuration: { value: 'MONTHLY' },
    formattedRenewalDate: { value: 'Jan 1, 2025' },
    formattedEndDate: { value: '' },
    isCancelled: { value: false },
    fetchStatus: vi.fn().mockResolvedValue(undefined),
    manageSubscription: vi.fn().mockResolvedValue(undefined),
    subscribe: vi.fn().mockResolvedValue(undefined),
    showSubscriptionDialog: vi.fn()
  })
}))

vi.mock(
  '@/platform/cloud/subscription/composables/useSubscriptionDialog',
  () => ({
    useSubscriptionDialog: () => ({
      show: vi.fn(),
      hide: vi.fn()
    })
  })
)

vi.mock('@/stores/firebaseAuthStore', () => ({
  useFirebaseAuthStore: () => ({
    balance: { amount_micros: 5000000 },
    fetchBalance: vi.fn().mockResolvedValue({ amount_micros: 5000000 })
  })
}))

vi.mock('@/platform/cloud/subscription/composables/useBillingPlans', () => ({
  useBillingPlans: () => ({
    get plans() {
      return mockPlans
    },
    currentPlanSlug: { value: null },
    isLoading: { value: false },
    error: { value: null },
    fetchPlans: vi.fn().mockResolvedValue(undefined),
    getPlanBySlug: vi.fn().mockReturnValue(null)
  })
}))

vi.mock('@/platform/workspace/api/workspaceApi', () => ({
  workspaceApi: {
    getBillingStatus: vi.fn().mockResolvedValue({
      is_active: true,
      has_funds: true,
      subscription_tier: 'PRO',
      subscription_duration: 'MONTHLY'
    }),
    getBillingBalance: vi.fn().mockResolvedValue({
      amount_micros: 10000000,
      currency: 'usd'
    }),
    subscribe: vi.fn().mockResolvedValue({ status: 'subscribed' }),
    previewSubscribe: vi.fn().mockResolvedValue({ allowed: true })
  }
}))

describe('useBillingContext', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
    mockTeamWorkspacesEnabled.value = false
    mockIsPersonal.value = true
    mockPlans.value = []
  })

  it('returns legacy type for personal workspace', () => {
    const { type } = useBillingContext()
    expect(type.value).toBe('legacy')
  })

  it('provides subscription info from legacy billing', () => {
    const { subscription } = useBillingContext()

    expect(subscription.value).toEqual({
      isActive: true,
      tier: 'PRO',
      duration: 'MONTHLY',
      planSlug: null,
      renewalDate: 'Jan 1, 2025',
      endDate: null,
      isCancelled: false,
      hasFunds: true
    })
  })

  it('provides balance info from legacy billing', () => {
    const { balance } = useBillingContext()

    expect(balance.value).toEqual({
      amountMicros: 5000000,
      currency: 'usd',
      effectiveBalanceMicros: 5000000,
      prepaidBalanceMicros: 0,
      cloudCreditBalanceMicros: 0
    })
  })

  it('exposes initialize action', async () => {
    const { initialize } = useBillingContext()
    await expect(initialize()).resolves.toBeUndefined()
  })

  it('exposes fetchStatus action', async () => {
    const { fetchStatus } = useBillingContext()
    await expect(fetchStatus()).resolves.toBeUndefined()
  })

  it('exposes fetchBalance action', async () => {
    const { fetchBalance } = useBillingContext()
    await expect(fetchBalance()).resolves.toBeUndefined()
  })

  it('exposes subscribe action', async () => {
    const { subscribe } = useBillingContext()
    await expect(subscribe('pro-monthly')).resolves.toBeUndefined()
  })

  it('exposes manageSubscription action', async () => {
    const { manageSubscription } = useBillingContext()
    await expect(manageSubscription()).resolves.toBeUndefined()
  })

  it('provides isActiveSubscription convenience computed', () => {
    const { isActiveSubscription } = useBillingContext()
    expect(isActiveSubscription.value).toBe(true)
  })

  it('exposes requireActiveSubscription action', async () => {
    const { requireActiveSubscription } = useBillingContext()
    await expect(requireActiveSubscription()).resolves.toBeUndefined()
  })

  it('exposes showSubscriptionDialog action', () => {
    const { showSubscriptionDialog } = useBillingContext()
    expect(() => showSubscriptionDialog()).not.toThrow()
  })

  describe('getMaxSeats', () => {
    it('returns 1 for personal workspaces regardless of tier', () => {
      const { getMaxSeats } = useBillingContext()
      expect(getMaxSeats('standard')).toBe(1)
      expect(getMaxSeats('creator')).toBe(1)
      expect(getMaxSeats('pro')).toBe(1)
      expect(getMaxSeats('founder')).toBe(1)
    })

    it('falls back to hardcoded values when no API plans available', () => {
      mockTeamWorkspacesEnabled.value = true
      mockIsPersonal.value = false

      const { getMaxSeats } = useBillingContext()
      expect(getMaxSeats('standard')).toBe(1)
      expect(getMaxSeats('creator')).toBe(5)
      expect(getMaxSeats('pro')).toBe(20)
      expect(getMaxSeats('founder')).toBe(1)
    })

    it('prefers API max_seats when plans are loaded', () => {
      mockTeamWorkspacesEnabled.value = true
      mockIsPersonal.value = false
      mockPlans.value = [
        {
          slug: 'pro-monthly',
          tier: 'PRO',
          duration: 'MONTHLY',
          price_cents: 10000,
          credits_cents: 2110000,
          max_seats: 50,
          availability: { available: true },
          seat_summary: {
            seat_count: 1,
            total_cost_cents: 10000,
            total_credits_cents: 2110000
          }
        }
      ]

      const { getMaxSeats } = useBillingContext()
      expect(getMaxSeats('pro')).toBe(50)
      // Tiers without API plans still fall back to hardcoded values
      expect(getMaxSeats('creator')).toBe(5)
    })
  })
})
