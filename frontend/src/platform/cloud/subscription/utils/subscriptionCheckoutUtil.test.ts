import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { computed, reactive } from 'vue'

import { performSubscriptionCheckout } from './subscriptionCheckoutUtil'

const {
  mockTelemetry,
  mockGetAuthHeader,
  mockUserId,
  mockIsCloud,
  mockGetCheckoutAttribution
} = vi.hoisted(() => ({
  mockTelemetry: {
    trackBeginCheckout: vi.fn()
  },
  mockGetAuthHeader: vi.fn(() =>
    Promise.resolve({ Authorization: 'Bearer test-token' })
  ),
  mockUserId: { value: 'user-123' as string | undefined },
  mockIsCloud: { value: true },
  mockGetCheckoutAttribution: vi.fn(() => ({
    ga_client_id: 'ga-client-id',
    ga_session_id: 'ga-session-id',
    ga_session_number: 'ga-session-number',
    im_ref: 'impact-click-123',
    utm_source: 'impact',
    utm_medium: 'affiliate',
    utm_campaign: 'spring-launch',
    gclid: 'gclid-123',
    gbraid: 'gbraid-456',
    wbraid: 'wbraid-789'
  }))
}))

vi.mock('@/platform/telemetry', () => ({
  useTelemetry: vi.fn(() => mockTelemetry)
}))

vi.mock('@/stores/firebaseAuthStore', () => ({
  useFirebaseAuthStore: vi.fn(() =>
    reactive({
      getAuthHeader: mockGetAuthHeader,
      userId: computed(() => mockUserId.value)
    })
  ),
  FirebaseAuthStoreError: class extends Error {}
}))

vi.mock('@/platform/distribution/types', () => ({
  get isCloud() {
    return mockIsCloud.value
  }
}))

vi.mock('@/platform/telemetry/utils/checkoutAttribution', () => ({
  getCheckoutAttribution: mockGetCheckoutAttribution
}))

global.fetch = vi.fn()

type Distribution = 'desktop' | 'localhost' | 'cloud'

const setDistribution = (distribution: Distribution) => {
  ;(
    globalThis as typeof globalThis & { __DISTRIBUTION__: Distribution }
  ).__DISTRIBUTION__ = distribution
}

function createDeferred<T>() {
  let resolve: (value: T) => void = () => {}
  const promise = new Promise<T>((res) => {
    resolve = res
  })

  return { promise, resolve }
}

describe('performSubscriptionCheckout', () => {
  beforeEach(() => {
    setDistribution('cloud')
    vi.clearAllMocks()
    mockIsCloud.value = true
    mockUserId.value = 'user-123'
  })

  afterEach(() => {
    vi.restoreAllMocks()
    setDistribution('localhost')
  })

  it('tracks begin_checkout with user id and tier metadata', async () => {
    const checkoutUrl = 'https://checkout.stripe.com/test'
    const openSpy = vi.spyOn(window, 'open').mockImplementation(() => null)

    vi.mocked(global.fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ checkout_url: checkoutUrl })
    } as Response)

    await performSubscriptionCheckout('pro', 'yearly', true)

    expect(mockTelemetry.trackBeginCheckout).toHaveBeenCalledWith({
      user_id: 'user-123',
      tier: 'pro',
      cycle: 'yearly',
      checkout_type: 'new',
      ga_client_id: 'ga-client-id',
      ga_session_id: 'ga-session-id',
      ga_session_number: 'ga-session-number',
      im_ref: 'impact-click-123',
      utm_source: 'impact',
      utm_medium: 'affiliate',
      utm_campaign: 'spring-launch',
      gclid: 'gclid-123',
      gbraid: 'gbraid-456',
      wbraid: 'wbraid-789'
    })
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining(
        '/customers/cloud-subscription-checkout/pro-yearly'
      ),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          ga_client_id: 'ga-client-id',
          ga_session_id: 'ga-session-id',
          ga_session_number: 'ga-session-number',
          im_ref: 'impact-click-123',
          utm_source: 'impact',
          utm_medium: 'affiliate',
          utm_campaign: 'spring-launch',
          gclid: 'gclid-123',
          gbraid: 'gbraid-456',
          wbraid: 'wbraid-789'
        })
      })
    )
    expect(openSpy).toHaveBeenCalledWith(checkoutUrl, '_blank')
  })

  it('continues checkout when attribution collection fails', async () => {
    const checkoutUrl = 'https://checkout.stripe.com/test'
    const openSpy = vi.spyOn(window, 'open').mockImplementation(() => null)
    const warnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {})

    mockGetCheckoutAttribution.mockRejectedValueOnce(
      new Error('Attribution failed')
    )
    vi.mocked(global.fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ checkout_url: checkoutUrl })
    } as Response)

    await performSubscriptionCheckout('pro', 'monthly', true)

    expect(warnSpy).toHaveBeenCalledWith(
      '[SubscriptionCheckout] Failed to collect checkout attribution',
      expect.any(Error)
    )
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining('/customers/cloud-subscription-checkout/pro'),
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({})
      })
    )
    expect(mockTelemetry.trackBeginCheckout).toHaveBeenCalledWith({
      user_id: 'user-123',
      tier: 'pro',
      cycle: 'monthly',
      checkout_type: 'new'
    })
    expect(openSpy).toHaveBeenCalledWith(checkoutUrl, '_blank')
  })

  it('uses the latest userId when it changes after checkout starts', async () => {
    const checkoutUrl = 'https://checkout.stripe.com/test'
    const openSpy = vi.spyOn(window, 'open').mockImplementation(() => null)
    const authHeader = createDeferred<{ Authorization: string }>()

    mockUserId.value = 'user-early'
    mockGetAuthHeader.mockImplementationOnce(() => authHeader.promise)
    vi.mocked(global.fetch).mockResolvedValue({
      ok: true,
      json: async () => ({ checkout_url: checkoutUrl })
    } as Response)

    const checkoutPromise = performSubscriptionCheckout('pro', 'yearly', true)

    mockUserId.value = 'user-late'
    authHeader.resolve({ Authorization: 'Bearer test-token' })

    await checkoutPromise

    expect(mockTelemetry.trackBeginCheckout).toHaveBeenCalledTimes(1)
    expect(mockTelemetry.trackBeginCheckout).toHaveBeenCalledWith(
      expect.objectContaining({
        user_id: 'user-late',
        tier: 'pro',
        cycle: 'yearly',
        checkout_type: 'new'
      })
    )
    expect(openSpy).toHaveBeenCalledWith(checkoutUrl, '_blank')
  })
})
