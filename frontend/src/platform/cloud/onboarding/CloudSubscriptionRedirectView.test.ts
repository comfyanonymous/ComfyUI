import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, test, vi } from 'vitest'
import { createI18n } from 'vue-i18n'

import CloudSubscriptionRedirectView from './CloudSubscriptionRedirectView.vue'

const flushPromises = () => new Promise((resolve) => setTimeout(resolve, 0))

// Router mocks
let mockQuery: Record<string, unknown> = {}
const mockRouterPush = vi.fn()

vi.mock('vue-router', () => ({
  useRoute: () => ({
    query: mockQuery
  }),
  useRouter: () => ({
    push: mockRouterPush
  })
}))

// Firebase / subscription mocks
const authActionMocks = vi.hoisted(() => ({
  reportError: vi.fn(),
  accessBillingPortal: vi.fn()
}))

vi.mock('@/composables/auth/useFirebaseAuthActions', () => ({
  useFirebaseAuthActions: () => authActionMocks
}))

vi.mock('@/composables/useErrorHandling', () => ({
  useErrorHandling: () => ({
    wrapWithErrorHandlingAsync:
      <T extends (...args: never[]) => unknown>(fn: T) =>
      (...args: Parameters<T>) =>
        fn(...args)
  })
}))

const subscriptionMocks = vi.hoisted(() => ({
  isActiveSubscription: { value: false },
  isInitialized: { value: true },
  subscriptionStatus: { value: null }
}))

vi.mock('@/platform/cloud/subscription/composables/useSubscription', () => ({
  useSubscription: () => subscriptionMocks
}))

vi.mock('@/composables/billing/useBillingContext', () => ({
  useBillingContext: () => subscriptionMocks
}))

// Avoid real network / isCloud behavior
const mockPerformSubscriptionCheckout = vi.fn()
vi.mock('@/platform/cloud/subscription/utils/subscriptionCheckoutUtil', () => ({
  performSubscriptionCheckout: (...args: unknown[]) =>
    mockPerformSubscriptionCheckout(...args)
}))

const createI18nInstance = () =>
  createI18n({
    legacy: false,
    locale: 'en',
    messages: {
      en: {
        cloudOnboarding: {
          skipToCloudApp: 'Skip to the cloud app'
        },
        g: {
          comfyOrgLogoAlt: 'Comfy org logo'
        },
        subscription: {
          subscribeTo: 'Subscribe to {plan}',
          tiers: {
            standard: { name: 'Standard' },
            creator: { name: 'Creator' },
            pro: { name: 'Pro' }
          }
        }
      }
    }
  })

const mountView = async (query: Record<string, unknown>) => {
  mockQuery = query

  const wrapper = mount(CloudSubscriptionRedirectView, {
    global: {
      plugins: [createI18nInstance()]
    }
  })

  await flushPromises()

  return { wrapper }
}

describe('CloudSubscriptionRedirectView', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockQuery = {}
    subscriptionMocks.isActiveSubscription.value = false
    subscriptionMocks.isInitialized.value = true
  })

  test('redirects to home when subscriptionType is missing', async () => {
    await mountView({})

    expect(mockRouterPush).toHaveBeenCalledWith('/')
  })

  test('redirects to home when subscriptionType is invalid', async () => {
    await mountView({ tier: 'invalid' })

    expect(mockRouterPush).toHaveBeenCalledWith('/')
  })

  test('shows subscription copy when subscriptionType is valid', async () => {
    const { wrapper } = await mountView({ tier: 'creator' })

    // Should not redirect to home
    expect(mockRouterPush).not.toHaveBeenCalledWith('/')

    // Shows copy under logo
    expect(wrapper.text()).toContain('Subscribe to Creator')

    // Triggers checkout flow
    expect(mockPerformSubscriptionCheckout).toHaveBeenCalledWith(
      'creator',
      'monthly',
      false
    )

    // Shows loading affordances
    expect(wrapper.findComponent({ name: 'ProgressSpinner' }).exists()).toBe(
      true
    )
    const skipLink = wrapper.get('a[href="/"]')
    expect(skipLink.text()).toContain('Skip to the cloud app')
  })

  test('opens billing portal when subscription is already active', async () => {
    subscriptionMocks.isActiveSubscription.value = true

    await mountView({ tier: 'creator' })

    expect(mockRouterPush).not.toHaveBeenCalledWith('/')
    expect(authActionMocks.accessBillingPortal).toHaveBeenCalledTimes(1)
    expect(mockPerformSubscriptionCheckout).not.toHaveBeenCalled()
  })

  test('uses first value when subscriptionType is an array', async () => {
    const { wrapper } = await mountView({
      tier: ['creator', 'pro']
    })

    expect(mockRouterPush).not.toHaveBeenCalledWith('/')
    expect(wrapper.text()).toContain('Subscribe to Creator')
    expect(mockPerformSubscriptionCheckout).toHaveBeenCalledWith(
      'creator',
      'monthly',
      false
    )
  })
})
