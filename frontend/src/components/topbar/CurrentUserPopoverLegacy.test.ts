import type { VueWrapper } from '@vue/test-utils'
import { mount } from '@vue/test-utils'
import { afterAll, beforeEach, describe, expect, it, vi } from 'vitest'
import { h } from 'vue'
import { createI18n } from 'vue-i18n'

import { formatCreditsFromCents } from '@/base/credits/comfyCredits'
import enMessages from '@/locales/en/main.json' with { type: 'json' }

import CurrentUserPopoverLegacy from './CurrentUserPopoverLegacy.vue'

// Mock all firebase modules
vi.mock('firebase/app', () => ({
  initializeApp: vi.fn(),
  getApp: vi.fn()
}))

vi.mock('firebase/auth', () => ({
  getAuth: vi.fn(),
  setPersistence: vi.fn(),
  browserLocalPersistence: {},
  onAuthStateChanged: vi.fn(),
  signInWithEmailAndPassword: vi.fn(),
  signOut: vi.fn()
}))

// Mock pinia
vi.mock('pinia')

// Mock showSettingsDialog and showTopUpCreditsDialog
const mockShowSettingsDialog = vi.fn()
const mockShowTopUpCreditsDialog = vi.fn()

// Mock the settings dialog composable
vi.mock('@/platform/settings/composables/useSettingsDialog', () => ({
  useSettingsDialog: vi.fn(() => ({
    show: mockShowSettingsDialog,
    hide: vi.fn(),
    showAbout: vi.fn()
  }))
}))

// Mock window.open
const originalWindowOpen = window.open
beforeEach(() => {
  window.open = vi.fn()
})

afterAll(() => {
  window.open = originalWindowOpen
})

// Mock the useCurrentUser composable
const mockHandleSignOut = vi.fn()
vi.mock('@/composables/auth/useCurrentUser', () => ({
  useCurrentUser: vi.fn(() => ({
    userPhotoUrl: 'https://example.com/avatar.jpg',
    userDisplayName: 'Test User',
    userEmail: 'test@example.com',
    handleSignOut: mockHandleSignOut
  }))
}))

// Mock the useFirebaseAuthActions composable
const mockLogout = vi.fn()
vi.mock('@/composables/auth/useFirebaseAuthActions', () => ({
  useFirebaseAuthActions: vi.fn(() => ({
    fetchBalance: vi.fn().mockResolvedValue(undefined),
    logout: mockLogout
  }))
}))

// Mock the dialog service
vi.mock('@/services/dialogService', () => ({
  useDialogService: vi.fn(() => ({
    showTopUpCreditsDialog: mockShowTopUpCreditsDialog
  }))
}))

// Mock the firebaseAuthStore with hoisted state for per-test manipulation
const mockAuthStoreState = vi.hoisted(() => ({
  balance: {
    amount_micros: 100_000,
    effective_balance_micros: 100_000,
    currency: 'usd'
  } as {
    amount_micros?: number
    effective_balance_micros?: number
    currency: string
  },
  isFetchingBalance: false
}))

vi.mock('@/stores/firebaseAuthStore', () => ({
  useFirebaseAuthStore: vi.fn(() => ({
    getAuthHeader: vi
      .fn()
      .mockResolvedValue({ Authorization: 'Bearer mock-token' }),
    balance: mockAuthStoreState.balance,
    isFetchingBalance: mockAuthStoreState.isFetchingBalance
  }))
}))

// Mock the useSubscription composable
const mockFetchStatus = vi.fn().mockResolvedValue(undefined)
vi.mock('@/platform/cloud/subscription/composables/useSubscription', () => ({
  useSubscription: vi.fn(() => ({
    isActiveSubscription: { value: true },
    subscriptionTierName: { value: 'Creator' },
    subscriptionTier: { value: 'CREATOR' },
    fetchStatus: mockFetchStatus
  }))
}))

// Mock the useSubscriptionDialog composable
const mockShowPricingTable = vi.fn()
vi.mock(
  '@/platform/cloud/subscription/composables/useSubscriptionDialog',
  () => ({
    useSubscriptionDialog: vi.fn(() => ({
      show: vi.fn(),
      showPricingTable: mockShowPricingTable,
      hide: vi.fn()
    }))
  })
)

// Mock UserAvatar component
vi.mock('@/components/common/UserAvatar.vue', () => ({
  default: {
    name: 'UserAvatarMock',
    render() {
      return h('div', 'Avatar')
    }
  }
}))

// Mock UserCredit component
vi.mock('@/components/common/UserCredit.vue', () => ({
  default: {
    name: 'UserCreditMock',
    render() {
      return h('div', 'Credit: 100')
    }
  }
}))

// Mock formatCreditsFromCents
vi.mock('@/base/credits/comfyCredits', () => ({
  formatCreditsFromCents: vi.fn(({ cents }) => (cents / 100).toString())
}))

// Mock useExternalLink
vi.mock('@/composables/useExternalLink', () => ({
  useExternalLink: vi.fn(() => ({
    buildDocsUrl: vi.fn((path) => `https://docs.comfy.org${path}`),
    docsPaths: {
      partnerNodesPricing: '/tutorials/partner-nodes/pricing'
    }
  }))
}))

// Mock useTelemetry
vi.mock('@/platform/telemetry', () => ({
  useTelemetry: vi.fn(() => ({
    trackAddApiCreditButtonClicked: vi.fn()
  }))
}))

// Mock isCloud
vi.mock('@/platform/distribution/types', () => ({
  isCloud: true
}))

vi.mock('@/platform/cloud/subscription/components/SubscribeButton.vue', () => ({
  default: {
    name: 'SubscribeButtonMock',
    render() {
      return h('div', 'Subscribe Button')
    }
  }
}))

describe('CurrentUserPopoverLegacy', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockAuthStoreState.balance = {
      amount_micros: 100_000,
      effective_balance_micros: 100_000,
      currency: 'usd'
    }
    mockAuthStoreState.isFetchingBalance = false
  })

  const mountComponent = (): VueWrapper => {
    const i18n = createI18n({
      legacy: false,
      locale: 'en',
      messages: { en: enMessages }
    })

    return mount(CurrentUserPopoverLegacy, {
      global: {
        plugins: [i18n],
        stubs: {
          Divider: true
        }
      }
    })
  }

  it('renders user information correctly', () => {
    const wrapper = mountComponent()

    expect(wrapper.text()).toContain('Test User')
    expect(wrapper.text()).toContain('test@example.com')
  })

  it('calls formatCreditsFromCents with correct parameters and displays formatted credits', () => {
    const wrapper = mountComponent()

    expect(formatCreditsFromCents).toHaveBeenCalledWith({
      cents: 100_000,
      locale: 'en',
      numberOptions: {
        minimumFractionDigits: 0,
        maximumFractionDigits: 2
      }
    })

    // Verify the formatted credit string (1000) is rendered in the DOM
    expect(wrapper.text()).toContain('1000')
  })

  it('renders logout menu item with correct text', () => {
    const wrapper = mountComponent()

    const logoutItem = wrapper.find('[data-testid="logout-menu-item"]')
    expect(logoutItem.exists()).toBe(true)
    expect(wrapper.text()).toContain('Log Out')
  })

  it('opens user settings and emits close event when settings item is clicked', async () => {
    const wrapper = mountComponent()

    const settingsItem = wrapper.find('[data-testid="user-settings-menu-item"]')
    expect(settingsItem.exists()).toBe(true)

    await settingsItem.trigger('click')

    // Verify showSettingsDialog was called with 'user'
    expect(mockShowSettingsDialog).toHaveBeenCalledWith('user')

    // Verify close event was emitted
    expect(wrapper.emitted('close')).toBeTruthy()
    expect(wrapper.emitted('close')!.length).toBe(1)
  })

  it('calls logout function and emits close event when logout item is clicked', async () => {
    const wrapper = mountComponent()

    const logoutItem = wrapper.find('[data-testid="logout-menu-item"]')
    expect(logoutItem.exists()).toBe(true)

    await logoutItem.trigger('click')

    // Verify handleSignOut was called
    expect(mockHandleSignOut).toHaveBeenCalled()

    // Verify close event was emitted
    expect(wrapper.emitted('close')).toBeTruthy()
    expect(wrapper.emitted('close')!.length).toBe(1)
  })

  it('opens API pricing docs and emits close event when partner nodes item is clicked', async () => {
    const wrapper = mountComponent()

    const partnerNodesItem = wrapper.find(
      '[data-testid="partner-nodes-menu-item"]'
    )
    expect(partnerNodesItem.exists()).toBe(true)

    await partnerNodesItem.trigger('click')

    // Verify window.open was called with the correct URL
    expect(window.open).toHaveBeenCalledWith(
      'https://docs.comfy.org/tutorials/partner-nodes/pricing',
      '_blank'
    )

    // Verify close event was emitted
    expect(wrapper.emitted('close')).toBeTruthy()
    expect(wrapper.emitted('close')!.length).toBe(1)
  })

  it('opens top-up dialog and emits close event when top-up button is clicked', async () => {
    const wrapper = mountComponent()

    const topUpButton = wrapper.find('[data-testid="add-credits-button"]')
    expect(topUpButton.exists()).toBe(true)

    await topUpButton.trigger('click')

    // Verify showTopUpCreditsDialog was called
    expect(mockShowTopUpCreditsDialog).toHaveBeenCalled()

    // Verify close event was emitted
    expect(wrapper.emitted('close')).toBeTruthy()
    expect(wrapper.emitted('close')!.length).toBe(1)
  })

  it('opens subscription dialog and emits close event when plans & pricing item is clicked', async () => {
    const wrapper = mountComponent()

    const plansPricingItem = wrapper.find(
      '[data-testid="plans-pricing-menu-item"]'
    )
    expect(plansPricingItem.exists()).toBe(true)

    await plansPricingItem.trigger('click')

    // Verify showPricingTable was called
    expect(mockShowPricingTable).toHaveBeenCalled()

    // Verify close event was emitted
    expect(wrapper.emitted('close')).toBeTruthy()
    expect(wrapper.emitted('close')!.length).toBe(1)
  })

  describe('effective_balance_micros handling', () => {
    it('uses effective_balance_micros when present (positive balance)', () => {
      mockAuthStoreState.balance = {
        amount_micros: 200_000,
        effective_balance_micros: 150_000,
        currency: 'usd'
      }

      const wrapper = mountComponent()

      expect(formatCreditsFromCents).toHaveBeenCalledWith({
        cents: 150_000,
        locale: 'en',
        numberOptions: {
          minimumFractionDigits: 0,
          maximumFractionDigits: 2
        }
      })
      expect(wrapper.text()).toContain('1500')
    })

    it('uses effective_balance_micros when zero', () => {
      mockAuthStoreState.balance = {
        amount_micros: 100_000,
        effective_balance_micros: 0,
        currency: 'usd'
      }

      const wrapper = mountComponent()

      expect(formatCreditsFromCents).toHaveBeenCalledWith({
        cents: 0,
        locale: 'en',
        numberOptions: {
          minimumFractionDigits: 0,
          maximumFractionDigits: 2
        }
      })
      expect(wrapper.text()).toContain('0')
    })

    it('uses effective_balance_micros when negative', () => {
      mockAuthStoreState.balance = {
        amount_micros: 0,
        effective_balance_micros: -50_000,
        currency: 'usd'
      }

      const wrapper = mountComponent()

      expect(formatCreditsFromCents).toHaveBeenCalledWith({
        cents: -50_000,
        locale: 'en',
        numberOptions: {
          minimumFractionDigits: 0,
          maximumFractionDigits: 2
        }
      })
      expect(wrapper.text()).toContain('-500')
    })

    it('falls back to amount_micros when effective_balance_micros is missing', () => {
      mockAuthStoreState.balance = {
        amount_micros: 100_000,
        currency: 'usd'
      }

      const wrapper = mountComponent()

      expect(formatCreditsFromCents).toHaveBeenCalledWith({
        cents: 100_000,
        locale: 'en',
        numberOptions: {
          minimumFractionDigits: 0,
          maximumFractionDigits: 2
        }
      })
      expect(wrapper.text()).toContain('1000')
    })

    it('falls back to 0 when both effective_balance_micros and amount_micros are missing', () => {
      mockAuthStoreState.balance = {
        currency: 'usd'
      }

      const wrapper = mountComponent()

      expect(formatCreditsFromCents).toHaveBeenCalledWith({
        cents: 0,
        locale: 'en',
        numberOptions: {
          minimumFractionDigits: 0,
          maximumFractionDigits: 2
        }
      })
      expect(wrapper.text()).toContain('0')
    })
  })
})
