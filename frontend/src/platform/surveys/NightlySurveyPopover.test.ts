import { mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

const FEATURE_USAGE_KEY = 'Comfy.FeatureUsage'
const POPOVER_SELECTOR = '[data-testid="nightly-survey-popover"]'

const mockIsNightly = vi.hoisted(() => ({ value: true }))
const mockIsCloud = vi.hoisted(() => ({ value: false }))
const mockIsDesktop = vi.hoisted(() => ({ value: false }))

vi.mock('@/platform/distribution/types', () => ({
  get isNightly() {
    return mockIsNightly.value
  },
  get isCloud() {
    return mockIsCloud.value
  },
  get isDesktop() {
    return mockIsDesktop.value
  }
}))

vi.mock('vue-i18n', () => ({
  useI18n: vi.fn(() => ({
    t: (key: string) => key
  }))
}))

describe('NightlySurveyPopover', () => {
  const defaultConfig = {
    featureId: 'test-feature',
    typeformId: 'abc123',
    triggerThreshold: 3,
    delayMs: 100,
    enabled: true
  }

  function setFeatureUsage(featureId: string, useCount: number) {
    const existing = JSON.parse(localStorage.getItem(FEATURE_USAGE_KEY) ?? '{}')
    existing[featureId] = {
      useCount,
      firstUsed: Date.now() - 1000,
      lastUsed: Date.now()
    }
    localStorage.setItem(FEATURE_USAGE_KEY, JSON.stringify(existing))
  }

  beforeEach(() => {
    localStorage.clear()
    vi.resetModules()
    vi.useFakeTimers()
    vi.setSystemTime(new Date('2024-06-15T12:00:00Z'))

    mockIsNightly.value = true
    mockIsCloud.value = false
    mockIsDesktop.value = false
  })

  afterEach(() => {
    localStorage.clear()
    vi.useRealTimers()
    document.body.innerHTML = ''
  })

  async function mountComponent(config = defaultConfig) {
    const { default: NightlySurveyPopover } =
      await import('./NightlySurveyPopover.vue')
    return mount(NightlySurveyPopover, {
      props: { config },
      global: {
        stubs: {
          Teleport: true
        }
      },
      attachTo: document.body
    })
  }

  describe('visibility', () => {
    it('shows popover after delay when eligible', async () => {
      setFeatureUsage('test-feature', 5)

      const wrapper = await mountComponent()
      await nextTick()

      expect(wrapper.find(POPOVER_SELECTOR).exists()).toBe(false)

      await vi.advanceTimersByTimeAsync(100)
      await nextTick()

      expect(wrapper.find(POPOVER_SELECTOR).exists()).toBe(true)
    })

    it('does not show when not eligible', async () => {
      setFeatureUsage('test-feature', 1)

      const wrapper = await mountComponent()
      await nextTick()
      await vi.advanceTimersByTimeAsync(1000)
      await nextTick()

      expect(wrapper.find(POPOVER_SELECTOR).exists()).toBe(false)
    })

    it('does not show on cloud', async () => {
      mockIsCloud.value = true
      setFeatureUsage('test-feature', 5)

      const wrapper = await mountComponent()
      await nextTick()
      await vi.advanceTimersByTimeAsync(1000)
      await nextTick()

      expect(wrapper.find(POPOVER_SELECTOR).exists()).toBe(false)
    })
  })

  describe('user actions', () => {
    it('emits shown event when displayed', async () => {
      setFeatureUsage('test-feature', 5)

      const wrapper = await mountComponent()
      await vi.advanceTimersByTimeAsync(100)
      await nextTick()

      expect(wrapper.emitted('shown')).toHaveLength(1)
    })

    it('emits dismissed when close button clicked', async () => {
      setFeatureUsage('test-feature', 5)

      const wrapper = await mountComponent()
      await vi.advanceTimersByTimeAsync(100)
      await nextTick()

      const closeButton = wrapper.find('[aria-label="g.close"]')
      await closeButton.trigger('click')

      expect(wrapper.emitted('dismissed')).toHaveLength(1)
    })

    it('emits optedOut when opt out button clicked', async () => {
      setFeatureUsage('test-feature', 5)

      const wrapper = await mountComponent()
      await vi.advanceTimersByTimeAsync(100)
      await nextTick()

      const buttons = wrapper.findAll('button')
      const optOutButton = buttons.find((b) =>
        b.text().includes('nightlySurvey.dontAskAgain')
      )
      expect(optOutButton).toBeDefined()
      await optOutButton!.trigger('click')

      expect(wrapper.emitted('optedOut')).toHaveLength(1)
    })
  })

  describe('config', () => {
    it('uses custom delay from config', async () => {
      setFeatureUsage('test-feature', 5)

      const wrapper = await mountComponent({
        ...defaultConfig,
        delayMs: 500
      })
      await nextTick()

      await vi.advanceTimersByTimeAsync(400)
      await nextTick()
      expect(wrapper.find(POPOVER_SELECTOR).exists()).toBe(false)

      await vi.advanceTimersByTimeAsync(100)
      await nextTick()
      expect(wrapper.find(POPOVER_SELECTOR).exists()).toBe(true)
    })

    it('does not show when config is disabled', async () => {
      setFeatureUsage('test-feature', 5)

      const wrapper = await mountComponent({
        ...defaultConfig,
        enabled: false
      })
      await nextTick()
      await vi.advanceTimersByTimeAsync(1000)
      await nextTick()

      expect(wrapper.find(POPOVER_SELECTOR).exists()).toBe(false)
    })
  })
})
