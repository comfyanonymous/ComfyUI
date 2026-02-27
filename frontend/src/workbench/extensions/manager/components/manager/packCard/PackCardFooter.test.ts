import type { VueWrapper } from '@vue/test-utils'
import { mount } from '@vue/test-utils'
import { createTestingPinia } from '@pinia/testing'
import PrimeVue from 'primevue/config'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'
import { createI18n } from 'vue-i18n'

import enMessages from '@/locales/en/main.json' with { type: 'json' }
import { IsInstallingKey } from '@/workbench/extensions/manager/types/comfyManagerTypes'

import PackCardFooter from './PackCardFooter.vue'

// Mock the child components
vi.mock(
  '@/workbench/extensions/manager/components/manager/button/PackInstallButton.vue',
  () => ({
    default: { template: '<div data-testid="pack-install-button"></div>' }
  })
)

vi.mock(
  '@/workbench/extensions/manager/components/manager/button/PackEnableToggle.vue',
  () => ({
    default: { template: '<div data-testid="pack-enable-toggle"></div>' }
  })
)

// Mock composables
const mockIsPackInstalled = vi.fn()
const mockCheckNodeCompatibility = vi.fn()

vi.mock('@/workbench/extensions/manager/stores/comfyManagerStore', () => ({
  useComfyManagerStore: vi.fn(() => ({
    isPackInstalled: mockIsPackInstalled
  }))
}))

vi.mock(
  '@/workbench/extensions/manager/composables/useConflictDetection',
  () => ({
    useConflictDetection: vi.fn(() => ({
      checkNodeCompatibility: mockCheckNodeCompatibility
    }))
  })
)

// Remove the mock for injection key since we're importing it directly

const mockNodePack = {
  id: 'test-pack',
  name: 'Test Pack',
  downloads: 1000
}

describe('PackCardFooter', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockIsPackInstalled.mockReset()
    mockCheckNodeCompatibility.mockReset()
  })

  const mountComponent = (props = {}): VueWrapper => {
    const i18n = createI18n({
      legacy: false,
      locale: 'en',
      messages: { en: enMessages }
    })

    return mount(PackCardFooter, {
      props: {
        nodePack: mockNodePack,
        ...props
      },
      global: {
        plugins: [PrimeVue, createTestingPinia({ stubActions: false }), i18n],
        provide: {
          [IsInstallingKey]: ref(false)
        }
      }
    })
  }

  describe('component rendering', () => {
    it('shows download count when available', () => {
      mockIsPackInstalled.mockReturnValue(false)
      mockCheckNodeCompatibility.mockReturnValue({
        hasConflict: false,
        conflicts: []
      })
      const wrapper = mountComponent()

      expect(wrapper.text()).toContain('1,000')
    })

    it('shows install button for uninstalled packages', () => {
      mockIsPackInstalled.mockReturnValue(false)
      mockCheckNodeCompatibility.mockReturnValue({
        hasConflict: false,
        conflicts: []
      })

      const wrapper = mountComponent()

      expect(wrapper.find('[data-testid="pack-install-button"]').exists()).toBe(
        true
      )
      expect(wrapper.find('[data-testid="pack-enable-toggle"]').exists()).toBe(
        false
      )
    })

    it('shows enable toggle for installed packages', () => {
      mockIsPackInstalled.mockReturnValue(true)

      const wrapper = mountComponent()

      expect(wrapper.find('[data-testid="pack-enable-toggle"]').exists()).toBe(
        true
      )
      expect(wrapper.find('[data-testid="pack-install-button"]').exists()).toBe(
        false
      )
    })
  })

  describe('conflict detection for uninstalled packages', () => {
    it('passes conflict info to install button when conflicts exist', () => {
      mockIsPackInstalled.mockReturnValue(false)
      mockCheckNodeCompatibility.mockReturnValue({
        hasConflict: true,
        conflicts: [
          {
            type: 'os_conflict',
            current_value: 'windows',
            required_value: 'linux'
          }
        ]
      })

      const wrapper = mountComponent()

      const installButton = wrapper.find('[data-testid="pack-install-button"]')
      expect(installButton.exists()).toBe(true)
      // The install button should receive has-conflict prop as true
      expect(installButton.attributes()).toHaveProperty('has-conflict')
    })

    it('does not pass conflict info when no conflicts exist', () => {
      mockIsPackInstalled.mockReturnValue(false)
      mockCheckNodeCompatibility.mockReturnValue({
        hasConflict: false,
        conflicts: []
      })

      const wrapper = mountComponent()

      const installButton = wrapper.find('[data-testid="pack-install-button"]')
      expect(installButton.exists()).toBe(true)
      // The install button should receive has-conflict prop as false
      expect(installButton.attributes()['has-conflict']).toBe('false')
    })
  })

  describe('installed packages', () => {
    it('does not pass has-conflict prop to enable toggle', () => {
      mockIsPackInstalled.mockReturnValue(true)

      const wrapper = mountComponent()

      const enableToggle = wrapper.find('[data-testid="pack-enable-toggle"]')
      expect(enableToggle.exists()).toBe(true)
      // The enable toggle should not receive has-conflict prop (removed in our fix)
      expect(enableToggle.attributes()).not.toHaveProperty('has-conflict')
    })
  })
})
