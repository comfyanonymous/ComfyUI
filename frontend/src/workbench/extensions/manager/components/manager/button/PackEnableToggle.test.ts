import type { VueWrapper } from '@vue/test-utils'
import { mount } from '@vue/test-utils'
import { createTestingPinia } from '@pinia/testing'
import PrimeVue from 'primevue/config'
import ToggleSwitch from 'primevue/toggleswitch'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import { createI18n } from 'vue-i18n'

import enMessages from '@/locales/en/main.json' with { type: 'json' }
import { useComfyManagerStore } from '@/workbench/extensions/manager/stores/comfyManagerStore'

import PackEnableToggle from './PackEnableToggle.vue'

// Mock debounce to execute immediately
vi.mock('es-toolkit/compat', async () => {
  const actual = await vi.importActual('es-toolkit/compat')
  return {
    ...actual,
    debounce: <T extends (...args: unknown[]) => unknown>(fn: T) => fn
  }
})

const mockNodePack = {
  id: 'test-pack',
  name: 'Test Pack',
  latest_version: {
    version: '1.0.0',
    createdAt: '2023-01-01T00:00:00Z'
  }
}

const mockIsPackEnabled = vi.fn()
const mockEnablePack = vi.fn().mockResolvedValue(undefined)
const mockDisablePack = vi.fn().mockResolvedValue(undefined)
const mockGetConflictsForPackageByID = vi.fn()

vi.mock('@/workbench/extensions/manager/stores/comfyManagerStore', () => ({
  useComfyManagerStore: vi.fn(() => ({
    isPackEnabled: mockIsPackEnabled,
    enablePack: mockEnablePack,
    disablePack: mockDisablePack,
    installedPacks: {}
  }))
}))

vi.mock('@/workbench/extensions/manager/stores/conflictDetectionStore', () => ({
  useConflictDetectionStore: vi.fn(() => ({
    getConflictsForPackageByID: mockGetConflictsForPackageByID
  }))
}))

describe('PackEnableToggle', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockIsPackEnabled.mockReset()
    mockEnablePack.mockReset().mockResolvedValue(undefined)
    mockDisablePack.mockReset().mockResolvedValue(undefined)
  })

  const mountComponent = ({
    props = {},
    installedPacks = {}
  }: {
    props?: Record<string, unknown>
    installedPacks?: Record<string, unknown>
  } = {}): VueWrapper => {
    const i18n = createI18n({
      legacy: false,
      locale: 'en',
      messages: { en: enMessages }
    })

    vi.mocked(useComfyManagerStore).mockReturnValue({
      isPackEnabled: mockIsPackEnabled,
      enablePack: mockEnablePack,
      disablePack: mockDisablePack,
      installedPacks
    } as Partial<ReturnType<typeof useComfyManagerStore>> as ReturnType<
      typeof useComfyManagerStore
    >)

    return mount(PackEnableToggle, {
      props: {
        nodePack: mockNodePack,
        ...props
      },
      global: {
        plugins: [PrimeVue, createTestingPinia({ stubActions: false }), i18n]
      }
    })
  }

  it('renders a toggle switch', () => {
    mockIsPackEnabled.mockReturnValue(true)
    const wrapper = mountComponent()

    const toggleSwitch = wrapper.findComponent(ToggleSwitch)
    expect(toggleSwitch.exists()).toBe(true)
  })

  it('checks if pack is enabled on mount', () => {
    mockIsPackEnabled.mockReturnValue(true)
    mountComponent()

    expect(mockIsPackEnabled).toHaveBeenCalledWith(mockNodePack.id)
  })

  it('sets toggle to on when pack is enabled', () => {
    mockIsPackEnabled.mockReturnValue(true)
    const wrapper = mountComponent()

    const toggleSwitch = wrapper.findComponent(ToggleSwitch)
    expect(toggleSwitch.props('modelValue')).toBe(true)
  })

  it('sets toggle to off when pack is disabled', () => {
    mockIsPackEnabled.mockReturnValue(false)
    const wrapper = mountComponent()

    const toggleSwitch = wrapper.findComponent(ToggleSwitch)
    expect(toggleSwitch.props('modelValue')).toBe(false)
  })

  it('calls enablePack when toggle is switched on', async () => {
    mockIsPackEnabled.mockReturnValue(false)
    const wrapper = mountComponent()

    const toggleSwitch = wrapper.findComponent(ToggleSwitch)
    await toggleSwitch.vm.$emit('update:modelValue', true)

    expect(mockEnablePack).toHaveBeenCalledWith(
      expect.objectContaining({
        id: mockNodePack.id,
        version: mockNodePack.latest_version.version
      })
    )
  })

  it('calls disablePack when toggle is switched off', async () => {
    mockIsPackEnabled.mockReturnValue(true)
    const wrapper = mountComponent()

    const toggleSwitch = wrapper.findComponent(ToggleSwitch)
    await toggleSwitch.vm.$emit('update:modelValue', false)

    expect(mockDisablePack).toHaveBeenCalledWith(
      expect.objectContaining({
        id: mockNodePack.id,
        version: mockNodePack.latest_version.version
      })
    )
  })

  it('disables toggle while loading', async () => {
    const pendingPromise = new Promise<void>((resolve) => {
      setTimeout(() => resolve(), 1000)
    })
    mockEnablePack.mockReturnValue(pendingPromise)

    mockIsPackEnabled.mockReturnValue(false)
    const wrapper = mountComponent()

    // Trigger the toggle
    const toggleSwitch = wrapper.findComponent(ToggleSwitch)
    await toggleSwitch.vm.$emit('update:modelValue', true)

    // Check that the toggle is disabled during loading
    await nextTick()
    expect(wrapper.findComponent(ToggleSwitch).props('disabled')).toBe(true)

    // Resolve the promise to simulate the operation completing
    await pendingPromise

    // Check that the toggle is enabled after the operation completes
    await nextTick()
    expect(wrapper.findComponent(ToggleSwitch).props('disabled')).toBe(false)
  })

  describe('conflict warning icon', () => {
    it('should show warning icon when package has conflicts', () => {
      mockGetConflictsForPackageByID.mockReturnValue({
        package_id: 'test-pack',
        package_name: 'Test Pack',
        has_conflict: true,
        conflicts: [
          {
            type: 'import_failed',
            current_value: 'installed',
            required_value: 'error message'
          }
        ],
        is_compatible: false
      })

      mockIsPackEnabled.mockReturnValue(true)
      const wrapper = mountComponent()

      // Check if warning icon exists
      const warningIcon = wrapper.find('.icon-\\[lucide--triangle-alert\\]')
      expect(warningIcon.exists()).toBe(true)
      expect(warningIcon.classes()).toContain('text-warning-background')
    })

    it('should not show warning icon when package has no conflicts', () => {
      mockGetConflictsForPackageByID.mockReturnValue(null)

      mockIsPackEnabled.mockReturnValue(true)
      const wrapper = mountComponent()

      // Check if warning icon does not exist
      const warningIcon = wrapper.find('.icon-\\[lucide--triangle-alert\\]')
      expect(warningIcon.exists()).toBe(false)
    })
  })
})
