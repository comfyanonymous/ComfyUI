import type { VueWrapper } from '@vue/test-utils'
import { mount } from '@vue/test-utils'
import { createTestingPinia } from '@pinia/testing'
import PrimeVue from 'primevue/config'
import Tooltip from 'primevue/tooltip'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import { createI18n } from 'vue-i18n'

import enMessages from '@/locales/en/main.json' with { type: 'json' }

import PackVersionBadge from './PackVersionBadge.vue'
import PackVersionSelectorPopover from './PackVersionSelectorPopover.vue'

// Mock config to prevent __COMFYUI_FRONTEND_VERSION__ error
vi.mock('@/config', () => ({
  default: {
    app_title: 'ComfyUI',
    app_version: '1.0.0'
  }
}))

const mockNodePack = {
  id: 'test-pack',
  name: 'Test Pack',
  latest_version: {
    version: '1.0.0'
  }
}

const mockInstalledPacks = {
  'test-pack': { ver: '1.5.0' },
  'installed-pack': { ver: '2.0.0' }
}

const mockIsPackEnabled = vi.fn(() => true)

vi.mock('@/workbench/extensions/manager/stores/comfyManagerStore', () => ({
  useComfyManagerStore: vi.fn(() => ({
    installedPacks: mockInstalledPacks,
    isPackInstalled: (id: string) =>
      !!mockInstalledPacks[id as keyof typeof mockInstalledPacks],
    isPackEnabled: mockIsPackEnabled,
    getInstalledPackVersion: (id: string) =>
      mockInstalledPacks[id as keyof typeof mockInstalledPacks]?.ver
  }))
}))

vi.mock(
  '@/workbench/extensions/manager/composables/nodePack/usePackUpdateStatus',
  () => ({
    usePackUpdateStatus: vi.fn(() => ({
      isUpdateAvailable: false
    }))
  })
)

const mockToggle = vi.fn()
const mockHide = vi.fn()
const PopoverStub = {
  name: 'Popover',
  template: '<div><slot></slot></div>',
  methods: {
    toggle: mockToggle,
    hide: mockHide
  }
}

describe('PackVersionBadge', () => {
  beforeEach(() => {
    mockToggle.mockReset()
    mockHide.mockReset()
    mockIsPackEnabled.mockReturnValue(true) // Reset to default enabled state
  })

  const mountComponent = ({
    props = {}
  }: { props?: Record<string, unknown> } = {}): VueWrapper => {
    const i18n = createI18n({
      legacy: false,
      locale: 'en',
      messages: { en: enMessages }
    })

    return mount(PackVersionBadge, {
      props: {
        nodePack: mockNodePack,
        isSelected: false,
        ...props
      },
      global: {
        plugins: [PrimeVue, createTestingPinia({ stubActions: false }), i18n],
        directives: {
          tooltip: Tooltip
        },
        stubs: {
          Popover: PopoverStub,
          PackVersionSelectorPopover: true
        }
      }
    })
  }

  it('renders with installed version from store', () => {
    const wrapper = mountComponent()

    const badge = wrapper.find('[role="button"]')
    expect(badge.exists()).toBe(true)
    expect(badge.find('span').text()).toBe('1.5.0') // From mockInstalledPacks
  })

  it('falls back to latest_version when not installed', () => {
    // Use a nodePack that's not in the installedPacks
    const uninstalledPack = {
      id: 'uninstalled-pack',
      name: 'Uninstalled Pack',
      latest_version: {
        version: '3.0.0'
      }
    }

    const wrapper = mountComponent({
      props: { nodePack: uninstalledPack }
    })

    const badge = wrapper.find('[role="button"]')
    expect(badge.exists()).toBe(true)
    expect(badge.find('span').text()).toBe('3.0.0') // From latest_version
  })

  it('falls back to NIGHTLY when no latest_version and not installed', () => {
    // Use a nodePack with no latest_version and not in installedPacks
    const noVersionPack = {
      id: 'no-version-pack',
      name: 'No Version Pack'
    }

    const wrapper = mountComponent({
      props: { nodePack: noVersionPack }
    })

    const badge = wrapper.find('[role="button"]')
    expect(badge.exists()).toBe(true)
    expect(badge.find('span').text()).toBe('nightly')
  })

  it('falls back to NIGHTLY when nodePack.id is missing', () => {
    const invalidPack = {
      name: 'Invalid Pack'
    }

    const wrapper = mountComponent({
      props: { nodePack: invalidPack }
    })

    const badge = wrapper.find('[role="button"]')
    expect(badge.exists()).toBe(true)
    expect(badge.find('span').text()).toBe('nightly')
  })

  it('toggles the popover when button is clicked', async () => {
    const wrapper = mountComponent()

    // Click the badge
    await wrapper.find('[role="button"]').trigger('click')

    // Verify that the toggle method was called
    expect(mockToggle).toHaveBeenCalled()
  })

  it('closes the popover when cancel is emitted', async () => {
    const wrapper = mountComponent()

    // Simulate the popover emitting a cancel event
    wrapper.findComponent(PackVersionSelectorPopover).vm.$emit('cancel')
    await nextTick()

    // Verify that the hide method was called
    expect(mockHide).toHaveBeenCalled()
  })

  it('closes the popover when submit is emitted', async () => {
    const wrapper = mountComponent()

    // Simulate the popover emitting a submit event
    wrapper.findComponent(PackVersionSelectorPopover).vm.$emit('submit')
    await nextTick()

    // Verify that the hide method was called
    expect(mockHide).toHaveBeenCalled()
  })

  describe('selection state changes', () => {
    it('closes the popover when card is deselected', async () => {
      const wrapper = mountComponent({
        props: { isSelected: true }
      })

      // Change isSelected from true to false
      await wrapper.setProps({ isSelected: false })
      await nextTick()

      // Verify that the hide method was called
      expect(mockHide).toHaveBeenCalled()
    })

    it('does not close the popover when card is selected', async () => {
      const wrapper = mountComponent({
        props: { isSelected: false }
      })

      // Change isSelected from false to true
      await wrapper.setProps({ isSelected: true })
      await nextTick()

      // Verify that the hide method was NOT called
      expect(mockHide).not.toHaveBeenCalled()
    })

    it('does not close the popover when isSelected remains false', async () => {
      const wrapper = mountComponent({
        props: { isSelected: false }
      })

      // Change isSelected from false to false (no change)
      await wrapper.setProps({ isSelected: false })
      await nextTick()

      // Verify that the hide method was NOT called
      expect(mockHide).not.toHaveBeenCalled()
    })

    it('does not close the popover when isSelected remains true', async () => {
      const wrapper = mountComponent({
        props: { isSelected: true }
      })

      // Change isSelected from true to true (no change)
      await wrapper.setProps({ isSelected: true })
      await nextTick()

      // Verify that the hide method was NOT called
      expect(mockHide).not.toHaveBeenCalled()
    })
  })

  describe('disabled state', () => {
    beforeEach(() => {
      mockIsPackEnabled.mockReturnValue(false) // Set all packs as disabled for these tests
    })

    it('adds disabled styles when pack is disabled', () => {
      const wrapper = mountComponent()

      const badge = wrapper.find('[role="text"]') // role changes to "text" when disabled
      expect(badge.exists()).toBe(true)
      expect(badge.classes()).toContain('cursor-not-allowed')
      expect(badge.classes()).toContain('opacity-60')
    })

    it('does not show chevron icon when disabled', () => {
      const wrapper = mountComponent()

      const chevronIcon = wrapper.find('.pi-chevron-right')
      expect(chevronIcon.exists()).toBe(false)
    })

    it('does not show update arrow when disabled', () => {
      const wrapper = mountComponent()

      const updateIcon = wrapper.find('.pi-arrow-circle-up')
      expect(updateIcon.exists()).toBe(false)
    })

    it('does not toggle popover when clicked while disabled', async () => {
      const wrapper = mountComponent()

      const badge = wrapper.find('[role="text"]') // role changes to "text" when disabled
      expect(badge.exists()).toBe(true)
      await badge.trigger('click')

      // Since it's disabled, the popover should not be toggled
      expect(mockToggle).not.toHaveBeenCalled()
    })

    it('has correct tabindex when disabled', () => {
      const wrapper = mountComponent()

      const badge = wrapper.find('[role="text"]') // role changes to "text" when disabled
      expect(badge.exists()).toBe(true)
      expect(badge.attributes('tabindex')).toBe('-1')
    })

    it('does not respond to keyboard events when disabled', async () => {
      const wrapper = mountComponent()

      const badge = wrapper.find('[role="text"]') // role changes to "text" when disabled
      expect(badge.exists()).toBe(true)
      await badge.trigger('keydown.enter')
      await badge.trigger('keydown.space')

      expect(mockToggle).not.toHaveBeenCalled()
    })
  })
})
