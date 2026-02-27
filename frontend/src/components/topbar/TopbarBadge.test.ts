import { mount } from '@vue/test-utils'
import Popover from 'primevue/popover'
import PrimeVue from 'primevue/config'
import Tooltip from 'primevue/tooltip'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'

import type { TopbarBadge as TopbarBadgeType } from '@/types/comfy'

import TopbarBadge from './TopbarBadge.vue'

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {}
  }
})

describe('TopbarBadge', () => {
  const exampleBadge: TopbarBadgeType = {
    text: 'Test Badge',
    label: 'BETA',
    variant: 'info'
  }

  const mountTopbarBadge = (
    badge: Partial<TopbarBadgeType> = {},
    displayMode: 'full' | 'compact' | 'icon-only' = 'full'
  ) => {
    return mount(TopbarBadge, {
      global: {
        plugins: [PrimeVue, i18n],
        directives: { tooltip: Tooltip },
        components: { Popover }
      },
      props: {
        badge: { ...exampleBadge, ...badge },
        displayMode
      }
    })
  }

  describe('full display mode', () => {
    it('renders all badge elements (icon, label, text)', () => {
      const wrapper = mountTopbarBadge(
        {
          text: 'Comfy Cloud',
          label: 'BETA',
          icon: 'pi pi-cloud'
        },
        'full'
      )

      expect(wrapper.find('.pi-cloud').exists()).toBe(true)
      expect(wrapper.text()).toContain('BETA')
      expect(wrapper.text()).toContain('Comfy Cloud')
    })

    it('renders without icon when not provided', () => {
      const wrapper = mountTopbarBadge(
        {
          text: 'Test',
          label: 'NEW'
        },
        'full'
      )

      expect(wrapper.find('i').exists()).toBe(false)
      expect(wrapper.text()).toContain('NEW')
      expect(wrapper.text()).toContain('Test')
    })
  })

  describe('compact display mode', () => {
    it('renders icon and label but not text', () => {
      const wrapper = mountTopbarBadge(
        {
          text: 'Hidden Text',
          label: 'BETA',
          icon: 'pi pi-cloud'
        },
        'compact'
      )

      expect(wrapper.find('.pi-cloud').exists()).toBe(true)
      expect(wrapper.text()).toContain('BETA')
      expect(wrapper.text()).not.toContain('Hidden Text')
    })

    it('opens popover on click', async () => {
      const wrapper = mountTopbarBadge(
        {
          text: 'Full Text',
          label: 'ALERT'
        },
        'compact'
      )

      const clickableArea = wrapper.find('[class*="flex h-full"]')
      await clickableArea.trigger('click')

      const popover = wrapper.findComponent(Popover)
      expect(popover.exists()).toBe(true)
    })
  })

  describe('icon-only display mode', () => {
    it('renders only icon', () => {
      const wrapper = mountTopbarBadge(
        {
          text: 'Hidden Text',
          label: 'BETA',
          icon: 'pi pi-cloud'
        },
        'icon-only'
      )

      expect(wrapper.find('.pi-cloud').exists()).toBe(true)
      expect(wrapper.text()).not.toContain('BETA')
      expect(wrapper.text()).not.toContain('Hidden Text')
    })

    it('renders label when no icon provided', () => {
      const wrapper = mountTopbarBadge(
        {
          text: 'Hidden Text',
          label: 'NEW'
        },
        'icon-only'
      )

      expect(wrapper.text()).toContain('NEW')
      expect(wrapper.text()).not.toContain('Hidden Text')
    })
  })

  describe('badge variants', () => {
    it('applies error variant styles', () => {
      const wrapper = mountTopbarBadge(
        {
          text: 'Error Message',
          label: 'ERROR',
          variant: 'error'
        },
        'full'
      )

      expect(wrapper.find('.bg-danger-100').exists()).toBe(true)
      expect(wrapper.find('.text-danger-100').exists()).toBe(true)
    })

    it('applies warning variant styles', () => {
      const wrapper = mountTopbarBadge(
        {
          text: 'Warning Message',
          label: 'WARN',
          variant: 'warning'
        },
        'full'
      )

      expect(wrapper.find('.bg-gold-600').exists()).toBe(true)
      expect(wrapper.find('.text-warning-background').exists()).toBe(true)
    })

    it('uses default error icon for error variant', () => {
      const wrapper = mountTopbarBadge(
        {
          text: 'Error',
          variant: 'error'
        },
        'full'
      )

      expect(wrapper.find('.pi-exclamation-circle').exists()).toBe(true)
    })

    it('uses default warning icon for warning variant', () => {
      const wrapper = mountTopbarBadge(
        {
          text: 'Warning',
          variant: 'warning'
        },
        'full'
      )

      expect(wrapper.find('.icon-\\[lucide--triangle-alert\\]').exists()).toBe(
        true
      )
    })
  })

  describe('popover', () => {
    it('includes popover component in compact and icon-only modes', () => {
      const compactWrapper = mountTopbarBadge({}, 'compact')
      const iconOnlyWrapper = mountTopbarBadge({}, 'icon-only')
      const fullWrapper = mountTopbarBadge({}, 'full')

      expect(compactWrapper.findComponent(Popover).exists()).toBe(true)
      expect(iconOnlyWrapper.findComponent(Popover).exists()).toBe(true)
      expect(fullWrapper.findComponent(Popover).exists()).toBe(false)
    })
  })

  describe('edge cases', () => {
    it('handles badge with only text', () => {
      const wrapper = mountTopbarBadge(
        {
          text: 'Simple Badge'
        },
        'full'
      )

      expect(wrapper.text()).toContain('Simple Badge')
      expect(wrapper.find('i').exists()).toBe(false)
    })

    it('handles custom icon override', () => {
      const wrapper = mountTopbarBadge(
        {
          text: 'Custom',
          variant: 'error',
          icon: 'pi pi-custom-icon'
        },
        'full'
      )

      expect(wrapper.find('.pi-custom-icon').exists()).toBe(true)
      expect(wrapper.find('.pi-exclamation-circle').exists()).toBe(false)
    })
  })
})
