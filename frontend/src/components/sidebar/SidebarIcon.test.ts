import { mount } from '@vue/test-utils'
import PrimeVue from 'primevue/config'
import Tooltip from 'primevue/tooltip'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'

import SidebarIcon from './SidebarIcon.vue'

type SidebarIconProps = {
  icon: string
  selected: boolean
  tooltip?: string
  class?: string
  iconBadge?: string | (() => string | null)
}

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {}
  }
})

describe('SidebarIcon', () => {
  const exampleProps: SidebarIconProps = {
    icon: 'pi pi-cog',
    selected: false
  }

  const mountSidebarIcon = (props: Partial<SidebarIconProps>, options = {}) => {
    return mount(SidebarIcon, {
      global: {
        plugins: [PrimeVue, i18n],
        directives: { tooltip: Tooltip }
      },
      props: { ...exampleProps, ...props },
      ...options
    })
  }

  it('renders button element', () => {
    const wrapper = mountSidebarIcon({})
    expect(wrapper.find('button.side-bar-button').exists()).toBe(true)
  })

  it('renders icon', () => {
    const wrapper = mountSidebarIcon({})
    expect(wrapper.find('.side-bar-button-icon').exists()).toBe(true)
  })

  it('creates badge when iconBadge prop is set', () => {
    const badge = '2'
    const wrapper = mountSidebarIcon({ iconBadge: badge })
    const badgeEl = wrapper.find('.sidebar-icon-badge')
    expect(badgeEl.exists()).toBe(true)
    expect(badgeEl.text()).toEqual(badge)
  })

  it('shows tooltip on hover', async () => {
    const tooltipShowDelay = 300
    const tooltipText = 'Settings'
    const wrapper = mountSidebarIcon({ tooltip: tooltipText })

    const tooltipElBeforeHover = document.querySelector('[role="tooltip"]')
    expect(tooltipElBeforeHover).toBeNull()

    // Hover over the icon
    await wrapper.trigger('mouseenter')
    await new Promise((resolve) => setTimeout(resolve, tooltipShowDelay + 16))

    const tooltipElAfterHover = document.querySelector('[role="tooltip"]')
    expect(tooltipElAfterHover).not.toBeNull()
  })

  it('sets aria-label attribute when tooltip is provided', () => {
    const tooltipText = 'Settings'
    const wrapper = mountSidebarIcon({ tooltip: tooltipText })
    expect(wrapper.attributes('aria-label')).toEqual(tooltipText)
  })
})
