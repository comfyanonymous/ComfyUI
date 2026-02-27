import type { ComponentProps } from 'vue-component-type-helpers'

import { mount } from '@vue/test-utils'
import Avatar from 'primevue/avatar'
import PrimeVue from 'primevue/config'
import { beforeEach, describe, expect, it } from 'vitest'
import { createApp, nextTick } from 'vue'
import { createI18n } from 'vue-i18n'

import UserAvatar from './UserAvatar.vue'

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {
      auth: {
        login: {
          userAvatar: 'User Avatar'
        }
      }
    }
  }
})

describe('UserAvatar', () => {
  beforeEach(() => {
    const app = createApp({})
    app.use(PrimeVue)
  })

  const mountComponent = (props: ComponentProps<typeof UserAvatar> = {}) => {
    return mount(UserAvatar, {
      global: {
        plugins: [PrimeVue, i18n],
        components: { Avatar }
      },
      props
    })
  }

  it('renders correctly with photo Url', async () => {
    const wrapper = mountComponent({
      photoUrl: 'https://example.com/avatar.jpg'
    })

    const avatar = wrapper.findComponent(Avatar)
    expect(avatar.exists()).toBe(true)
    expect(avatar.props('image')).toBe('https://example.com/avatar.jpg')
    expect(avatar.props('icon')).toBeNull()
  })

  it('renders with default icon when no photo Url is provided', () => {
    const wrapper = mountComponent({
      photoUrl: undefined
    })

    const avatar = wrapper.findComponent(Avatar)
    expect(avatar.exists()).toBe(true)
    expect(avatar.props('image')).toBeNull()
    expect(avatar.props('icon')).toBe('icon-[lucide--user]')
  })

  it('renders with default icon when provided photo Url is null', () => {
    const wrapper = mountComponent({
      photoUrl: null
    })

    const avatar = wrapper.findComponent(Avatar)
    expect(avatar.exists()).toBe(true)
    expect(avatar.props('image')).toBeNull()
    expect(avatar.props('icon')).toBe('icon-[lucide--user]')
  })

  it('falls back to icon when image fails to load', async () => {
    const wrapper = mountComponent({
      photoUrl: 'https://example.com/broken-image.jpg'
    })

    const avatar = wrapper.findComponent(Avatar)
    expect(avatar.props('icon')).toBeNull()

    // Simulate image load error
    avatar.vm.$emit('error')
    await nextTick()

    expect(avatar.props('icon')).toBe('icon-[lucide--user]')
  })

  it('uses provided ariaLabel', () => {
    const wrapper = mountComponent({
      photoUrl: 'https://example.com/avatar.jpg',
      ariaLabel: 'Custom Label'
    })

    const avatar = wrapper.findComponent(Avatar)
    expect(avatar.attributes('aria-label')).toBe('Custom Label')
  })

  it('falls back to i18n translation when no ariaLabel is provided', () => {
    const wrapper = mountComponent({
      photoUrl: 'https://example.com/avatar.jpg'
    })

    const avatar = wrapper.findComponent(Avatar)
    expect(avatar.attributes('aria-label')).toBe('User Avatar')
  })
})
