import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

import BaseThumbnail from '@/components/templates/thumbnails/BaseThumbnail.vue'

type ComponentInstance = InstanceType<typeof BaseThumbnail> & {
  error: boolean
}

vi.mock('@vueuse/core', () => ({
  useEventListener: vi.fn()
}))

describe('BaseThumbnail', () => {
  const mountThumbnail = (props = {}, slots = {}) => {
    return mount(BaseThumbnail, {
      props,
      slots: {
        default: '<img src="/test.jpg" alt="test" />',
        ...slots
      }
    })
  }

  it('renders slot content', () => {
    const wrapper = mountThumbnail()
    expect(wrapper.find('img').exists()).toBe(true)
  })

  it('applies hover zoom with correct style', () => {
    const wrapper = mountThumbnail({ isHovered: true })
    const contentDiv = wrapper.find('.transform-gpu')
    expect(contentDiv.attributes('style')).toContain('transform')
    expect(contentDiv.attributes('style')).toContain('scale')
  })

  it('applies custom hover zoom value', () => {
    const wrapper = mountThumbnail({ hoverZoom: 10, isHovered: true })
    const contentDiv = wrapper.find('.transform-gpu')
    expect(contentDiv.attributes('style')).toContain('scale(1.1)')
  })

  it('does not apply scale when not hovered', () => {
    const wrapper = mountThumbnail({ isHovered: false })
    const contentDiv = wrapper.find('.transform-gpu')
    expect(contentDiv.attributes('style')).toBeUndefined()
  })

  it('shows error state when image fails to load', async () => {
    const wrapper = mountThumbnail()
    const vm = wrapper.vm as ComponentInstance

    // Manually set error since useEventListener is mocked
    vm.error = true
    await nextTick()

    expect(
      wrapper.find('img[src="/assets/images/default-template.png"]').exists()
    ).toBe(true)
  })

  it('applies transition classes to content', () => {
    const wrapper = mountThumbnail()
    const contentDiv = wrapper.find('.transform-gpu')
    expect(contentDiv.classes()).toContain('transform-gpu')
    expect(contentDiv.classes()).toContain('transition-transform')
    expect(contentDiv.classes()).toContain('duration-1000')
    expect(contentDiv.classes()).toContain('ease-out')
  })
})
