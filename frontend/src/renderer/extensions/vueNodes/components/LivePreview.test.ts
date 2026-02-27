import { createTestingPinia } from '@pinia/testing'
import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import { createI18n } from 'vue-i18n'

import LivePreview from '@/renderer/extensions/vueNodes/components/LivePreview.vue'

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {
      g: {
        liveSamplingPreview: 'Live sampling preview',
        imageFailedToLoad: 'Image failed to load',
        errorLoadingImage: 'Error loading image',
        calculatingDimensions: 'Calculating dimensions'
      }
    }
  }
})

describe('LivePreview', () => {
  const defaultProps = {
    imageUrl: '/api/view?filename=test_sample.png&type=temp'
  }

  const mountLivePreview = (props = {}) => {
    return mount(LivePreview, {
      props: { ...defaultProps, ...props },
      global: {
        plugins: [
          createTestingPinia({
            createSpy: vi.fn
          }),
          i18n
        ],
        stubs: {
          'i-lucide:image-off': true
        }
      }
    })
  }

  it('renders preview when imageUrl provided', () => {
    const wrapper = mountLivePreview()

    expect(wrapper.find('img').exists()).toBe(true)
    expect(wrapper.find('img').attributes('src')).toBe(defaultProps.imageUrl)
  })

  it('does not render when no imageUrl provided', () => {
    const wrapper = mountLivePreview({ imageUrl: null })

    expect(wrapper.find('img').exists()).toBe(false)
    expect(wrapper.text()).toBe('')
  })

  it('displays calculating dimensions text initially', () => {
    const wrapper = mountLivePreview()

    expect(wrapper.text()).toContain('Calculating dimensions')
  })

  it('has proper accessibility attributes', () => {
    const wrapper = mountLivePreview()

    const img = wrapper.find('img')
    expect(img.attributes('alt')).toBe('Live sampling preview')
  })

  it('handles image load event', async () => {
    const wrapper = mountLivePreview()
    const img = wrapper.find('img')

    // Mock the naturalWidth and naturalHeight properties on the img element
    Object.defineProperty(img.element, 'naturalWidth', {
      writable: false,
      value: 512
    })
    Object.defineProperty(img.element, 'naturalHeight', {
      writable: false,
      value: 512
    })

    // Trigger the load event
    await img.trigger('load')

    expect(wrapper.text()).toContain('512 x 512')
  })

  it('handles image error state', async () => {
    const wrapper = mountLivePreview()
    const img = wrapper.find('img')

    // Trigger the error event
    await img.trigger('error')

    // Check that the image is hidden and error content is shown
    expect(wrapper.find('img').exists()).toBe(false)
    expect(wrapper.text()).toContain('Image failed to load')
  })

  it('resets state when imageUrl changes', async () => {
    const wrapper = mountLivePreview()
    const img = wrapper.find('img')

    // Set error state via event
    await img.trigger('error')
    expect(wrapper.text()).toContain('Error loading image')

    // Change imageUrl prop
    await wrapper.setProps({ imageUrl: '/new-image.png' })
    await nextTick()

    // State should be reset - dimensions text should show calculating
    expect(wrapper.text()).toContain('Calculating dimensions')
    expect(wrapper.text()).not.toContain('Error loading image')
  })

  it('shows error state when image fails to load', async () => {
    const wrapper = mountLivePreview()
    const img = wrapper.find('img')

    // Trigger error event
    await img.trigger('error')

    // Should show error state instead of image
    expect(wrapper.find('img').exists()).toBe(false)
    expect(wrapper.text()).toContain('Image failed to load')
    expect(wrapper.text()).toContain('Error loading image')
  })
})
