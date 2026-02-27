import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

import CompareSliderThumbnail from '@/components/templates/thumbnails/CompareSliderThumbnail.vue'

vi.mock('@/components/templates/thumbnails/BaseThumbnail.vue', () => ({
  default: {
    name: 'BaseThumbnail',
    template: '<div class="base-thumbnail"><slot /></div>',
    props: ['isHovered']
  }
}))

vi.mock('@/components/common/LazyImage.vue', () => ({
  default: {
    name: 'LazyImage',
    template:
      '<img :src="src" :alt="alt" :class="imageClass" :style="imageStyle" draggable="false" />',
    props: ['src', 'alt', 'imageClass', 'imageStyle']
  }
}))

vi.mock('@vueuse/core', () => ({
  useMouseInElement: () => ({
    elementX: ref(50),
    elementWidth: ref(100),
    isOutside: ref(false)
  })
}))

describe('CompareSliderThumbnail', () => {
  const mountThumbnail = (props = {}) => {
    return mount(CompareSliderThumbnail, {
      props: {
        baseImageSrc: '/base-image.jpg',
        overlayImageSrc: '/overlay-image.jpg',
        alt: 'Comparison Image',
        isVideo: false,
        ...props
      }
    })
  }

  it('renders both base and overlay images', () => {
    const wrapper = mountThumbnail()
    const lazyImages = wrapper.findAllComponents({ name: 'LazyImage' })
    expect(lazyImages.length).toBe(2)
    expect(lazyImages[0].props('src')).toBe('/base-image.jpg')
    expect(lazyImages[1].props('src')).toBe('/overlay-image.jpg')
  })

  it('applies correct alt text to both images', () => {
    const wrapper = mountThumbnail({ alt: 'Custom Alt Text' })
    const lazyImages = wrapper.findAllComponents({ name: 'LazyImage' })
    expect(lazyImages[0].props('alt')).toBe('Custom Alt Text')
    expect(lazyImages[1].props('alt')).toBe('Custom Alt Text')
  })

  it('applies clip-path style to overlay image', () => {
    const wrapper = mountThumbnail()
    const overlayLazyImage = wrapper.findAllComponents({ name: 'LazyImage' })[1]
    const imageStyle = overlayLazyImage.props('imageStyle')
    expect(imageStyle.clipPath).toContain('inset')
  })

  it('renders slider divider', () => {
    const wrapper = mountThumbnail()
    const divider = wrapper.find('.bg-white\\/30')
    expect(divider.exists()).toBe(true)
  })

  it('positions slider based on default value', () => {
    const wrapper = mountThumbnail()
    const divider = wrapper.find('.bg-white\\/30')
    expect(divider.attributes('style')).toContain('left: 50%')
  })

  it('passes isHovered prop to BaseThumbnail', () => {
    const wrapper = mountThumbnail({ isHovered: true })
    const baseThumbnail = wrapper.findComponent({ name: 'BaseThumbnail' })
    expect(baseThumbnail.props('isHovered')).toBe(true)
  })
})
