import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

import HoverDissolveThumbnail from '@/components/templates/thumbnails/HoverDissolveThumbnail.vue'

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

describe('HoverDissolveThumbnail', () => {
  const mountThumbnail = (props = {}) => {
    return mount(HoverDissolveThumbnail, {
      props: {
        baseImageSrc: '/base-image.jpg',
        overlayImageSrc: '/overlay-image.jpg',
        alt: 'Dissolve Image',
        isHovered: false,
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

  it('makes overlay image visible when hovered', () => {
    const wrapper = mountThumbnail({ isHovered: true })
    const overlayLazyImage = wrapper.findAllComponents({ name: 'LazyImage' })[1]
    const imageClass = overlayLazyImage.props('imageClass')
    const classString = Array.isArray(imageClass)
      ? imageClass.join(' ')
      : imageClass
    expect(classString).toContain('opacity-100')
    expect(classString).not.toContain('opacity-0')
  })

  it('makes overlay image hidden when not hovered', () => {
    const wrapper = mountThumbnail({ isHovered: false })
    const overlayLazyImage = wrapper.findAllComponents({ name: 'LazyImage' })[1]
    const imageClass = overlayLazyImage.props('imageClass')
    const classString = Array.isArray(imageClass)
      ? imageClass.join(' ')
      : imageClass
    expect(classString).toContain('opacity-0')
    expect(classString).not.toContain('opacity-100')
  })

  it('passes isHovered prop to BaseThumbnail', () => {
    const wrapper = mountThumbnail({ isHovered: true })
    const baseThumbnail = wrapper.findComponent({ name: 'BaseThumbnail' })
    expect(baseThumbnail.props('isHovered')).toBe(true)
  })

  it('applies transition classes to overlay image', () => {
    const wrapper = mountThumbnail()
    const overlayLazyImage = wrapper.findAllComponents({ name: 'LazyImage' })[1]
    const imageClass = overlayLazyImage.props('imageClass')
    const classString = Array.isArray(imageClass)
      ? imageClass.join(' ')
      : imageClass
    expect(classString).toContain('transition-opacity')
    expect(classString).toContain('duration-300')
  })

  it('applies correct positioning to both images', () => {
    const wrapper = mountThumbnail()
    const lazyImages = wrapper.findAllComponents({ name: 'LazyImage' })

    // Check base image
    const baseImageClass = lazyImages[0].props('imageClass')
    const baseClassList = Array.isArray(baseImageClass)
      ? baseImageClass
      : baseImageClass.split(' ')
    expect(baseClassList).toContain('size-full')

    // Check overlay image
    const overlayImageClass = lazyImages[1].props('imageClass')
    const overlayClassList = Array.isArray(overlayImageClass)
      ? overlayImageClass
      : overlayImageClass.split(' ')
    expect(overlayClassList).toContain('size-full')
  })
})
