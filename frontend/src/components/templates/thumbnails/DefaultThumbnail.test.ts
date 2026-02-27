import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

import DefaultThumbnail from '@/components/templates/thumbnails/DefaultThumbnail.vue'

vi.mock('@/components/templates/thumbnails/BaseThumbnail.vue', () => ({
  default: {
    name: 'BaseThumbnail',
    template: '<div class="base-thumbnail"><slot /></div>',
    props: ['hoverZoom', 'isHovered']
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

describe('DefaultThumbnail', () => {
  const mountThumbnail = (props = {}) => {
    return mount(DefaultThumbnail, {
      props: {
        src: '/test-image.jpg',
        alt: 'Test Image',
        hoverZoom: 5,
        ...props
      }
    })
  }

  it('renders image with correct src and alt', () => {
    const wrapper = mountThumbnail()
    const lazyImage = wrapper.findComponent({ name: 'LazyImage' })
    expect(lazyImage.props('src')).toBe('/test-image.jpg')
    expect(lazyImage.props('alt')).toBe('Test Image')
  })

  it('applies scale transform when hovered', () => {
    const wrapper = mountThumbnail({
      isHovered: true,
      hoverZoom: 10
    })
    const lazyImage = wrapper.findComponent({ name: 'LazyImage' })
    expect(lazyImage.props('imageStyle')).toEqual({ transform: 'scale(1.1)' })
  })

  it('does not apply scale transform when not hovered', () => {
    const wrapper = mountThumbnail({
      isHovered: false
    })
    const lazyImage = wrapper.findComponent({ name: 'LazyImage' })
    expect(lazyImage.props('imageStyle')).toBeUndefined()
  })

  it('applies video styling for video type', () => {
    const wrapper = mountThumbnail({
      isVideo: true
    })
    const lazyImage = wrapper.findComponent({ name: 'LazyImage' })
    const imageClass = lazyImage.props('imageClass')
    const classString = Array.isArray(imageClass)
      ? imageClass.join(' ')
      : imageClass
    expect(classString).toContain('w-full')
    expect(classString).toContain('h-full')
    expect(classString).toContain('object-cover')
  })

  it('applies image styling for non-video type', () => {
    const wrapper = mountThumbnail({
      isVideo: false
    })
    const lazyImage = wrapper.findComponent({ name: 'LazyImage' })
    const imageClass = lazyImage.props('imageClass')
    const classString = Array.isArray(imageClass)
      ? imageClass.join(' ')
      : imageClass
    expect(classString).toContain('max-w-full')
    expect(classString).toContain('object-contain')
  })

  it('applies correct styling for webp images', () => {
    const wrapper = mountThumbnail({
      src: '/test-video.webp',
      isVideo: true
    })
    const lazyImage = wrapper.findComponent({ name: 'LazyImage' })
    const imageClass = lazyImage.props('imageClass')
    const classString = Array.isArray(imageClass)
      ? imageClass.join(' ')
      : imageClass
    expect(classString).toContain('object-cover')
  })

  it('image is not draggable', () => {
    const wrapper = mountThumbnail()
    const img = wrapper.find('img')
    expect(img.attributes('draggable')).toBe('false')
  })

  it('applies transition classes', () => {
    const wrapper = mountThumbnail()
    const lazyImage = wrapper.findComponent({ name: 'LazyImage' })
    const imageClass = lazyImage.props('imageClass')
    const classString = Array.isArray(imageClass)
      ? imageClass.join(' ')
      : imageClass
    expect(classString).toContain('transform-gpu')
    expect(classString).toContain('transition-transform')
    expect(classString).toContain('duration-300')
    expect(classString).toContain('ease-out')
  })

  it('passes correct props to BaseThumbnail', () => {
    const wrapper = mountThumbnail({
      hoverZoom: 20,
      isHovered: true
    })
    const baseThumbnail = wrapper.findComponent({ name: 'BaseThumbnail' })
    expect(baseThumbnail.props('hoverZoom')).toBe(20)
    expect(baseThumbnail.props('isHovered')).toBe(true)
  })
})
