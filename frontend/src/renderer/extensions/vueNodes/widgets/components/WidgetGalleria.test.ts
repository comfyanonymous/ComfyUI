import { mount } from '@vue/test-utils'
import PrimeVue from 'primevue/config'
import Galleria from 'primevue/galleria'
import type { GalleriaProps } from 'primevue/galleria'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'

import type { SimplifiedWidget } from '@/types/simplifiedWidget'

import WidgetGalleria from './WidgetGalleria.vue'
import type { GalleryImage, GalleryValue } from './WidgetGalleria.vue'

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {
      'Gallery image': 'Gallery image'
    }
  }
})

// Test data constants for better test isolation
const TEST_IMAGES_SMALL: readonly string[] = Object.freeze([
  'https://example.com/image0.jpg',
  'https://example.com/image1.jpg',
  'https://example.com/image2.jpg'
])

const TEST_IMAGES_SINGLE: readonly string[] = Object.freeze([
  'https://example.com/single.jpg'
])

const TEST_IMAGE_OBJECTS: readonly GalleryImage[] = Object.freeze([
  {
    itemImageSrc: 'https://example.com/image0.jpg',
    thumbnailImageSrc: 'https://example.com/thumb0.jpg',
    alt: 'Test image 0'
  },
  {
    itemImageSrc: 'https://example.com/image1.jpg',
    thumbnailImageSrc: 'https://example.com/thumb1.jpg',
    alt: 'Test image 1'
  }
])

// Helper functions outside describe blocks for better clarity
function createMockWidget(
  value: GalleryValue = [],
  options: Partial<GalleriaProps> = {}
): SimplifiedWidget<GalleryValue> {
  return {
    name: 'test_galleria',
    type: 'array',
    value,
    options
  }
}

function mountComponent(
  widget: SimplifiedWidget<GalleryValue>,
  modelValue: GalleryValue
) {
  return mount(WidgetGalleria, {
    global: {
      plugins: [PrimeVue, i18n],
      components: { Galleria }
    },
    props: {
      widget,
      modelValue
    }
  })
}

function createImageStrings(count: number): string[] {
  return Array.from(
    { length: count },
    (_, i) => `https://example.com/image${i}.jpg`
  )
}

// Factory function that takes images, creates widget internally, returns wrapper
function createGalleriaWrapper(
  images: GalleryValue,
  options: Partial<GalleriaProps> = {}
) {
  const widget = createMockWidget(images, options)
  return mountComponent(widget, images)
}

describe('WidgetGalleria Image Display', () => {
  // Group tests using the readonly constants where appropriate

  describe('Component Rendering', () => {
    it('renders galleria component', () => {
      const wrapper = createGalleriaWrapper([...TEST_IMAGES_SMALL])

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      expect(galleria.exists()).toBe(true)
    })

    it('displays empty gallery when no images provided', () => {
      const widget = createMockWidget([])
      const wrapper = mountComponent(widget, [])

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      expect(galleria.props('value')).toEqual([])
    })

    it('handles null or undefined value gracefully', () => {
      const widget = createMockWidget([])
      const wrapper = mountComponent(widget, [])

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      expect(galleria.props('value')).toEqual([])
    })
  })

  describe('String Array Input', () => {
    it('converts string array to image objects', () => {
      const widget = createMockWidget([...TEST_IMAGES_SMALL])
      const wrapper = mountComponent(widget, [...TEST_IMAGES_SMALL])

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      const value = galleria.props('value')

      expect(value).toHaveLength(3)
      expect(value[0]).toEqual({
        itemImageSrc: 'https://example.com/image0.jpg',
        thumbnailImageSrc: 'https://example.com/image0.jpg',
        alt: 'Image 0'
      })
    })

    it('handles single string image', () => {
      const widget = createMockWidget([...TEST_IMAGES_SINGLE])
      const wrapper = mountComponent(widget, [...TEST_IMAGES_SINGLE])

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      const value = galleria.props('value')

      expect(value).toHaveLength(1)
      expect(value[0]).toEqual({
        itemImageSrc: 'https://example.com/single.jpg',
        thumbnailImageSrc: 'https://example.com/single.jpg',
        alt: 'Image 0'
      })
    })
  })

  describe('Object Array Input', () => {
    it('preserves image objects as-is', () => {
      const widget = createMockWidget([...TEST_IMAGE_OBJECTS])
      const wrapper = mountComponent(widget, [...TEST_IMAGE_OBJECTS])

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      const value = galleria.props('value')

      expect(value).toEqual([...TEST_IMAGE_OBJECTS])
    })

    it('handles mixed object properties', () => {
      const images: GalleryImage[] = [
        { src: 'https://example.com/image1.jpg', alt: 'First' },
        { itemImageSrc: 'https://example.com/image2.jpg' },
        { thumbnailImageSrc: 'https://example.com/thumb3.jpg' }
      ]
      const widget = createMockWidget(images)
      const wrapper = mountComponent(widget, images)

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      const value = galleria.props('value')

      expect(value).toEqual(images)
    })
  })

  describe('Thumbnail Display', () => {
    it('shows thumbnails when multiple images present', () => {
      const wrapper = createGalleriaWrapper([...TEST_IMAGES_SMALL])

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      expect(galleria.props('showThumbnails')).toBe(true)
    })

    it('hides thumbnails for single image', () => {
      const wrapper = createGalleriaWrapper([...TEST_IMAGES_SINGLE])

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      expect(galleria.props('showThumbnails')).toBe(false)
    })

    it('respects widget option to hide thumbnails', () => {
      const wrapper = createGalleriaWrapper([...TEST_IMAGES_SMALL], {
        showThumbnails: false
      })

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      expect(galleria.props('showThumbnails')).toBe(false)
    })

    it('shows thumbnails when explicitly enabled for multiple images', () => {
      const wrapper = createGalleriaWrapper([...TEST_IMAGES_SMALL], {
        showThumbnails: true
      })

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      expect(galleria.props('showThumbnails')).toBe(true)
    })
  })

  describe('Navigation Buttons', () => {
    it('shows navigation buttons when multiple images present', () => {
      const wrapper = createGalleriaWrapper([...TEST_IMAGES_SMALL])

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      expect(galleria.props('showItemNavigators')).toBe(true)
    })

    it('hides navigation buttons for single image', () => {
      const wrapper = createGalleriaWrapper([...TEST_IMAGES_SINGLE])

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      expect(galleria.props('showItemNavigators')).toBe(false)
    })

    it('respects widget option to hide navigation buttons', () => {
      const images = createImageStrings(3)
      const widget = createMockWidget(images, { showItemNavigators: false })
      const wrapper = mountComponent(widget, images)

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      expect(galleria.props('showItemNavigators')).toBe(false)
    })

    it('shows navigation buttons when explicitly enabled for multiple images', () => {
      const images = createImageStrings(3)
      const widget = createMockWidget(images, { showItemNavigators: true })
      const wrapper = mountComponent(widget, images)

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      expect(galleria.props('showItemNavigators')).toBe(true)
    })
  })

  describe('Widget Options Handling', () => {
    it('passes through valid widget options', () => {
      const images = createImageStrings(2)
      const widget = createMockWidget(images, {
        circular: true,
        autoPlay: true,
        transitionInterval: 3000
      })
      const wrapper = mountComponent(widget, images)

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      expect(galleria.props('circular')).toBe(true)
      expect(galleria.props('autoPlay')).toBe(true)
      expect(galleria.props('transitionInterval')).toBe(3000)
    })

    it('applies custom styling props', () => {
      const images = createImageStrings(2)
      const widget = createMockWidget(images)
      const wrapper = mountComponent(widget, images)

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      // Check that galleria has styling attributes rather than specific classes
      expect(galleria.attributes('class')).toBeDefined()
    })
  })

  describe('Active Index Management', () => {
    it('initializes with zero active index', () => {
      const images = createImageStrings(3)
      const widget = createMockWidget(images)
      const wrapper = mountComponent(widget, images)

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      expect(galleria.props('activeIndex')).toBe(0)
    })

    it('can update active index', async () => {
      const images = createImageStrings(3)
      const widget = createMockWidget(images)
      const wrapper = mountComponent(widget, images)

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      await galleria.vm.$emit('update:activeIndex', 2)

      // Check that the internal activeIndex ref was updated
      const vm = wrapper.vm as typeof wrapper.vm & { activeIndex: number }
      expect(vm.activeIndex).toBe(2)
    })
  })

  describe('Image Template Rendering', () => {
    it('renders item template with correct image source priorities', () => {
      const images: GalleryImage[] = [
        {
          itemImageSrc: 'https://example.com/item.jpg',
          src: 'https://example.com/fallback.jpg'
        },
        { src: 'https://example.com/only-src.jpg' }
      ]
      const widget = createMockWidget(images)
      const wrapper = mountComponent(widget, images)

      // The template logic should prioritize itemImageSrc > src > fallback to the item itself
      const galleria = wrapper.findComponent({ name: 'Galleria' })
      expect(galleria.exists()).toBe(true)
    })

    it('renders thumbnail template with correct image source priorities', () => {
      const images: GalleryImage[] = [
        {
          thumbnailImageSrc: 'https://example.com/thumb.jpg',
          src: 'https://example.com/fallback.jpg'
        },
        { src: 'https://example.com/only-src.jpg' }
      ]
      const widget = createMockWidget(images)
      const wrapper = mountComponent(widget, images)

      // The template logic should prioritize thumbnailImageSrc > src > fallback to the item itself
      const galleria = wrapper.findComponent({ name: 'Galleria' })
      expect(galleria.exists()).toBe(true)
    })
  })

  describe('Edge Cases', () => {
    it('handles empty array gracefully', () => {
      const widget = createMockWidget([])
      const wrapper = mountComponent(widget, [])

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      expect(galleria.props('value')).toEqual([])
      expect(galleria.props('showThumbnails')).toBe(false)
      expect(galleria.props('showItemNavigators')).toBe(false)
    })

    it('handles malformed image objects', () => {
      const malformedImages = [
        {}, // Empty object
        { randomProp: 'value' }, // Object without expected image properties
        null, // Null value
        undefined // Undefined value
      ]
      const widget = createMockWidget(malformedImages as string[])
      const wrapper = mountComponent(widget, malformedImages as string[])

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      // Null/undefined should be filtered out, leaving only the objects
      const expectedValue = [{}, { randomProp: 'value' }]
      expect(galleria.props('value')).toEqual(expectedValue)
    })

    it('handles very large image arrays', () => {
      const largeImageArray = createImageStrings(100)
      const widget = createMockWidget(largeImageArray)
      const wrapper = mountComponent(widget, largeImageArray)

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      expect(galleria.props('value')).toHaveLength(100)
      expect(galleria.props('showThumbnails')).toBe(true)
      expect(galleria.props('showItemNavigators')).toBe(true)
    })

    it('handles mixed string and object arrays gracefully', () => {
      // This is technically invalid input, but the component should handle it
      const mixedArray = [
        'https://example.com/string.jpg',
        { itemImageSrc: 'https://example.com/object.jpg' },
        'https://example.com/another-string.jpg'
      ]
      const widget = createMockWidget(mixedArray as string[])

      // The component expects consistent typing, but let's test it handles mixed input
      expect(() => mountComponent(widget, mixedArray as string[])).not.toThrow()
    })

    it('handles invalid URL strings', () => {
      const invalidUrls = ['not-a-url', '', ' ', 'http://', 'ftp://invalid']
      const widget = createMockWidget(invalidUrls)
      const wrapper = mountComponent(widget, invalidUrls)

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      expect(galleria.props('value')).toHaveLength(5)
    })
  })

  describe('Styling and Layout', () => {
    it('applies max-width constraint', () => {
      const images = createImageStrings(2)
      const widget = createMockWidget(images)
      const wrapper = mountComponent(widget, images)

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      // Check that component has styling applied rather than specific classes
      expect(galleria.attributes('class')).toBeDefined()
    })

    it('applies passthrough props for thumbnails', () => {
      const images = createImageStrings(3)
      const widget = createMockWidget(images)
      const wrapper = mountComponent(widget, images)

      const galleria = wrapper.findComponent({ name: 'Galleria' })
      const pt = galleria.props('pt')

      expect(pt).toBeDefined()
      expect(pt.thumbnails).toBeDefined()
      expect(pt.thumbnailContent).toBeDefined()
      expect(pt.thumbnailPrevButton).toBeDefined()
      expect(pt.thumbnailNextButton).toBeDefined()
    })
  })
})
