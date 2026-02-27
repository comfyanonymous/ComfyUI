import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import type { SimplifiedWidget } from '@/types/simplifiedWidget'

import WidgetImageCompare from './WidgetImageCompare.vue'
import type { ImageCompareValue } from './WidgetImageCompare.vue'

describe('WidgetImageCompare Display', () => {
  const createMockWidget = (
    value: ImageCompareValue | string,
    options: SimplifiedWidget['options'] = {}
  ): SimplifiedWidget<ImageCompareValue | string> => ({
    name: 'test_imagecompare',
    type: 'object',
    value,
    options
  })

  const mountComponent = (
    widget: SimplifiedWidget<ImageCompareValue | string>,
    readonly = false
  ) => {
    return mount(WidgetImageCompare, {
      global: {
        mocks: {
          $t: (key: string, params?: Record<string, unknown>) => {
            if (key === 'batch.index' && params) {
              return `${params.current} / ${params.total}`
            }
            return key
          }
        }
      },
      props: {
        widget,
        readonly
      }
    })
  }

  describe('Component Rendering', () => {
    it('renders with proper structure and styling when images are provided', () => {
      const value: ImageCompareValue = {
        beforeImages: ['https://example.com/before.jpg'],
        afterImages: ['https://example.com/after.jpg']
      }
      const widget = createMockWidget(value)
      const wrapper = mountComponent(widget)

      const images = wrapper.findAll('img')
      expect(images).toHaveLength(2)

      // After image is first (background), before image is second (overlay)
      expect(images[0].attributes('src')).toBe('https://example.com/after.jpg')
      expect(images[1].attributes('src')).toBe('https://example.com/before.jpg')

      images.forEach((img) => {
        expect(img.classes()).toContain('object-contain')
      })
    })
  })

  describe('Object Value Input', () => {
    it('handles alt text correctly - custom, default, and empty', () => {
      // Test custom alt text
      const customAltValue: ImageCompareValue = {
        beforeImages: ['https://example.com/before.jpg'],
        afterImages: ['https://example.com/after.jpg'],
        beforeAlt: 'Original design',
        afterAlt: 'Updated design'
      }
      const customWrapper = mountComponent(createMockWidget(customAltValue))
      const customImages = customWrapper.findAll('img')
      // DOM order: [after, before]
      expect(customImages[0].attributes('alt')).toBe('Updated design')
      expect(customImages[1].attributes('alt')).toBe('Original design')

      // Test default alt text
      const defaultAltValue: ImageCompareValue = {
        beforeImages: ['https://example.com/before.jpg'],
        afterImages: ['https://example.com/after.jpg']
      }
      const defaultWrapper = mountComponent(createMockWidget(defaultAltValue))
      const defaultImages = defaultWrapper.findAll('img')
      expect(defaultImages[0].attributes('alt')).toBe('After image')
      expect(defaultImages[1].attributes('alt')).toBe('Before image')

      // Test empty string alt text (falls back to default)
      const emptyAltValue: ImageCompareValue = {
        beforeImages: ['https://example.com/before.jpg'],
        afterImages: ['https://example.com/after.jpg'],
        beforeAlt: '',
        afterAlt: ''
      }
      const emptyWrapper = mountComponent(createMockWidget(emptyAltValue))
      const emptyImages = emptyWrapper.findAll('img')
      expect(emptyImages[0].attributes('alt')).toBe('After image')
      expect(emptyImages[1].attributes('alt')).toBe('Before image')
    })

    it('handles partial image URLs gracefully', () => {
      // Only before image provided
      const beforeOnlyWrapper = mountComponent(
        createMockWidget({
          beforeImages: ['https://example.com/before.jpg']
        })
      )
      const beforeOnlyImages = beforeOnlyWrapper.findAll('img')
      expect(beforeOnlyImages).toHaveLength(1)
      expect(beforeOnlyImages[0].attributes('src')).toBe(
        'https://example.com/before.jpg'
      )

      // Only after image provided
      const afterOnlyWrapper = mountComponent(
        createMockWidget({
          afterImages: ['https://example.com/after.jpg']
        })
      )
      const afterOnlyImages = afterOnlyWrapper.findAll('img')
      expect(afterOnlyImages).toHaveLength(1)
      expect(afterOnlyImages[0].attributes('src')).toBe(
        'https://example.com/after.jpg'
      )
    })
  })

  describe('String Value Input', () => {
    it('handles string value as before image only', () => {
      const value = 'https://example.com/single.jpg'
      const widget = createMockWidget(value)
      const wrapper = mountComponent(widget)

      const images = wrapper.findAll('img')
      expect(images).toHaveLength(1)
      expect(images[0].attributes('src')).toBe('https://example.com/single.jpg')
      expect(images[0].attributes('alt')).toBe('Before image')
    })
  })

  describe('Readonly Mode', () => {
    it('renders normally in readonly mode', () => {
      const value: ImageCompareValue = {
        beforeImages: ['https://example.com/before.jpg'],
        afterImages: ['https://example.com/after.jpg']
      }
      const widget = createMockWidget(value)
      const wrapper = mountComponent(widget, true)

      const images = wrapper.findAll('img')
      expect(images).toHaveLength(2)
    })
  })

  describe('Edge Cases', () => {
    it('shows no images message when widget value is empty string', () => {
      const widget = createMockWidget('')
      const wrapper = mountComponent(widget)

      const images = wrapper.findAll('img')
      expect(images).toHaveLength(0)
      expect(wrapper.text()).toContain('imageCompare.noImages')
    })

    it('shows no images message when both arrays are empty', () => {
      const value: ImageCompareValue = {
        beforeImages: [],
        afterImages: []
      }
      const widget = createMockWidget(value)
      const wrapper = mountComponent(widget)

      const images = wrapper.findAll('img')
      expect(images).toHaveLength(0)
      expect(wrapper.text()).toContain('imageCompare.noImages')
    })

    it('shows no images message for empty object value', () => {
      const value: ImageCompareValue = {} as ImageCompareValue
      const widget = createMockWidget(value)
      const wrapper = mountComponent(widget)

      const images = wrapper.findAll('img')
      expect(images).toHaveLength(0)
      expect(wrapper.text()).toContain('imageCompare.noImages')
    })

    it('handles special content - long URLs, special characters, and long alt text', () => {
      const longUrl = 'https://example.com/' + 'a'.repeat(1000) + '.jpg'
      const longUrlWrapper = mountComponent(
        createMockWidget({
          beforeImages: [longUrl],
          afterImages: [longUrl]
        })
      )
      const longUrlImages = longUrlWrapper.findAll('img')
      expect(longUrlImages[0].attributes('src')).toBe(longUrl)
      expect(longUrlImages[1].attributes('src')).toBe(longUrl)

      const specialUrl =
        'https://example.com/path with spaces & symbols!@#$.jpg'
      const specialUrlWrapper = mountComponent(
        createMockWidget({
          beforeImages: [specialUrl],
          afterImages: [specialUrl]
        })
      )
      const specialUrlImages = specialUrlWrapper.findAll('img')
      expect(specialUrlImages[0].attributes('src')).toBe(specialUrl)
      expect(specialUrlImages[1].attributes('src')).toBe(specialUrl)

      const longAlt =
        'Very long alt text that exceeds normal length: ' +
        'description '.repeat(50)
      const longAltWrapper = mountComponent(
        createMockWidget({
          beforeImages: ['https://example.com/before.jpg'],
          afterImages: ['https://example.com/after.jpg'],
          beforeAlt: longAlt,
          afterAlt: longAlt
        })
      )
      const longAltImages = longAltWrapper.findAll('img')
      expect(longAltImages[0].attributes('alt')).toBe(longAlt)
      expect(longAltImages[1].attributes('alt')).toBe(longAlt)
    })
  })

  describe('Template Structure', () => {
    it('correctly renders after image as background and before image as overlay', () => {
      const value: ImageCompareValue = {
        beforeImages: ['https://example.com/before.jpg'],
        afterImages: ['https://example.com/after.jpg']
      }
      const widget = createMockWidget(value)
      const wrapper = mountComponent(widget)

      const images = wrapper.findAll('img')
      expect(images[0].attributes('src')).toBe('https://example.com/after.jpg')
      expect(images[1].attributes('src')).toBe('https://example.com/before.jpg')
      expect(images[1].classes()).toContain('absolute')
    })
  })

  describe('Integration', () => {
    it('works with various URL types - data URLs and blob URLs', () => {
      const dataUrl =
        'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=='
      const dataUrlWrapper = mountComponent(
        createMockWidget({
          beforeImages: [dataUrl],
          afterImages: [dataUrl]
        })
      )
      const dataUrlImages = dataUrlWrapper.findAll('img')
      expect(dataUrlImages[0].attributes('src')).toBe(dataUrl)
      expect(dataUrlImages[1].attributes('src')).toBe(dataUrl)

      const blobUrl =
        'blob:http://example.com/12345678-1234-1234-1234-123456789012'
      const blobUrlWrapper = mountComponent(
        createMockWidget({
          beforeImages: [blobUrl],
          afterImages: [blobUrl]
        })
      )
      const blobUrlImages = blobUrlWrapper.findAll('img')
      expect(blobUrlImages[0].attributes('src')).toBe(blobUrl)
      expect(blobUrlImages[1].attributes('src')).toBe(blobUrl)
    })
  })

  describe('Slider Element', () => {
    it('renders slider divider when images are present', () => {
      const value: ImageCompareValue = {
        beforeImages: ['https://example.com/before.jpg'],
        afterImages: ['https://example.com/after.jpg']
      }
      const widget = createMockWidget(value)
      const wrapper = mountComponent(widget)

      const slider = wrapper.find('[role="presentation"]')
      expect(slider.exists()).toBe(true)
      expect(slider.classes()).toContain('bg-white')
    })

    it('does not render slider when no images', () => {
      const widget = createMockWidget('')
      const wrapper = mountComponent(widget)

      const slider = wrapper.find('[role="presentation"]')
      expect(slider.exists()).toBe(false)
    })
  })

  describe('Batch Navigation', () => {
    const beforeImages = [
      'https://example.com/a1.jpg',
      'https://example.com/a2.jpg',
      'https://example.com/a3.jpg'
    ]
    const afterImages = [
      'https://example.com/b1.jpg',
      'https://example.com/b2.jpg'
    ]

    it('shows batch nav when either side has multiple images', () => {
      const value: ImageCompareValue = { beforeImages, afterImages }
      const wrapper = mountComponent(createMockWidget(value))

      expect(wrapper.find('[data-testid="batch-nav"]').exists()).toBe(true)

      const beforeBatch = wrapper.find('[data-testid="before-batch"]')
      const afterBatch = wrapper.find('[data-testid="after-batch"]')
      expect(beforeBatch.find('[data-testid="batch-counter"]').text()).toBe(
        '1 / 3'
      )
      expect(afterBatch.find('[data-testid="batch-counter"]').text()).toBe(
        '1 / 2'
      )
    })

    it('hides batch nav for single images', () => {
      const value: ImageCompareValue = {
        beforeImages: ['https://example.com/a1.jpg'],
        afterImages: ['https://example.com/b1.jpg']
      }
      const wrapper = mountComponent(createMockWidget(value))

      expect(wrapper.find('[data-testid="batch-nav"]').exists()).toBe(false)
    })

    it('hides batch nav when no batch arrays are provided', () => {
      const wrapper = mountComponent(createMockWidget({} as ImageCompareValue))

      expect(wrapper.find('[data-testid="batch-nav"]').exists()).toBe(false)
    })

    it('navigates before images with prev/next buttons', async () => {
      const value: ImageCompareValue = { beforeImages, afterImages }
      const wrapper = mountComponent(createMockWidget(value))
      const beforeBatch = wrapper.find('[data-testid="before-batch"]')

      // Initially shows first before image
      expect(wrapper.findAll('img')[1].attributes('src')).toBe(
        'https://example.com/a1.jpg'
      )

      // Click next on before
      await beforeBatch.find('[data-testid="batch-next"]').trigger('click')
      expect(wrapper.findAll('img')[1].attributes('src')).toBe(
        'https://example.com/a2.jpg'
      )
      expect(beforeBatch.find('[data-testid="batch-counter"]').text()).toBe(
        '2 / 3'
      )

      // Click next again
      await beforeBatch.find('[data-testid="batch-next"]').trigger('click')
      expect(wrapper.findAll('img')[1].attributes('src')).toBe(
        'https://example.com/a3.jpg'
      )
      expect(beforeBatch.find('[data-testid="batch-counter"]').text()).toBe(
        '3 / 3'
      )

      // Next button should be disabled at last index
      expect(
        beforeBatch.find('[data-testid="batch-next"]').attributes('disabled')
      ).toBeDefined()

      // Click prev
      await beforeBatch.find('[data-testid="batch-prev"]').trigger('click')
      expect(wrapper.findAll('img')[1].attributes('src')).toBe(
        'https://example.com/a2.jpg'
      )
    })

    it('navigates after images independently from before images', async () => {
      const value: ImageCompareValue = { beforeImages, afterImages }
      const wrapper = mountComponent(createMockWidget(value))
      const afterBatch = wrapper.find('[data-testid="after-batch"]')

      // Navigate after to index 1
      await afterBatch.find('[data-testid="batch-next"]').trigger('click')
      expect(afterBatch.find('[data-testid="batch-counter"]').text()).toBe(
        '2 / 2'
      )

      // After image should be b2, before image should still be a1
      const images = wrapper.findAll('img')
      expect(images[0].attributes('src')).toBe('https://example.com/b2.jpg')
      expect(images[1].attributes('src')).toBe('https://example.com/a1.jpg')
    })

    it('disables prev button at first index', () => {
      const value: ImageCompareValue = { beforeImages, afterImages }
      const wrapper = mountComponent(createMockWidget(value))

      expect(
        wrapper
          .find('[data-testid="before-batch"]')
          .find('[data-testid="batch-prev"]')
          .attributes('disabled')
      ).toBeDefined()
      expect(
        wrapper
          .find('[data-testid="after-batch"]')
          .find('[data-testid="batch-prev"]')
          .attributes('disabled')
      ).toBeDefined()
    })

    it('only shows controls for the side with multiple images', () => {
      const value: ImageCompareValue = {
        beforeImages,
        afterImages: ['https://example.com/b1.jpg']
      }
      const wrapper = mountComponent(createMockWidget(value))

      expect(wrapper.find('[data-testid="batch-nav"]').exists()).toBe(true)
      expect(
        wrapper
          .find('[data-testid="before-batch"]')
          .find('[data-testid="batch-counter"]')
          .exists()
      ).toBe(true)
      expect(wrapper.find('[data-testid="after-batch"]').exists()).toBe(false)
    })
  })
})
