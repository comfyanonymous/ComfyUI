import { createTestingPinia } from '@pinia/testing'
import type { VueWrapper } from '@vue/test-utils'
import { mount } from '@vue/test-utils'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import { createI18n } from 'vue-i18n'

import { downloadFile } from '@/base/common/downloadUtil'
import ImagePreview from '@/renderer/extensions/vueNodes/components/ImagePreview.vue'

// Mock downloadFile to avoid DOM errors
vi.mock('@/base/common/downloadUtil', () => ({
  downloadFile: vi.fn()
}))

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {
      g: {
        editOrMaskImage: 'Edit or mask image',
        downloadImage: 'Download image',
        removeImage: 'Remove image',
        viewImageOfTotal: 'View image {index} of {total}',
        imagePreview:
          'Image preview - Use arrow keys to navigate between images',
        errorLoadingImage: 'Error loading image',
        failedToDownloadImage: 'Failed to download image',
        calculatingDimensions: 'Calculating dimensions',
        imageFailedToLoad: 'Image failed to load',
        imageDoesNotExist: 'Image does not exist',
        unknownFile: 'Unknown file',
        loading: 'Loading'
      }
    }
  }
})

describe('ImagePreview', () => {
  const defaultProps = {
    imageUrls: [
      '/api/view?filename=test1.png&type=output',
      '/api/view?filename=test2.png&type=output'
    ]
  }
  const wrapperRegistry = new Set<VueWrapper>()

  const mountImagePreview = (props = {}) => {
    const wrapper = mount(ImagePreview, {
      props: { ...defaultProps, ...props },
      global: {
        plugins: [
          createTestingPinia({
            createSpy: vi.fn
          }),
          i18n
        ],
        stubs: {
          'i-lucide:venetian-mask': true,
          'i-lucide:download': true,
          'i-lucide:x': true,
          'i-lucide:image-off': true,
          Skeleton: true
        }
      }
    })
    wrapperRegistry.add(wrapper)
    return wrapper
  }

  afterEach(() => {
    wrapperRegistry.forEach((wrapper) => {
      wrapper.unmount()
    })
    wrapperRegistry.clear()
  })

  it('renders image preview when imageUrls provided', () => {
    const wrapper = mountImagePreview()

    expect(wrapper.find('.image-preview').exists()).toBe(true)
    expect(wrapper.find('img').exists()).toBe(true)
    expect(wrapper.find('img').attributes('src')).toBe(
      defaultProps.imageUrls[0]
    )
  })

  it('does not render when no imageUrls provided', () => {
    const wrapper = mountImagePreview({ imageUrls: [] })

    expect(wrapper.find('.image-preview').exists()).toBe(false)
  })

  it('displays calculating dimensions text initially', () => {
    const wrapper = mountImagePreview()

    expect(wrapper.text()).toContain('Calculating dimensions')
  })

  it('shows navigation dots for multiple images', () => {
    const wrapper = mountImagePreview()

    const navigationDots = wrapper.findAll('.w-2.h-2.rounded-full')
    expect(navigationDots).toHaveLength(2)
  })

  it('does not show navigation dots for single image', () => {
    const wrapper = mountImagePreview({
      imageUrls: [defaultProps.imageUrls[0]]
    })

    const navigationDots = wrapper.findAll('.w-2.h-2.rounded-full')
    expect(navigationDots).toHaveLength(0)
  })

  it('shows action buttons on hover', async () => {
    const wrapper = mountImagePreview()

    // Initially buttons should not be visible
    expect(wrapper.find('.actions').exists()).toBe(false)

    // Trigger hover on the image wrapper (the element with role="img" has the hover handlers)
    const imageWrapper = wrapper.find('[role="img"]')
    await imageWrapper.trigger('mouseenter')
    await nextTick()

    // Action buttons should now be visible
    expect(wrapper.find('.actions').exists()).toBe(true)
    // For multiple images: download and remove buttons (no mask button)
    expect(wrapper.find('[aria-label="Download image"]').exists()).toBe(true)
    expect(wrapper.find('[aria-label="Remove image"]').exists()).toBe(true)
    expect(wrapper.find('[aria-label="Edit or mask image"]').exists()).toBe(
      false
    )
  })

  it('hides action buttons when not hovering', async () => {
    const wrapper = mountImagePreview()
    const imageWrapper = wrapper.find('[role="img"]')

    // Trigger hover
    await imageWrapper.trigger('mouseenter')
    await nextTick()
    expect(wrapper.find('.actions').exists()).toBe(true)

    // Trigger mouse leave
    await imageWrapper.trigger('mouseleave')
    await nextTick()
    expect(wrapper.find('.actions').exists()).toBe(false)
  })

  it('shows action buttons on focus', async () => {
    const wrapper = mountImagePreview()

    // Initially buttons should not be visible
    expect(wrapper.find('.actions').exists()).toBe(false)

    // Trigger focusin on the image wrapper (useFocusWithin listens to focusin/focusout)
    const imageWrapper = wrapper.find('[role="img"]')
    await imageWrapper.trigger('focusin')
    await nextTick()

    // Action buttons should now be visible
    expect(wrapper.find('.actions').exists()).toBe(true)
  })

  it('hides action buttons on blur', async () => {
    const wrapper = mountImagePreview()
    const imageWrapper = wrapper.find('[role="img"]')

    // Trigger focus
    await imageWrapper.trigger('focusin')
    await nextTick()
    expect(wrapper.find('.actions').exists()).toBe(true)

    // Trigger focusout
    await imageWrapper.trigger('focusout')
    await nextTick()
    expect(wrapper.find('.actions').exists()).toBe(false)
  })

  it('shows mask/edit button only for single images', async () => {
    // Multiple images - should not show mask button
    const multipleImagesWrapper = mountImagePreview()
    await multipleImagesWrapper.find('[role="img"]').trigger('mouseenter')
    await nextTick()

    const maskButtonMultiple = multipleImagesWrapper.find(
      '[aria-label="Edit or mask image"]'
    )
    expect(maskButtonMultiple.exists()).toBe(false)

    // Single image - should show mask button
    const singleImageWrapper = mountImagePreview({
      imageUrls: [defaultProps.imageUrls[0]]
    })
    await singleImageWrapper.find('[role="img"]').trigger('mouseenter')
    await nextTick()

    const maskButtonSingle = singleImageWrapper.find(
      '[aria-label="Edit or mask image"]'
    )
    expect(maskButtonSingle.exists()).toBe(true)
  })

  it('handles action button clicks', async () => {
    const wrapper = mountImagePreview({
      imageUrls: [defaultProps.imageUrls[0]]
    })

    await wrapper.find('[role="img"]').trigger('mouseenter')
    await nextTick()

    // Test Edit/Mask button - just verify it can be clicked without errors
    const editButton = wrapper.find('[aria-label="Edit or mask image"]')
    expect(editButton.exists()).toBe(true)
    await editButton.trigger('click')

    // Test Remove button - just verify it can be clicked without errors
    const removeButton = wrapper.find('[aria-label="Remove image"]')
    expect(removeButton.exists()).toBe(true)
    await removeButton.trigger('click')
  })

  it('handles download button click', async () => {
    const wrapper = mountImagePreview({
      imageUrls: [defaultProps.imageUrls[0]]
    })

    await wrapper.find('[role="img"]').trigger('mouseenter')
    await nextTick()

    // Test Download button
    const downloadButton = wrapper.find('[aria-label="Download image"]')
    expect(downloadButton.exists()).toBe(true)
    await downloadButton.trigger('click')

    // Verify the mocked downloadFile was called
    expect(downloadFile).toHaveBeenCalledWith(defaultProps.imageUrls[0])
  })

  it('switches images when navigation dots are clicked', async () => {
    const wrapper = mountImagePreview()

    // Initially shows first image
    expect(wrapper.find('img').attributes('src')).toBe(
      defaultProps.imageUrls[0]
    )

    // Click second navigation dot
    const navigationDots = wrapper.findAll('.w-2.h-2.rounded-full')
    await navigationDots[1].trigger('click')
    await nextTick()

    // Now should show second image
    const imgElement = wrapper.find('img')
    expect(imgElement.exists()).toBe(true)
    expect(imgElement.attributes('src')).toBe(defaultProps.imageUrls[1])
  })

  it('applies correct classes to navigation dots based on current image', async () => {
    const wrapper = mountImagePreview()

    const navigationDots = wrapper.findAll('.w-2.h-2.rounded-full')

    // First dot should be active (has bg-white class)
    expect(navigationDots[0].classes()).toContain('bg-base-foreground')
    expect(navigationDots[1].classes()).toContain('bg-base-foreground/50')

    // Switch to second image
    await navigationDots[1].trigger('click')
    await nextTick()

    // Second dot should now be active
    expect(navigationDots[0].classes()).toContain('bg-base-foreground/50')
    expect(navigationDots[1].classes()).toContain('bg-base-foreground')
  })

  it('loads image without errors', async () => {
    const wrapper = mountImagePreview()

    const img = wrapper.find('img')
    expect(img.exists()).toBe(true)

    // Just verify the image element is properly set up
    expect(img.attributes('src')).toBe(defaultProps.imageUrls[0])
  })

  it('has proper accessibility attributes', () => {
    const wrapper = mountImagePreview()

    const img = wrapper.find('img')
    expect(img.attributes('alt')).toBe('Node output 1')
  })

  it('updates alt text when switching images', async () => {
    const wrapper = mountImagePreview()

    // Initially first image
    expect(wrapper.find('img').attributes('alt')).toBe('Node output 1')

    // Switch to second image
    const navigationDots = wrapper.findAll('.w-2.h-2.rounded-full')
    await navigationDots[1].trigger('click')
    await nextTick()

    // Alt text should update
    const imgElement = wrapper.find('img')
    expect(imgElement.exists()).toBe(true)
    expect(imgElement.attributes('alt')).toBe('Node output 2')
  })

  describe('URL change detection', () => {
    it('should NOT reset loading state when imageUrls prop is reassigned with identical URLs', async () => {
      vi.useFakeTimers()
      try {
        const urls = ['/api/view?filename=test.png&type=output']
        const wrapper = mountImagePreview({ imageUrls: urls })

        // Simulate image load completing
        const img = wrapper.find('img')
        await img.trigger('load')
        await nextTick()

        // Verify loader is hidden after load
        expect(wrapper.find('[aria-busy="true"]').exists()).toBe(false)

        // Reassign with new array reference but same content
        await wrapper.setProps({ imageUrls: [...urls] })
        await nextTick()

        // Advance past the 250ms delayed loader timeout
        await vi.advanceTimersByTimeAsync(300)
        await nextTick()

        // Loading state should NOT have been reset - aria-busy should still be false
        // because the URLs are identical (just a new array reference)
        expect(wrapper.find('[aria-busy="true"]').exists()).toBe(false)
      } finally {
        vi.useRealTimers()
      }
    })

    it('should reset loading state when imageUrls prop changes to different URLs', async () => {
      const urls = ['/api/view?filename=test.png&type=output']
      const wrapper = mountImagePreview({ imageUrls: urls })

      // Simulate image load completing
      const img = wrapper.find('img')
      await img.trigger('load')
      await nextTick()

      // Verify loader is hidden
      expect(wrapper.find('[aria-busy="true"]').exists()).toBe(false)

      // Change to different URL
      await wrapper.setProps({
        imageUrls: ['/api/view?filename=different.png&type=output']
      })
      await nextTick()

      // After 250ms timeout, loading state should be reset (aria-busy="true")
      // We can check the internal state via the Skeleton appearing
      // or wait for the timeout
      await new Promise((resolve) => setTimeout(resolve, 300))
      await nextTick()

      expect(wrapper.find('[aria-busy="true"]').exists()).toBe(true)
    })

    it('should handle empty to non-empty URL transitions correctly', async () => {
      const wrapper = mountImagePreview({ imageUrls: [] })

      // No preview initially
      expect(wrapper.find('.image-preview').exists()).toBe(false)

      // Add URLs
      await wrapper.setProps({
        imageUrls: ['/api/view?filename=test.png&type=output']
      })
      await nextTick()

      // Preview should appear
      expect(wrapper.find('.image-preview').exists()).toBe(true)
      expect(wrapper.find('img').exists()).toBe(true)
    })
  })
})
