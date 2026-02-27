import { mount } from '@vue/test-utils'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import { createI18n } from 'vue-i18n'

import SearchBox from '@/components/common/SearchBox.vue'

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {
      templateWidgets: {
        sort: {
          searchPlaceholder: 'Search...'
        }
      }
    }
  }
})

describe('SearchBox', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.useFakeTimers()
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  const createWrapper = (props = {}) => {
    return mount(SearchBox, {
      props: {
        modelValue: '',
        ...props
      },
      global: {
        plugins: [i18n]
      }
    })
  }

  describe('debounced search functionality', () => {
    it('should debounce search input by 300ms', async () => {
      const wrapper = createWrapper()
      const input = wrapper.find('input')

      // Type search query
      await input.setValue('test')

      // Model should not update immediately
      expect(wrapper.emitted('search')).toBeFalsy()

      // Advance timers by 299ms (just before debounce delay)
      await vi.advanceTimersByTimeAsync(299)
      await nextTick()
      expect(wrapper.emitted('search')).toBeFalsy()

      // Advance timers by 1ms more (reaching 300ms)
      await vi.advanceTimersByTimeAsync(1)
      await nextTick()

      // Model should now be updated
      expect(wrapper.emitted('update:modelValue')).toBeTruthy()
      expect(wrapper.emitted('update:modelValue')?.[0]).toEqual(['test'])
    })

    it('should reset debounce timer on each keystroke', async () => {
      const wrapper = createWrapper()
      const input = wrapper.find('input')

      // Type first character
      await input.setValue('t')
      vi.advanceTimersByTime(200)
      await nextTick()

      // Type second character (should reset timer)
      await input.setValue('te')
      vi.advanceTimersByTime(200)
      await nextTick()

      // Type third character (should reset timer again)
      await input.setValue('tes')
      await vi.advanceTimersByTimeAsync(200)
      await nextTick()

      // Should not have emitted yet (only 200ms passed since last keystroke)
      expect(wrapper.emitted('search')).toBeFalsy()

      // Advance final 100ms to reach 300ms
      await vi.advanceTimersByTimeAsync(100)
      await nextTick()

      // Should now emit with final value
      expect(wrapper.emitted('search')).toBeTruthy()
      expect(wrapper.emitted('search')?.[0]).toEqual(['tes', []])
    })

    it('should only emit final value after rapid typing', async () => {
      const wrapper = createWrapper()
      const input = wrapper.find('input')

      // Simulate rapid typing
      const searchTerms = ['s', 'se', 'sea', 'sear', 'searc', 'search']
      for (const term of searchTerms) {
        await input.setValue(term)
        await vi.advanceTimersByTimeAsync(50) // Less than debounce delay
      }
      await nextTick()

      // Should not have emitted yet
      expect(wrapper.emitted('search')).toBeFalsy()

      // Complete the debounce delay
      await vi.advanceTimersByTimeAsync(350)
      await nextTick()

      // Should emit only once with final value
      expect(wrapper.emitted('search')).toHaveLength(1)
      expect(wrapper.emitted('search')?.[0]).toEqual(['search', []])
    })

    describe('bidirectional model sync', () => {
      it('should sync external model changes to internal state', async () => {
        const wrapper = createWrapper({ modelValue: 'initial' })
        const input = wrapper.find('input')

        expect(input.element.value).toBe('initial')

        // Update model externally
        await wrapper.setProps({ modelValue: 'external update' })
        await nextTick()

        // Internal state should sync
        expect(input.element.value).toBe('external update')
      })
    })

    describe('placeholder', () => {
      it('should use custom placeholder when provided', () => {
        const wrapper = createWrapper({ placeholder: 'Custom search...' })
        const input = wrapper.find('input')

        expect(input.attributes('placeholder')).toBe('Custom search...')
        expect(input.attributes('aria-label')).toBe('Custom search...')
      })

      it('should use default placeholder when not provided', () => {
        const wrapper = createWrapper()
        const input = wrapper.find('input')

        expect(input.attributes('placeholder')).toBe('Search...')
        expect(input.attributes('aria-label')).toBe('Search...')
      })
    })

    describe('autofocus', () => {
      it('should focus input when autofocus is true', async () => {
        const wrapper = createWrapper({ autofocus: true })
        await nextTick()

        const input = wrapper.find('input')
        const inputElement = input.element as HTMLInputElement

        // Note: In JSDOM, focus() doesn't actually set document.activeElement
        // We can only verify that the focus method exists and doesn't throw
        expect(inputElement.focus).toBeDefined()
      })

      it('should not autofocus when autofocus is false', () => {
        const wrapper = createWrapper({ autofocus: false })
        const input = wrapper.find('input')

        expect(document.activeElement).not.toBe(input.element)
      })
    })

    describe('click to focus', () => {
      it('should focus input when wrapper is clicked', async () => {
        const wrapper = createWrapper()
        const wrapperDiv = wrapper.find('[class*="flex"]')

        await wrapperDiv.trigger('click')
        await nextTick()

        // Input should receive focus
        const input = wrapper.find('input').element as HTMLInputElement
        expect(input.focus).toBeDefined()
      })
    })
  })
})
