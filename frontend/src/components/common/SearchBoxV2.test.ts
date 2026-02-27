import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'

import SearchBoxV2 from './SearchBoxV2.vue'

vi.mock('@vueuse/core', () => ({
  watchDebounced: vi.fn(() => vi.fn())
}))

describe('SearchBoxV2', () => {
  const i18n = createI18n({
    legacy: false,
    locale: 'en',
    messages: {
      en: {
        g: {
          clear: 'Clear',
          searchPlaceholder: 'Search...'
        }
      }
    }
  })

  function mountComponent(props = {}) {
    return mount(SearchBoxV2, {
      global: {
        plugins: [i18n],
        stubs: {
          ComboboxRoot: {
            template: '<div><slot /></div>'
          },
          ComboboxAnchor: {
            template: '<div><slot /></div>'
          },
          ComboboxInput: {
            template:
              '<input :placeholder="placeholder" :value="modelValue" @input="$emit(\'update:modelValue\', $event.target.value)" />',
            props: ['placeholder', 'modelValue', 'autoFocus']
          }
        }
      },
      props: {
        modelValue: '',
        ...props
      }
    })
  }

  it('uses i18n placeholder when no placeholder prop provided', () => {
    const wrapper = mountComponent()
    const input = wrapper.find('input')
    expect(input.attributes('placeholder')).toBe('Search...')
  })

  it('uses custom placeholder when provided', () => {
    const wrapper = mountComponent({
      placeholder: 'Custom placeholder'
    })
    const input = wrapper.find('input')
    expect(input.attributes('placeholder')).toBe('Custom placeholder')
  })

  it('shows search icon when search term is empty', () => {
    const wrapper = mountComponent({ modelValue: '' })
    expect(wrapper.find('i.icon-\\[lucide--search\\]').exists()).toBe(true)
  })

  it('shows clear button when search term is not empty', () => {
    const wrapper = mountComponent({ modelValue: 'test' })
    expect(wrapper.find('button').exists()).toBe(true)
  })

  it('clears search term when clear button is clicked', async () => {
    const wrapper = mountComponent({ modelValue: 'test' })
    const clearButton = wrapper.find('button')
    await clearButton.trigger('click')
    expect(wrapper.emitted('update:modelValue')?.[0]).toEqual([''])
  })

  it('applies large size classes when size is lg', () => {
    const wrapper = mountComponent({ size: 'lg' })
    expect(wrapper.html()).toContain('size-5')
  })

  it('applies medium size classes when size is md', () => {
    const wrapper = mountComponent({ size: 'md' })
    expect(wrapper.html()).toContain('size-4')
  })
})
