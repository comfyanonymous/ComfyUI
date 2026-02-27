import { mount } from '@vue/test-utils'
import { createPinia } from 'pinia'
import PrimeVue from 'primevue/config'
import Tag from 'primevue/tag'
import Tooltip from 'primevue/tooltip'
import { describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'

import SettingItem from '@/platform/settings/components/SettingItem.vue'

const i18n = createI18n({
  legacy: false,
  locale: 'en'
})

vi.mock('@/utils/formatUtil', () => ({
  normalizeI18nKey: vi.fn()
}))

describe('SettingItem', () => {
  const mountComponent = (props: Record<string, unknown>, options = {}) => {
    return mount(SettingItem, {
      global: {
        plugins: [PrimeVue, i18n, createPinia()],
        components: {
          Tag
        },
        directives: {
          tooltip: Tooltip
        },
        stubs: {
          'i-material-symbols:experiment-outline': true
        }
      },
      // @ts-expect-error - Test utility accepts flexible props for testing edge cases
      props,
      ...options
    })
  }

  it('translates options that use legacy type', () => {
    const wrapper = mountComponent({
      setting: {
        id: 'Comfy.NodeInputConversionSubmenus',
        name: 'Node Input Conversion Submenus',
        type: 'combo',
        value: 'Top',
        options: () => ['Correctly Translated']
      }
    })

    // Check the FormItem component's item prop for the options
    const formItem = wrapper.findComponent({ name: 'FormItem' })
    const options = formItem.props('item').options
    expect(options).toEqual([
      { text: 'Correctly Translated', value: 'Correctly Translated' }
    ])
  })

  it('handles tooltips with @ symbols without errors', () => {
    const wrapper = mountComponent({
      setting: {
        id: 'TestSetting',
        name: 'Test Setting',
        type: 'boolean',
        tooltip:
          'This will load a larger version of @mtb/markdown-parser that bundles shiki'
      }
    })

    // Should not throw an error and tooltip should be preserved as-is
    const formItem = wrapper.findComponent({ name: 'FormItem' })
    expect(formItem.props('item').tooltip).toBe(
      'This will load a larger version of @mtb/markdown-parser that bundles shiki'
    )
  })
})
