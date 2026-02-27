import { mount } from '@vue/test-utils'
import PrimeVue from 'primevue/config'
import InputText from 'primevue/inputtext'
import type { InputTextProps } from 'primevue/inputtext'
import Textarea from 'primevue/textarea'
import { describe, expect, it } from 'vitest'

import type { IWidgetOptions } from '@/lib/litegraph/src/types/widgets'
import type { SimplifiedWidget } from '@/types/simplifiedWidget'

import WidgetInputText from './WidgetInputText.vue'

describe('WidgetInputText Value Binding', () => {
  const createMockWidget = (
    value: string = 'default',
    options: Partial<InputTextProps> = {},
    callback?: (value: string) => void
  ): SimplifiedWidget<string> => ({
    name: 'test_input',
    type: 'string',
    value,
    options: options as IWidgetOptions,
    callback
  })

  const mountComponent = (
    widget: SimplifiedWidget<string>,
    modelValue: string,
    readonly = false
  ) => {
    return mount(WidgetInputText, {
      global: {
        plugins: [PrimeVue],
        components: { InputText, Textarea }
      },
      props: {
        widget,
        modelValue,
        readonly
      }
    })
  }

  const setInputValueAndTrigger = async (
    wrapper: ReturnType<typeof mount>,
    value: string,
    trigger: 'blur' | 'keydown.enter' = 'blur'
  ) => {
    const input = wrapper.find('input[type="text"]')
    if (!(input.element instanceof HTMLInputElement)) {
      throw new Error('Input element not found or is not an HTMLInputElement')
    }
    await input.setValue(value)
    await input.trigger(trigger)
    return input
  }

  describe('Vue Event Emission', () => {
    it('emits Vue event when input value changes on blur', async () => {
      const widget = createMockWidget('hello')
      const wrapper = mountComponent(widget, 'hello')

      await setInputValueAndTrigger(wrapper, 'world', 'blur')

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain('world')
    })

    it('emits Vue event when enter key is pressed', async () => {
      const widget = createMockWidget('initial')
      const wrapper = mountComponent(widget, 'initial')

      await setInputValueAndTrigger(wrapper, 'new value', 'keydown.enter')

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain('new value')
    })

    it('handles empty string values', async () => {
      const widget = createMockWidget('something')
      const wrapper = mountComponent(widget, 'something')

      await setInputValueAndTrigger(wrapper, '')

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain('')
    })

    it('handles special characters correctly', async () => {
      const widget = createMockWidget('normal')
      const wrapper = mountComponent(widget, 'normal')

      const specialText = 'special @#$%^&*()[]{}|\\:";\'<>?,./'
      await setInputValueAndTrigger(wrapper, specialText)

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain(specialText)
    })

    it('handles missing callback gracefully', async () => {
      const widget = createMockWidget('test', {}, undefined)
      const wrapper = mountComponent(widget, 'test')

      await setInputValueAndTrigger(wrapper, 'new value')

      // Should still emit Vue event
      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain('new value')
    })
  })

  describe('User Interactions', () => {
    it('emits update:modelValue on blur', async () => {
      const widget = createMockWidget('original')
      const wrapper = mountComponent(widget, 'original')

      await setInputValueAndTrigger(wrapper, 'updated')

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain('updated')
    })

    it('emits update:modelValue on enter key', async () => {
      const widget = createMockWidget('start')
      const wrapper = mountComponent(widget, 'start')

      await setInputValueAndTrigger(wrapper, 'finish', 'keydown.enter')

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain('finish')
    })
  })

  describe('Component Rendering', () => {
    it('always renders InputText component', () => {
      const widget = createMockWidget('test value')
      const wrapper = mountComponent(widget, 'test value')

      // WidgetInputText always uses InputText, not Textarea
      const input = wrapper.find('input[type="text"]')
      expect(input.exists()).toBe(true)

      // Should not render textarea (that's handled by WidgetTextarea component)
      const textarea = wrapper.find('textarea')
      expect(textarea.exists()).toBe(false)
    })
  })

  describe('Edge Cases', () => {
    it('handles very long strings', async () => {
      const widget = createMockWidget('short')
      const wrapper = mountComponent(widget, 'short')

      const longString = 'a'.repeat(10000)
      await setInputValueAndTrigger(wrapper, longString)

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain(longString)
    })

    it('handles unicode characters', async () => {
      const widget = createMockWidget('ascii')
      const wrapper = mountComponent(widget, 'ascii')

      const unicodeText = '🎨 Unicode: αβγ 中文 العربية 🚀'
      await setInputValueAndTrigger(wrapper, unicodeText)

      const emitted = wrapper.emitted('update:modelValue')
      expect(emitted).toBeDefined()
      expect(emitted![0]).toContain(unicodeText)
    })
  })
})
