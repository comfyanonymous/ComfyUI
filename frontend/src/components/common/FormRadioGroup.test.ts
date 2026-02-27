import { mount } from '@vue/test-utils'
import PrimeVue from 'primevue/config'
import RadioButton from 'primevue/radiobutton'
import { beforeAll, describe, expect, it } from 'vitest'
import { createApp } from 'vue'

import type { SettingOption } from '@/platform/settings/types'

import FormRadioGroup from './FormRadioGroup.vue'
import type { ComponentProps } from 'vue-component-type-helpers'

describe('FormRadioGroup', () => {
  beforeAll(() => {
    const app = createApp({})
    app.use(PrimeVue)
  })

  type FormRadioGroupProps = ComponentProps<typeof FormRadioGroup>
  const mountComponent = (props: FormRadioGroupProps, options = {}) => {
    return mount(FormRadioGroup, {
      global: {
        plugins: [PrimeVue],
        components: { RadioButton }
      },
      props,
      ...options
    })
  }

  describe('normalizedOptions computed property', () => {
    it('handles string array options', () => {
      const wrapper = mountComponent({
        modelValue: 'option1',
        options: ['option1', 'option2', 'option3'],
        id: 'test-radio'
      })

      const radioButtons = wrapper.findAllComponents(RadioButton)
      expect(radioButtons).toHaveLength(3)

      expect(radioButtons[0].props('value')).toBe('option1')
      expect(radioButtons[1].props('value')).toBe('option2')
      expect(radioButtons[2].props('value')).toBe('option3')

      const labels = wrapper.findAll('label')
      expect(labels[0].text()).toBe('option1')
      expect(labels[1].text()).toBe('option2')
      expect(labels[2].text()).toBe('option3')
    })

    it('handles SettingOption array', () => {
      const options: SettingOption[] = [
        { text: 'Small', value: 'sm' },
        { text: 'Medium', value: 'md' },
        { text: 'Large', value: 'lg' }
      ]

      const wrapper = mountComponent({
        modelValue: 'md',
        options,
        id: 'test-radio'
      })

      const radioButtons = wrapper.findAllComponents(RadioButton)
      expect(radioButtons).toHaveLength(3)

      expect(radioButtons[0].props('value')).toBe('sm')
      expect(radioButtons[1].props('value')).toBe('md')
      expect(radioButtons[2].props('value')).toBe('lg')

      const labels = wrapper.findAll('label')
      expect(labels[0].text()).toBe('Small')
      expect(labels[1].text()).toBe('Medium')
      expect(labels[2].text()).toBe('Large')
    })

    it('handles SettingOption with undefined value (uses text as value)', () => {
      const options: SettingOption[] = [
        { text: 'Option A', value: undefined },
        { text: 'Option B' }
      ]

      const wrapper = mountComponent({
        modelValue: 'Option A',
        options,
        id: 'test-radio'
      })

      const radioButtons = wrapper.findAllComponents(RadioButton)

      expect(radioButtons[0].props('value')).toBe('Option A')
      expect(radioButtons[1].props('value')).toBe('Option B')
    })

    it('handles custom object with optionLabel and optionValue', () => {
      const options = [
        { name: 'First Option', id: '1' },
        { name: 'Second Option', id: '2' },
        { name: 'Third Option', id: '3' }
      ]

      const wrapper = mountComponent({
        modelValue: 2,
        options,
        optionLabel: 'name',
        optionValue: 'id',
        id: 'test-radio'
      })

      const radioButtons = wrapper.findAllComponents(RadioButton)
      expect(radioButtons).toHaveLength(3)

      expect(radioButtons[0].props('value')).toBe('1')
      expect(radioButtons[1].props('value')).toBe('2')
      expect(radioButtons[2].props('value')).toBe('3')

      const labels = wrapper.findAll('label')
      expect(labels[0].text()).toBe('First Option')
      expect(labels[1].text()).toBe('Second Option')
      expect(labels[2].text()).toBe('Third Option')
    })

    it('handles mixed array with strings and SettingOptions', () => {
      const options: (string | SettingOption)[] = [
        'Simple String',
        { text: 'Complex Option', value: 'complex' },
        'Another String'
      ]

      const wrapper = mountComponent({
        modelValue: 'complex',
        options,
        id: 'test-radio'
      })

      const radioButtons = wrapper.findAllComponents(RadioButton)
      expect(radioButtons).toHaveLength(3)

      expect(radioButtons[0].props('value')).toBe('Simple String')
      expect(radioButtons[1].props('value')).toBe('complex')
      expect(radioButtons[2].props('value')).toBe('Another String')

      const labels = wrapper.findAll('label')
      expect(labels[0].text()).toBe('Simple String')
      expect(labels[1].text()).toBe('Complex Option')
      expect(labels[2].text()).toBe('Another String')
    })

    it('handles empty options array', () => {
      const wrapper = mountComponent({
        modelValue: null,
        options: [],
        id: 'test-radio'
      })

      const radioButtons = wrapper.findAllComponents(RadioButton)
      expect(radioButtons).toHaveLength(0)
    })

    it('handles undefined options gracefully', () => {
      const wrapper = mountComponent({
        modelValue: null,
        options: undefined,
        id: 'test-radio'
      })

      const radioButtons = wrapper.findAllComponents(RadioButton)
      expect(radioButtons).toHaveLength(0)
    })

    it('handles object with missing properties gracefully', () => {
      const options = [{ label: 'Option 1', val: 'opt1' }]

      const wrapper = mountComponent({
        modelValue: 'opt1',
        options,
        id: 'test-radio'
      })

      const radioButtons = wrapper.findAllComponents(RadioButton)
      expect(radioButtons).toHaveLength(1)

      const labels = wrapper.findAll('label')
      expect(labels[0].text()).toBe('Unknown')
    })
  })

  describe('component functionality', () => {
    it('sets correct input-id and name attributes', () => {
      const options = ['A', 'B']

      const wrapper = mountComponent({
        modelValue: 'A',
        options,
        id: 'my-radio-group'
      })

      const radioButtons = wrapper.findAllComponents(RadioButton)

      expect(radioButtons[0].props('inputId')).toBe('my-radio-group-A')
      expect(radioButtons[0].props('name')).toBe('my-radio-group')
      expect(radioButtons[1].props('inputId')).toBe('my-radio-group-B')
      expect(radioButtons[1].props('name')).toBe('my-radio-group')
    })

    it('associates labels with radio buttons correctly', () => {
      const options = ['Yes', 'No']

      const wrapper = mountComponent({
        modelValue: 'Yes',
        options,
        id: 'confirm-radio'
      })

      const labels = wrapper.findAll('label')

      expect(labels[0].attributes('for')).toBe('confirm-radio-Yes')
      expect(labels[1].attributes('for')).toBe('confirm-radio-No')
    })

    it('sets aria-describedby attribute correctly', () => {
      const options: SettingOption[] = [
        { text: 'Option 1', value: 'opt1' },
        { text: 'Option 2', value: 'opt2' }
      ]

      const wrapper = mountComponent({
        modelValue: 'opt1',
        options,
        id: 'test-radio'
      })

      const radioButtons = wrapper.findAllComponents(RadioButton)

      expect(radioButtons[0].attributes('aria-describedby')).toBe(
        'Option 1-label'
      )
      expect(radioButtons[1].attributes('aria-describedby')).toBe(
        'Option 2-label'
      )
    })
  })
})
