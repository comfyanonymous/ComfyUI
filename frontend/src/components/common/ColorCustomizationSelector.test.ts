import { mount } from '@vue/test-utils'
import ColorPicker from 'primevue/colorpicker'
import PrimeVue from 'primevue/config'
import SelectButton from 'primevue/selectbutton'
import { beforeEach, describe, expect, it } from 'vitest'
import { createApp, nextTick } from 'vue'

import ColorCustomizationSelector from './ColorCustomizationSelector.vue'

describe('ColorCustomizationSelector', () => {
  const colorOptions = [
    { name: 'Blue', value: '#0d6efd' },
    { name: 'Green', value: '#28a745' }
  ]

  beforeEach(() => {
    // Setup PrimeVue
    const app = createApp({})
    app.use(PrimeVue)
  })

  const mountComponent = (props = {}) => {
    return mount(ColorCustomizationSelector, {
      global: {
        plugins: [PrimeVue],
        components: { SelectButton, ColorPicker }
      },
      props: {
        modelValue: null,
        colorOptions,
        ...props
      }
    })
  }

  it('renders predefined color options and custom option', () => {
    const wrapper = mountComponent()
    const selectButton = wrapper.findComponent(SelectButton)

    expect(selectButton.props('options')).toHaveLength(colorOptions.length + 1)
    expect(selectButton.props('options')?.at(-1)?.name).toBe('_custom')
  })

  it('initializes with predefined color when provided', async () => {
    const wrapper = mountComponent({
      modelValue: '#0d6efd'
    })

    await nextTick()
    const selectButton = wrapper.findComponent(SelectButton)
    expect(selectButton.props('modelValue')).toEqual({
      name: 'Blue',
      value: '#0d6efd'
    })
  })

  it('initializes with custom color when non-predefined color provided', async () => {
    const wrapper = mountComponent({
      modelValue: '#123456'
    })

    await nextTick()
    const selectButton = wrapper.findComponent(SelectButton)
    const colorPicker = wrapper.findComponent(ColorPicker)

    expect(selectButton.props('modelValue').name).toBe('_custom')
    expect(colorPicker.props('modelValue')).toBe('123456')
  })

  it('shows color picker when custom option is selected', async () => {
    const wrapper = mountComponent()
    const selectButton = wrapper.findComponent(SelectButton)

    // Select custom option
    await selectButton.setValue({ name: '_custom', value: '' })

    expect(wrapper.findComponent(ColorPicker).exists()).toBe(true)
  })

  it('emits update when predefined color is selected', async () => {
    const wrapper = mountComponent()
    const selectButton = wrapper.findComponent(SelectButton)

    await selectButton.setValue(colorOptions[0])

    expect(wrapper.emitted('update:modelValue')?.[0]).toEqual(['#0d6efd'])
  })

  it('emits update when custom color is changed', async () => {
    const wrapper = mountComponent()
    const selectButton = wrapper.findComponent(SelectButton)

    // Select custom option
    await selectButton.setValue({ name: '_custom', value: '' })

    // Change custom color
    const colorPicker = wrapper.findComponent(ColorPicker)
    await colorPicker.setValue('ff0000')

    expect(wrapper.emitted('update:modelValue')?.[0]).toEqual(['#ff0000'])
  })

  it('inherits color from previous selection when switching to custom', async () => {
    const wrapper = mountComponent()
    const selectButton = wrapper.findComponent(SelectButton)

    // First select a predefined color
    await selectButton.setValue(colorOptions[0])

    // Then switch to custom
    await selectButton.setValue({ name: '_custom', value: '' })

    const colorPicker = wrapper.findComponent(ColorPicker)
    expect(colorPicker.props('modelValue')).toBe('0d6efd')
  })

  it('handles null modelValue correctly', async () => {
    const wrapper = mountComponent({
      modelValue: null
    })

    await nextTick()
    const selectButton = wrapper.findComponent(SelectButton)
    expect(selectButton.props('modelValue')).toEqual({
      name: '_custom',
      value: ''
    })
  })
})
