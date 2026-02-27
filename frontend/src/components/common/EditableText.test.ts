import { mount } from '@vue/test-utils'
import PrimeVue from 'primevue/config'
import InputText from 'primevue/inputtext'
import { beforeAll, describe, expect, it } from 'vitest'
import { createApp } from 'vue'

import EditableText from './EditableText.vue'

describe('EditableText', () => {
  beforeAll(() => {
    // Create a Vue app instance for PrimeVue
    const app = createApp({})
    app.use(PrimeVue)
  })

  // @ts-expect-error fixme ts strict error
  const mountComponent = (props, options = {}) => {
    return mount(EditableText, {
      global: {
        plugins: [PrimeVue],
        components: { InputText }
      },
      props,
      ...options
    })
  }

  it('renders span with modelValue when not editing', () => {
    const wrapper = mountComponent({
      modelValue: 'Test Text',
      isEditing: false
    })
    expect(wrapper.find('span').text()).toBe('Test Text')
    expect(wrapper.findComponent(InputText).exists()).toBe(false)
  })

  it('renders input with modelValue when editing', () => {
    const wrapper = mountComponent({
      modelValue: 'Test Text',
      isEditing: true
    })
    expect(wrapper.find('span').exists()).toBe(false)
    expect(wrapper.findComponent(InputText).props()['modelValue']).toBe(
      'Test Text'
    )
  })

  it('emits edit event when input is submitted', async () => {
    const wrapper = mountComponent({
      modelValue: 'Test Text',
      isEditing: true
    })
    await wrapper.findComponent(InputText).setValue('New Text')
    await wrapper.findComponent(InputText).trigger('keydown.enter')
    // Blur event should have been triggered
    expect(wrapper.findComponent(InputText).element).not.toBe(
      document.activeElement
    )
  })

  it('finishes editing on blur', async () => {
    const wrapper = mountComponent({
      modelValue: 'Test Text',
      isEditing: true
    })
    await wrapper.findComponent(InputText).trigger('blur')
    expect(wrapper.emitted('edit')).toBeTruthy()
    // @ts-expect-error fixme ts strict error
    expect(wrapper.emitted('edit')[0]).toEqual(['Test Text'])
  })

  it('cancels editing on escape key', async () => {
    const wrapper = mountComponent({
      modelValue: 'Original Text',
      isEditing: true
    })

    // Change the input value
    await wrapper.findComponent(InputText).setValue('Modified Text')

    // Press escape
    await wrapper.findComponent(InputText).trigger('keydown.escape')

    // Should emit cancel event
    expect(wrapper.emitted('cancel')).toBeTruthy()

    // Should NOT emit edit event
    expect(wrapper.emitted('edit')).toBeFalsy()

    // Input value should be reset to original
    expect(wrapper.findComponent(InputText).props()['modelValue']).toBe(
      'Original Text'
    )
  })

  it('does not save changes when escape is pressed and blur occurs', async () => {
    const wrapper = mountComponent({
      modelValue: 'Original Text',
      isEditing: true
    })

    // Change the input value
    await wrapper.findComponent(InputText).setValue('Modified Text')

    // Press escape (which triggers blur internally)
    await wrapper.findComponent(InputText).trigger('keydown.escape')

    // Manually trigger blur to simulate the blur that happens after escape
    await wrapper.findComponent(InputText).trigger('blur')

    // Should emit cancel but not edit
    expect(wrapper.emitted('cancel')).toBeTruthy()
    expect(wrapper.emitted('edit')).toBeFalsy()
  })

  it('saves changes on enter but not on escape', async () => {
    // Test Enter key saves changes
    const enterWrapper = mountComponent({
      modelValue: 'Original Text',
      isEditing: true
    })
    await enterWrapper.findComponent(InputText).setValue('Saved Text')
    await enterWrapper.findComponent(InputText).trigger('keydown.enter')
    // Trigger blur that happens after enter
    await enterWrapper.findComponent(InputText).trigger('blur')
    expect(enterWrapper.emitted('edit')).toBeTruthy()
    // @ts-expect-error fixme ts strict error
    expect(enterWrapper.emitted('edit')[0]).toEqual(['Saved Text'])

    // Test Escape key cancels changes with a fresh wrapper
    const escapeWrapper = mountComponent({
      modelValue: 'Original Text',
      isEditing: true
    })
    await escapeWrapper.findComponent(InputText).setValue('Cancelled Text')
    await escapeWrapper.findComponent(InputText).trigger('keydown.escape')
    expect(escapeWrapper.emitted('cancel')).toBeTruthy()
    expect(escapeWrapper.emitted('edit')).toBeFalsy()
  })
})
