import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import MarqueeLine from './MarqueeLine.vue'

describe(MarqueeLine, () => {
  it('renders slot content', () => {
    const wrapper = mount(MarqueeLine, {
      slots: { default: 'Hello World' }
    })
    expect(wrapper.text()).toBe('Hello World')
  })

  it('renders content inside a span within the container', () => {
    const wrapper = mount(MarqueeLine, {
      slots: { default: 'Test Text' }
    })
    const span = wrapper.find('span')
    expect(span.exists()).toBe(true)
    expect(span.text()).toBe('Test Text')
  })
})
