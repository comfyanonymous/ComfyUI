import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import BadgePill from './BadgePill.vue'

describe('BadgePill', () => {
  it('renders text content', () => {
    const wrapper = mount(BadgePill, {
      props: { text: 'Test Badge' }
    })
    expect(wrapper.text()).toBe('Test Badge')
  })

  it('renders icon when provided', () => {
    const wrapper = mount(BadgePill, {
      props: { icon: 'icon-[comfy--credits]', text: 'Credits' }
    })
    expect(wrapper.find('i.icon-\\[comfy--credits\\]').exists()).toBe(true)
  })

  it('applies iconClass to icon', () => {
    const wrapper = mount(BadgePill, {
      props: {
        icon: 'icon-[comfy--credits]',
        iconClass: 'text-amber-400'
      }
    })
    const icon = wrapper.find('i')
    expect(icon.classes()).toContain('text-amber-400')
  })

  it('uses default border color when no borderStyle', () => {
    const wrapper = mount(BadgePill, {
      props: { text: 'Default' }
    })
    expect(wrapper.attributes('style')).toContain(
      'border-color: var(--border-color)'
    )
  })

  it('applies solid border color when borderStyle is a color', () => {
    const wrapper = mount(BadgePill, {
      props: { text: 'Colored', borderStyle: '#f59e0b' }
    })
    expect(wrapper.attributes('style')).toContain('border-color: #f59e0b')
  })

  it('applies gradient border when borderStyle contains linear-gradient', () => {
    const gradient = 'linear-gradient(90deg, #3186FF, #FABC12)'
    const wrapper = mount(BadgePill, {
      props: { text: 'Gradient', borderStyle: gradient }
    })
    const element = wrapper.element as HTMLElement
    expect(element.style.borderColor).toBe('transparent')
    expect(element.style.backgroundOrigin).toBe('border-box')
    expect(element.style.backgroundClip).toBe('padding-box, border-box')
  })

  it('applies filled style with background and text color', () => {
    const wrapper = mount(BadgePill, {
      props: { text: 'Filled', borderStyle: '#f59e0b', filled: true }
    })
    const style = wrapper.attributes('style')
    expect(style).toContain('border-color: #f59e0b')
    expect(style).toContain('background-color: #f59e0b33')
    expect(style).toContain('color: #f59e0b')
  })

  it('has foreground text when not filled', () => {
    const wrapper = mount(BadgePill, {
      props: { text: 'Not Filled', borderStyle: '#f59e0b' }
    })
    expect(wrapper.classes()).toContain('text-foreground')
  })

  it('does not have foreground text class when filled', () => {
    const wrapper = mount(BadgePill, {
      props: { text: 'Filled', borderStyle: '#f59e0b', filled: true }
    })
    expect(wrapper.classes()).not.toContain('text-foreground')
  })

  it('renders slot content', () => {
    const wrapper = mount(BadgePill, {
      slots: { default: 'Slot Content' }
    })
    expect(wrapper.text()).toBe('Slot Content')
  })
})
