import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import TextTicker from './TextTicker.vue'

function mockScrollWidth(el: HTMLElement, scrollWidth: number) {
  Object.defineProperty(el, 'scrollWidth', {
    value: scrollWidth,
    configurable: true
  })
}

describe(TextTicker, () => {
  let rafCallbacks: ((time: number) => void)[]
  let wrapper: ReturnType<typeof mount>

  beforeEach(() => {
    vi.useFakeTimers()
    rafCallbacks = []
    vi.spyOn(window, 'requestAnimationFrame').mockImplementation((cb) => {
      rafCallbacks.push(cb)
      return rafCallbacks.length
    })
    vi.spyOn(window, 'cancelAnimationFrame').mockImplementation(() => {})
  })

  afterEach(() => {
    wrapper?.unmount()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('renders slot content', () => {
    wrapper = mount(TextTicker, {
      slots: { default: 'Hello World' }
    })
    expect(wrapper.text()).toBe('Hello World')
  })

  it('scrolls on hover after delay', async () => {
    wrapper = mount(TextTicker, {
      slots: { default: 'Very long text that overflows' },
      props: { speed: 100 }
    })

    const el = wrapper.element as HTMLElement
    mockScrollWidth(el, 300)

    await nextTick()
    await wrapper.trigger('mouseenter')
    await nextTick()

    expect(rafCallbacks.length).toBe(0)

    vi.advanceTimersByTime(350)
    await nextTick()
    expect(rafCallbacks.length).toBeGreaterThan(0)

    rafCallbacks[0](performance.now() + 500)
    expect(el.scrollLeft).toBeGreaterThan(0)
  })

  it('cancels delayed scroll on mouse leave before delay elapses', async () => {
    wrapper = mount(TextTicker, {
      slots: { default: 'Very long text that overflows' },
      props: { speed: 100 }
    })

    mockScrollWidth(wrapper.element as HTMLElement, 300)

    await nextTick()
    await wrapper.trigger('mouseenter')
    await nextTick()

    vi.advanceTimersByTime(200)
    await wrapper.trigger('mouseleave')
    await nextTick()

    vi.advanceTimersByTime(350)
    await nextTick()
    expect(rafCallbacks.length).toBe(0)
  })

  it('resets scroll position on mouse leave', async () => {
    wrapper = mount(TextTicker, {
      slots: { default: 'Very long text that overflows' },
      props: { speed: 100 }
    })

    const el = wrapper.element as HTMLElement
    mockScrollWidth(el, 300)

    await nextTick()
    await wrapper.trigger('mouseenter')
    await nextTick()
    vi.advanceTimersByTime(350)
    await nextTick()

    rafCallbacks[0](performance.now() + 500)
    expect(el.scrollLeft).toBeGreaterThan(0)

    await wrapper.trigger('mouseleave')
    await nextTick()

    expect(el.scrollLeft).toBe(0)
  })

  it('does not scroll when content fits', async () => {
    wrapper = mount(TextTicker, {
      slots: { default: 'Short' }
    })

    await nextTick()
    await wrapper.trigger('mouseenter')
    await nextTick()
    vi.advanceTimersByTime(350)
    await nextTick()

    expect(rafCallbacks.length).toBe(0)
  })
})
