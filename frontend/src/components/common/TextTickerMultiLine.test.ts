import { mount } from '@vue/test-utils'
import { nextTick } from 'vue'
import { afterEach, describe, expect, it, vi } from 'vitest'

import MarqueeLine from './MarqueeLine.vue'
import TextTickerMultiLine from './TextTickerMultiLine.vue'

type Callback = () => void

const resizeCallbacks: Callback[] = []
const mutationCallbacks: Callback[] = []

vi.mock('@vueuse/core', async () => {
  const actual = await vi.importActual('@vueuse/core')
  return {
    ...actual,
    useResizeObserver: (_target: unknown, cb: Callback) => {
      resizeCallbacks.push(cb)
      return { stop: vi.fn() }
    },
    useMutationObserver: (_target: unknown, cb: Callback) => {
      mutationCallbacks.push(cb)
      return { stop: vi.fn() }
    }
  }
})

function mockElementSize(
  el: HTMLElement,
  clientWidth: number,
  scrollWidth: number
) {
  Object.defineProperty(el, 'clientWidth', {
    value: clientWidth,
    configurable: true
  })
  Object.defineProperty(el, 'scrollWidth', {
    value: scrollWidth,
    configurable: true
  })
}

describe(TextTickerMultiLine, () => {
  let wrapper: ReturnType<typeof mount>

  afterEach(() => {
    wrapper?.unmount()
    resizeCallbacks.length = 0
    mutationCallbacks.length = 0
  })

  function mountComponent(text: string) {
    wrapper = mount(TextTickerMultiLine, {
      slots: { default: text }
    })
    return wrapper
  }

  function getMeasureEl(): HTMLElement {
    return wrapper.find('[aria-hidden="true"]').element as HTMLElement
  }

  async function triggerSplitLines() {
    resizeCallbacks.forEach((cb) => cb())
    await nextTick()
  }

  it('renders slot content', () => {
    mountComponent('Load Checkpoint')
    expect(wrapper.text()).toContain('Load Checkpoint')
  })

  it('renders a single MarqueeLine when text fits', async () => {
    mountComponent('Short')
    mockElementSize(getMeasureEl(), 200, 100)
    await triggerSplitLines()

    expect(wrapper.findAllComponents(MarqueeLine)).toHaveLength(1)
  })

  it('renders two MarqueeLines when text overflows', async () => {
    mountComponent('Load Checkpoint Loader Simple')
    mockElementSize(getMeasureEl(), 100, 300)
    await triggerSplitLines()

    expect(wrapper.findAllComponents(MarqueeLine)).toHaveLength(2)
  })

  it('splits text at word boundary when overflowing', async () => {
    mountComponent('Load Checkpoint Loader')
    mockElementSize(getMeasureEl(), 100, 200)
    await triggerSplitLines()

    const lines = wrapper.findAllComponents(MarqueeLine)
    expect(lines[0].text()).toBe('Load')
    expect(lines[1].text()).toBe('Checkpoint Loader')
  })

  it('has hidden measurement element with aria-hidden', () => {
    mountComponent('Test')
    const measureEl = wrapper.find('[aria-hidden="true"]')
    expect(measureEl.exists()).toBe(true)
    expect(measureEl.classes()).toContain('invisible')
  })
})
