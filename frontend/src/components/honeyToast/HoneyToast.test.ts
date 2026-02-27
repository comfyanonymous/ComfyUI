import type { VueWrapper } from '@vue/test-utils'
import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { defineComponent, h, nextTick, ref } from 'vue'

import HoneyToast from './HoneyToast.vue'

describe('HoneyToast', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    document.body.innerHTML = ''
  })

  function mountComponent(
    props: { visible: boolean; expanded?: boolean } = { visible: true }
  ): VueWrapper {
    return mount(HoneyToast, {
      props,
      slots: {
        default: (slotProps: { isExpanded: boolean }) =>
          h(
            'div',
            { 'data-testid': 'content' },
            slotProps.isExpanded ? 'expanded' : 'collapsed'
          ),
        footer: (slotProps: { isExpanded: boolean; toggle: () => void }) =>
          h(
            'button',
            {
              'data-testid': 'toggle-btn',
              onClick: slotProps.toggle
            },
            slotProps.isExpanded ? 'Collapse' : 'Expand'
          )
      },
      attachTo: document.body
    })
  }

  it('renders when visible is true', async () => {
    const wrapper = mountComponent({ visible: true })
    await nextTick()

    const toast = document.body.querySelector('[role="status"]')
    expect(toast).toBeTruthy()

    wrapper.unmount()
  })

  it('does not render when visible is false', async () => {
    const wrapper = mountComponent({ visible: false })
    await nextTick()

    const toast = document.body.querySelector('[role="status"]')
    expect(toast).toBeFalsy()

    wrapper.unmount()
  })

  it('passes is-expanded=false to slots by default', async () => {
    const wrapper = mountComponent({ visible: true })
    await nextTick()

    const content = document.body.querySelector('[data-testid="content"]')
    expect(content?.textContent).toBe('collapsed')

    wrapper.unmount()
  })

  it('has aria-live="polite" for accessibility', async () => {
    const wrapper = mountComponent({ visible: true })
    await nextTick()

    const toast = document.body.querySelector('[role="status"]')
    expect(toast?.getAttribute('aria-live')).toBe('polite')

    wrapper.unmount()
  })

  it('supports v-model:expanded with reactive parent state', async () => {
    const TestWrapper = defineComponent({
      components: { HoneyToast },
      setup() {
        const expanded = ref(false)
        return { expanded }
      },
      template: `
        <HoneyToast :visible="true" v-model:expanded="expanded">
          <template #default="slotProps">
            <div data-testid="content">{{ slotProps.isExpanded ? 'expanded' : 'collapsed' }}</div>
          </template>
          <template #footer="slotProps">
            <button data-testid="toggle-btn" @click="slotProps.toggle">
              {{ slotProps.isExpanded ? 'Collapse' : 'Expand' }}
            </button>
          </template>
        </HoneyToast>
      `
    })

    const wrapper = mount(TestWrapper, { attachTo: document.body })
    await nextTick()

    const content = document.body.querySelector('[data-testid="content"]')
    expect(content?.textContent).toBe('collapsed')

    const toggleBtn = document.body.querySelector(
      '[data-testid="toggle-btn"]'
    ) as HTMLButtonElement
    expect(toggleBtn?.textContent?.trim()).toBe('Expand')

    toggleBtn?.click()
    await nextTick()

    expect(content?.textContent).toBe('expanded')
    expect(toggleBtn?.textContent?.trim()).toBe('Collapse')

    wrapper.unmount()
  })
})
