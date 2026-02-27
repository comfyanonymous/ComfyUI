import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

import JobContextMenu from '@/components/queue/job/JobContextMenu.vue'
import type { MenuEntry } from '@/composables/queue/useJobMenu'

const buttonStub = {
  props: {
    disabled: {
      type: Boolean,
      default: false
    }
  },
  template: `
    <div
      class="button-stub"
      :data-disabled="String(disabled)"
    >
      <slot />
    </div>
  `
}

const createEntries = (): MenuEntry[] => [
  { key: 'enabled', label: 'Enabled action', onClick: vi.fn() },
  {
    key: 'disabled',
    label: 'Disabled action',
    disabled: true,
    onClick: vi.fn()
  },
  { kind: 'divider', key: 'divider-1' }
]

const mountComponent = (entries: MenuEntry[]) =>
  mount(JobContextMenu, {
    props: { entries },
    global: {
      stubs: {
        Popover: {
          template: '<div class="popover-stub"><slot /></div>'
        },
        Button: buttonStub
      }
    }
  })

describe('JobContextMenu', () => {
  it('passes disabled state to action buttons', () => {
    const wrapper = mountComponent(createEntries())

    const buttons = wrapper.findAll('.button-stub')
    expect(buttons).toHaveLength(2)
    expect(buttons[0].attributes('data-disabled')).toBe('false')
    expect(buttons[1].attributes('data-disabled')).toBe('true')
  })

  it('emits action for enabled entries', async () => {
    const entries = createEntries()
    const wrapper = mountComponent(entries)

    await wrapper.findAll('.button-stub')[0].trigger('click')

    expect(wrapper.emitted('action')).toEqual([[entries[0]]])
  })

  it('does not emit action for disabled entries', async () => {
    const wrapper = mountComponent([
      {
        key: 'disabled',
        label: 'Disabled action',
        disabled: true,
        onClick: vi.fn()
      }
    ])

    await wrapper.get('.button-stub').trigger('click')

    expect(wrapper.emitted('action')).toBeUndefined()
  })
})
