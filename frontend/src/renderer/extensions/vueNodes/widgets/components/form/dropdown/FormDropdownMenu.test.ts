import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { nextTick } from 'vue'

import FormDropdownMenu from './FormDropdownMenu.vue'
import type { FormDropdownItem, LayoutMode } from './types'

function createItem(id: string, name: string): FormDropdownItem {
  return {
    id,
    preview_url: '',
    name,
    label: name
  }
}

describe('FormDropdownMenu', () => {
  const defaultProps = {
    items: [createItem('1', 'Item 1'), createItem('2', 'Item 2')],
    isSelected: () => false,
    filterOptions: [],
    sortOptions: []
  }

  it('renders empty state when no items', async () => {
    const wrapper = mount(FormDropdownMenu, {
      props: {
        ...defaultProps,
        items: []
      },
      global: {
        stubs: {
          FormDropdownMenuFilter: true,
          FormDropdownMenuActions: true,
          VirtualGrid: true
        },
        mocks: {
          $t: (key: string) => key
        }
      }
    })

    await nextTick()

    const emptyIcon = wrapper.find('.icon-\\[lucide--circle-off\\]')
    expect(emptyIcon.exists()).toBe(true)
  })

  it('renders VirtualGrid when items exist', async () => {
    const wrapper = mount(FormDropdownMenu, {
      props: defaultProps,
      global: {
        stubs: {
          FormDropdownMenuFilter: true,
          FormDropdownMenuActions: true,
          VirtualGrid: true
        }
      }
    })

    await nextTick()

    const virtualGrid = wrapper.findComponent({ name: 'VirtualGrid' })
    expect(virtualGrid.exists()).toBe(true)
  })

  it('transforms items to include key property for VirtualGrid', async () => {
    const items = [createItem('1', 'Item 1'), createItem('2', 'Item 2')]
    const wrapper = mount(FormDropdownMenu, {
      props: {
        ...defaultProps,
        items
      },
      global: {
        stubs: {
          FormDropdownMenuFilter: true,
          FormDropdownMenuActions: true,
          VirtualGrid: true
        }
      }
    })

    await nextTick()

    const virtualGrid = wrapper.findComponent({ name: 'VirtualGrid' })
    const virtualItems = virtualGrid.props('items')

    expect(virtualItems).toHaveLength(2)
    expect(virtualItems[0]).toHaveProperty('key', '1')
    expect(virtualItems[1]).toHaveProperty('key', '2')
  })

  it('uses single column layout for list modes', async () => {
    const wrapper = mount(FormDropdownMenu, {
      props: {
        ...defaultProps,
        layoutMode: 'list' as LayoutMode
      },
      global: {
        stubs: {
          FormDropdownMenuFilter: true,
          FormDropdownMenuActions: true,
          VirtualGrid: true
        }
      }
    })

    await nextTick()

    const virtualGrid = wrapper.findComponent({ name: 'VirtualGrid' })
    expect(virtualGrid.props('maxColumns')).toBe(1)
  })
})
