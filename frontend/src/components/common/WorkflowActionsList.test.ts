import { shallowMount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

import WorkflowActionsList from '@/components/common/WorkflowActionsList.vue'
import type {
  WorkflowMenuAction,
  WorkflowMenuItem
} from '@/types/workflowMenuItem'

function createWrapper(items: WorkflowMenuItem[]) {
  return shallowMount(WorkflowActionsList, {
    props: { items },
    global: { renderStubDefaultSlot: true }
  })
}

describe('WorkflowActionsList', () => {
  it('renders action items with label and icon', () => {
    const items: WorkflowMenuItem[] = [
      { label: 'Save', icon: 'pi pi-save', command: vi.fn() }
    ]

    const wrapper = createWrapper(items)

    expect(wrapper.text()).toContain('Save')
    expect(wrapper.find('.pi-save').exists()).toBe(true)
  })

  it('renders separator items', () => {
    const items: WorkflowMenuItem[] = [
      { label: 'Before', icon: 'pi pi-a', command: vi.fn() },
      { separator: true },
      { label: 'After', icon: 'pi pi-b', command: vi.fn() }
    ]

    const wrapper = createWrapper(items)
    const html = wrapper.html()

    expect(html).toContain('dropdown-menu-separator-stub')
    expect(wrapper.text()).toContain('Before')
    expect(wrapper.text()).toContain('After')
  })

  it('dispatches command on select', async () => {
    const command = vi.fn()
    const items: WorkflowMenuItem[] = [
      { label: 'Action', icon: 'pi pi-play', command }
    ]

    const wrapper = createWrapper(items)
    const item = wrapper.findComponent({ name: 'DropdownMenuItem' })
    await item.vm.$emit('select')

    expect(command).toHaveBeenCalledOnce()
  })

  it('renders badge when present', () => {
    const items: WorkflowMenuItem[] = [
      {
        label: 'New Feature',
        icon: 'pi pi-star',
        command: vi.fn(),
        badge: 'NEW'
      }
    ]

    const wrapper = createWrapper(items)

    expect(wrapper.text()).toContain('NEW')
  })

  it('does not render badge when absent', () => {
    const items: WorkflowMenuAction[] = [
      { label: 'Plain', icon: 'pi pi-check', command: vi.fn() }
    ]

    const wrapper = createWrapper(items)

    expect(wrapper.text()).not.toContain('NEW')
  })
})
