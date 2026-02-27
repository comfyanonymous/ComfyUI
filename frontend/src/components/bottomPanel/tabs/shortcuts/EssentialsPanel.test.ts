import { mount } from '@vue/test-utils'
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import EssentialsPanel from '@/components/bottomPanel/tabs/shortcuts/EssentialsPanel.vue'
import ShortcutsList from '@/components/bottomPanel/tabs/shortcuts/ShortcutsList.vue'
import type { ComfyCommandImpl } from '@/stores/commandStore'

// Mock ShortcutsList component
vi.mock('@/components/bottomPanel/tabs/shortcuts/ShortcutsList.vue', () => ({
  default: {
    name: 'ShortcutsList',
    props: ['commands', 'subcategories', 'columns'],
    template:
      '<div class="shortcuts-list-mock">{{ commands.length }} commands</div>'
  }
}))

// Mock command store
const mockCommands: ComfyCommandImpl[] = [
  {
    id: 'Workflow.New',
    label: 'New Workflow',
    category: 'essentials'
  } as ComfyCommandImpl,
  {
    id: 'Node.Add',
    label: 'Add Node',
    category: 'essentials'
  } as ComfyCommandImpl,
  {
    id: 'Queue.Clear',
    label: 'Clear Queue',
    category: 'essentials'
  } as ComfyCommandImpl,
  {
    id: 'Other.Command',
    label: 'Other Command',
    category: 'view-controls',
    function: vi.fn(),
    icon: 'pi pi-test',
    tooltip: 'Test tooltip',
    menubarLabel: 'Other Command',
    keybinding: null
  } as ComfyCommandImpl
]

vi.mock('@/stores/commandStore', () => ({
  useCommandStore: () => ({
    commands: mockCommands
  })
}))

describe('EssentialsPanel', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('should render ShortcutsList with essentials commands', () => {
    const wrapper = mount(EssentialsPanel)

    const shortcutsList = wrapper.findComponent(ShortcutsList)
    expect(shortcutsList.exists()).toBe(true)
  })

  it('should categorize commands into subcategories', () => {
    const wrapper = mount(EssentialsPanel)

    const shortcutsList = wrapper.findComponent(ShortcutsList)
    const subcategories = shortcutsList.props('subcategories')

    expect(subcategories).toHaveProperty('workflow')
    expect(subcategories).toHaveProperty('node')
    expect(subcategories).toHaveProperty('queue')

    expect(subcategories.workflow).toContain(mockCommands[0])
    expect(subcategories.node).toContain(mockCommands[1])
    expect(subcategories.queue).toContain(mockCommands[2])
  })
})
