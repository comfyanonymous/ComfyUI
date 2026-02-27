import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

import ShortcutsList from '@/components/bottomPanel/tabs/shortcuts/ShortcutsList.vue'
import type { ComfyCommandImpl } from '@/stores/commandStore'

// Mock vue-i18n
const mockT = vi.fn((key: string) => {
  const translations: Record<string, string> = {
    'shortcuts.subcategories.workflow': 'Workflow',
    'shortcuts.subcategories.node': 'Node',
    'shortcuts.subcategories.queue': 'Queue',
    'shortcuts.subcategories.view': 'View',
    'shortcuts.subcategories.panelControls': 'Panel Controls',
    'commands.Workflow_New.label': 'New Blank Workflow'
  }
  return translations[key] || key
})

vi.mock('vue-i18n', () => ({
  useI18n: () => ({
    t: mockT
  })
}))

describe('ShortcutsList', () => {
  const mockCommands: ComfyCommandImpl[] = [
    {
      id: 'Workflow.New',
      label: 'New Workflow',
      category: 'essentials',
      keybinding: {
        combo: {
          getKeySequences: () => ['Control', 'n']
        }
      }
    } as ComfyCommandImpl,
    {
      id: 'Node.Add',
      label: 'Add Node',
      category: 'essentials',
      keybinding: {
        combo: {
          getKeySequences: () => ['Shift', 'a']
        }
      }
    } as ComfyCommandImpl,
    {
      id: 'Queue.Clear',
      label: 'Clear Queue',
      category: 'essentials',
      keybinding: {
        combo: {
          getKeySequences: () => ['Control', 'Shift', 'c']
        }
      }
    } as ComfyCommandImpl
  ]

  const mockSubcategories = {
    workflow: [mockCommands[0]],
    node: [mockCommands[1]],
    queue: [mockCommands[2]]
  }

  it('should render shortcuts organized by subcategories', () => {
    const wrapper = mount(ShortcutsList, {
      props: {
        commands: mockCommands,
        subcategories: mockSubcategories
      }
    })

    // Check that subcategories are rendered
    expect(wrapper.text()).toContain('Workflow')
    expect(wrapper.text()).toContain('Node')
    expect(wrapper.text()).toContain('Queue')

    // Check that commands are rendered
    expect(wrapper.text()).toContain('New Blank Workflow')
  })

  it('should format keyboard shortcuts correctly', () => {
    const wrapper = mount(ShortcutsList, {
      props: {
        commands: mockCommands,
        subcategories: mockSubcategories
      }
    })

    // Check for formatted keys
    expect(wrapper.text()).toContain('Ctrl')
    expect(wrapper.text()).toContain('n')
    expect(wrapper.text()).toContain('Shift')
    expect(wrapper.text()).toContain('a')
    expect(wrapper.text()).toContain('c')
  })

  it('should filter out commands without keybindings', () => {
    const commandsWithoutKeybinding: ComfyCommandImpl[] = [
      ...mockCommands,
      {
        id: 'No.Keybinding',
        label: 'No Keybinding',
        category: 'essentials',
        keybinding: null
      } as ComfyCommandImpl
    ]

    const wrapper = mount(ShortcutsList, {
      props: {
        commands: commandsWithoutKeybinding,
        subcategories: {
          ...mockSubcategories,
          other: [commandsWithoutKeybinding[3]]
        }
      }
    })

    expect(wrapper.text()).not.toContain('No Keybinding')
  })

  it('should handle special key formatting', () => {
    const specialKeyCommand: ComfyCommandImpl = {
      id: 'Special.Keys',
      label: 'Special Keys',
      category: 'essentials',
      keybinding: {
        combo: {
          getKeySequences: () => ['Meta', 'ArrowUp', 'Enter', 'Escape', ' ']
        }
      }
    } as ComfyCommandImpl

    const wrapper = mount(ShortcutsList, {
      props: {
        commands: [specialKeyCommand],
        subcategories: {
          special: [specialKeyCommand]
        }
      }
    })

    const text = wrapper.text()
    expect(text).toContain('Cmd') // Meta -> Cmd
    expect(text).toContain('↑') // ArrowUp -> ↑
    expect(text).toContain('↵') // Enter -> ↵
    expect(text).toContain('Esc') // Escape -> Esc
    expect(text).toContain('Space') // ' ' -> Space
  })

  it('should use fallback subcategory titles', () => {
    const wrapper = mount(ShortcutsList, {
      props: {
        commands: mockCommands,
        subcategories: {
          unknown: [mockCommands[0]]
        }
      }
    })

    expect(wrapper.text()).toContain('unknown')
  })
})
