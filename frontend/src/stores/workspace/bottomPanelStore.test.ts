import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useBottomPanelStore } from '@/stores/workspace/bottomPanelStore'
import type { BottomPanelExtension } from '@/types/extensionTypes'

// Mock dependencies
vi.mock('@/composables/bottomPanelTabs/useShortcutsTab', () => ({
  useShortcutsTab: () => [
    {
      id: 'shortcuts-essentials',
      title: 'Essentials',
      component: {},
      type: 'vue',
      targetPanel: 'shortcuts'
    },
    {
      id: 'shortcuts-view-controls',
      title: 'View Controls',
      component: {},
      type: 'vue',
      targetPanel: 'shortcuts'
    }
  ]
}))

vi.mock('@/composables/bottomPanelTabs/useTerminalTabs', () => ({
  useLogsTerminalTab: () => ({
    id: 'logs',
    title: 'Logs',
    component: {},
    type: 'vue',
    targetPanel: 'terminal'
  }),
  useCommandTerminalTab: () => ({
    id: 'command',
    title: 'Command',
    component: {},
    type: 'vue',
    targetPanel: 'terminal'
  })
}))

vi.mock('@/stores/commandStore', () => ({
  useCommandStore: () => ({
    registerCommand: vi.fn()
  })
}))

const mockData = vi.hoisted(() => ({ isDesktop: false }))

vi.mock('@/platform/distribution/types', () => ({
  get isDesktop() {
    return mockData.isDesktop
  }
}))

describe('useBottomPanelStore', () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
  })

  it('should initialize with empty panels', () => {
    const store = useBottomPanelStore()

    expect(store.activePanel).toBeNull()
    expect(store.bottomPanelVisible).toBe(false)
    expect(store.bottomPanelTabs).toEqual([])
    expect(store.activeBottomPanelTab).toBeNull()
  })

  it('should register bottom panel tabs', () => {
    const store = useBottomPanelStore()
    const tab: BottomPanelExtension = {
      id: 'test-tab',
      title: 'Test Tab',
      component: {},
      type: 'vue',
      targetPanel: 'terminal'
    }

    store.registerBottomPanelTab(tab)

    expect(store.panels.terminal.tabs.find((t) => t.id === 'test-tab')).toEqual(
      tab
    )
    expect(store.panels.terminal.activeTabId).toBe('test-tab')
  })

  it('should toggle panel visibility', () => {
    const store = useBottomPanelStore()
    const tab: BottomPanelExtension = {
      id: 'test-tab',
      title: 'Test Tab',
      component: {},
      type: 'vue',
      targetPanel: 'shortcuts'
    }

    store.registerBottomPanelTab(tab)

    // Panel should be hidden initially
    expect(store.activePanel).toBeNull()

    // Toggle should show panel
    store.togglePanel('shortcuts')
    expect(store.activePanel).toBe('shortcuts')
    expect(store.bottomPanelVisible).toBe(true)

    // Toggle again should hide panel
    store.togglePanel('shortcuts')
    expect(store.activePanel).toBeNull()
    expect(store.bottomPanelVisible).toBe(false)
  })

  it('should switch between panel types', () => {
    const store = useBottomPanelStore()

    const terminalTab: BottomPanelExtension = {
      id: 'terminal-tab',
      title: 'Terminal',
      component: {},
      type: 'vue',
      targetPanel: 'terminal'
    }

    const shortcutsTab: BottomPanelExtension = {
      id: 'shortcuts-tab',
      title: 'Shortcuts',
      component: {},
      type: 'vue',
      targetPanel: 'shortcuts'
    }

    store.registerBottomPanelTab(terminalTab)
    store.registerBottomPanelTab(shortcutsTab)

    // Show terminal panel
    store.togglePanel('terminal')
    expect(store.activePanel).toBe('terminal')
    expect(store.activeBottomPanelTab?.id).toBe('terminal-tab')

    // Switch to shortcuts panel
    store.togglePanel('shortcuts')
    expect(store.activePanel).toBe('shortcuts')
    expect(store.activeBottomPanelTab?.id).toBe('shortcuts-tab')
  })

  it('should toggle specific tabs', () => {
    const store = useBottomPanelStore()
    const tab: BottomPanelExtension = {
      id: 'specific-tab',
      title: 'Specific Tab',
      component: {},
      type: 'vue',
      targetPanel: 'shortcuts'
    }

    store.registerBottomPanelTab(tab)

    // Toggle specific tab should show it
    store.toggleBottomPanelTab('specific-tab')
    expect(store.activePanel).toBe('shortcuts')
    expect(store.panels.shortcuts.activeTabId).toBe('specific-tab')

    // Toggle same tab again should hide panel
    store.toggleBottomPanelTab('specific-tab')
    expect(store.activePanel).toBeNull()
  })
})
