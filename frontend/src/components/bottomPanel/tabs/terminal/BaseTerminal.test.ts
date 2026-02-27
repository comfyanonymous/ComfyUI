import { createTestingPinia } from '@pinia/testing'
import type { VueWrapper } from '@vue/test-utils'
import { mount } from '@vue/test-utils'
import type { Mock } from 'vitest'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'
import { createI18n } from 'vue-i18n'

import BaseTerminal from '@/components/bottomPanel/tabs/terminal/BaseTerminal.vue'

// Mock xterm and related modules
vi.mock('@xterm/xterm', () => ({
  Terminal: vi.fn().mockImplementation(() => ({
    open: vi.fn(),
    dispose: vi.fn(),
    onSelectionChange: vi.fn(() => {
      // Return a disposable
      return {
        dispose: vi.fn()
      }
    }),
    hasSelection: vi.fn(() => false),
    getSelection: vi.fn(() => ''),
    selectAll: vi.fn(),
    clearSelection: vi.fn(),
    loadAddon: vi.fn()
  })),
  IDisposable: vi.fn()
}))

vi.mock('@xterm/addon-fit', () => ({
  FitAddon: vi.fn().mockImplementation(() => ({
    fit: vi.fn(),
    proposeDimensions: vi.fn(() => ({ rows: 24, cols: 80 }))
  }))
}))

const mockTerminal = {
  open: vi.fn(),
  dispose: vi.fn(),
  onSelectionChange: vi.fn(() => ({
    dispose: vi.fn()
  })),
  hasSelection: vi.fn(() => false),
  getSelection: vi.fn(() => ''),
  selectAll: vi.fn(),
  clearSelection: vi.fn()
}

vi.mock('@/composables/bottomPanelTabs/useTerminal', () => ({
  useTerminal: vi.fn(() => ({
    terminal: mockTerminal,
    useAutoSize: vi.fn(() => ({ resize: vi.fn() }))
  }))
}))

vi.mock('@/utils/envUtil', () => ({
  electronAPI: vi.fn(() => null)
}))

const mockData = vi.hoisted(() => ({ isDesktop: false }))

vi.mock('@/platform/distribution/types', () => ({
  get isDesktop() {
    return mockData.isDesktop
  }
}))

// Mock clipboard API
Object.defineProperty(navigator, 'clipboard', {
  value: {
    writeText: vi.fn().mockResolvedValue(undefined)
  },
  configurable: true
})

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {
      serverStart: {
        copySelectionTooltip: 'Copy selection',
        copyAllTooltip: 'Copy all'
      }
    }
  }
})

const mountBaseTerminal = () => {
  return mount(BaseTerminal, {
    global: {
      plugins: [
        createTestingPinia({
          createSpy: vi.fn
        }),
        i18n
      ],
      stubs: {
        Button: {
          template: '<button v-bind="$attrs"><slot /></button>',
          props: ['icon', 'severity', 'size']
        }
      }
    }
  })
}

describe('BaseTerminal', () => {
  let wrapper: VueWrapper<InstanceType<typeof BaseTerminal>> | undefined

  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    wrapper?.unmount()
  })

  it('emits created event on mount', () => {
    wrapper = mountBaseTerminal()

    expect(wrapper.emitted('created')).toBeTruthy()
    expect(wrapper.emitted('created')![0]).toHaveLength(2)
  })

  it('emits unmounted event on unmount', () => {
    wrapper = mountBaseTerminal()
    wrapper.unmount()

    expect(wrapper.emitted('unmounted')).toBeTruthy()
  })

  it('button exists and has correct initial state', async () => {
    wrapper = mountBaseTerminal()

    const button = wrapper.find('button[aria-label]')
    expect(button.exists()).toBe(true)

    expect(button.classes()).toContain('opacity-0')
    expect(button.classes()).toContain('pointer-events-none')
  })

  it('shows correct tooltip when no selection', async () => {
    mockTerminal.hasSelection.mockReturnValue(false)
    wrapper = mountBaseTerminal()

    await wrapper.trigger('mouseenter')
    await nextTick()

    const button = wrapper.find('button[aria-label]')
    expect(button.attributes('aria-label')).toBe('Copy all')
  })

  it('shows correct tooltip when selection exists', async () => {
    mockTerminal.hasSelection.mockReturnValue(true)
    wrapper = mountBaseTerminal()

    // Trigger the selection change callback that was registered during mount
    expect(mockTerminal.onSelectionChange).toHaveBeenCalled()
    // Access the mock calls - TypeScript can't infer the mock structure dynamically
    const mockCalls = (mockTerminal.onSelectionChange as Mock).mock.calls
    const selectionCallback = mockCalls[0][0] as () => void
    selectionCallback()
    await nextTick()

    await wrapper.trigger('mouseenter')
    await nextTick()

    const button = wrapper.find('button[aria-label]')
    expect(button.attributes('aria-label')).toBe('Copy selection')
  })

  it('copies selected text when selection exists', async () => {
    const selectedText = 'selected text'
    mockTerminal.hasSelection.mockReturnValue(true)
    mockTerminal.getSelection.mockReturnValue(selectedText)

    wrapper = mountBaseTerminal()

    await wrapper.trigger('mouseenter')
    await nextTick()

    const button = wrapper.find('button[aria-label]')
    await button.trigger('click')

    expect(mockTerminal.selectAll).not.toHaveBeenCalled()
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(selectedText)
    expect(mockTerminal.clearSelection).not.toHaveBeenCalled()
  })

  it('copies all text when no selection exists', async () => {
    const allText = 'all terminal content'
    mockTerminal.hasSelection.mockReturnValue(false)
    mockTerminal.getSelection
      .mockReturnValueOnce('') // First call returns empty (no selection)
      .mockReturnValueOnce(allText) // Second call after selectAll returns all text

    wrapper = mountBaseTerminal()

    await wrapper.trigger('mouseenter')
    await nextTick()

    const button = wrapper.find('button[aria-label]')
    await button.trigger('click')

    expect(mockTerminal.selectAll).toHaveBeenCalled()
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith(allText)
    expect(mockTerminal.clearSelection).toHaveBeenCalled()
  })

  it('does not copy when no text available', async () => {
    mockTerminal.hasSelection.mockReturnValue(false)
    mockTerminal.getSelection.mockReturnValue('')

    wrapper = mountBaseTerminal()

    await wrapper.trigger('mouseenter')
    await nextTick()

    const button = wrapper.find('button[aria-label]')
    await button.trigger('click')

    expect(mockTerminal.selectAll).toHaveBeenCalled()
    expect(navigator.clipboard.writeText).not.toHaveBeenCalled()
  })
})
