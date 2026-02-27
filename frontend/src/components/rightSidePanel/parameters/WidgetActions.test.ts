import { createTestingPinia } from '@pinia/testing'
import { mount } from '@vue/test-utils'
import { setActivePinia } from 'pinia'
import type { Slots } from 'vue'
import { h } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'

import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'

import WidgetActions from './WidgetActions.vue'

const { mockGetInputSpecForWidget } = vi.hoisted(() => ({
  mockGetInputSpecForWidget: vi.fn()
}))

vi.mock('@/stores/nodeDefStore', () => ({
  useNodeDefStore: () => ({
    getInputSpecForWidget: mockGetInputSpecForWidget
  })
}))

vi.mock('@/renderer/core/canvas/canvasStore', () => ({
  useCanvasStore: () => ({
    canvas: { setDirty: vi.fn() }
  })
}))

vi.mock('@/stores/workspace/favoritedWidgetsStore', () => ({
  useFavoritedWidgetsStore: () => ({
    isFavorited: vi.fn().mockReturnValue(false),
    toggleFavorite: vi.fn()
  })
}))

vi.mock('@/services/dialogService', () => ({
  useDialogService: () => ({
    prompt: vi.fn()
  })
}))

vi.mock('@/components/button/MoreButton.vue', () => ({
  default: (_: unknown, { slots }: { slots: Slots }) =>
    h('div', slots.default?.({ close: () => {} }))
}))

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {
      g: {
        rename: 'Rename',
        enterNewName: 'Enter new name'
      },
      rightSidePanel: {
        hideInput: 'Hide input',
        showInput: 'Show input',
        addFavorite: 'Favorite',
        removeFavorite: 'Unfavorite',
        resetToDefault: 'Reset to default'
      }
    }
  }
})

describe('WidgetActions', () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    vi.resetAllMocks()
    mockGetInputSpecForWidget.mockReturnValue({
      type: 'INT',
      default: 42
    })
  })

  function createMockWidget(
    value: number = 100,
    callback?: () => void
  ): IBaseWidget {
    return {
      name: 'test_widget',
      type: 'number',
      value,
      label: 'Test Widget',
      options: {},
      y: 0,
      callback
    } as IBaseWidget
  }

  function createMockNode(): LGraphNode {
    return {
      id: 1,
      type: 'TestNode'
    } as LGraphNode
  }

  function mountWidgetActions(widget: IBaseWidget, node: LGraphNode) {
    return mount(WidgetActions, {
      props: {
        widget,
        node,
        label: 'Test Widget'
      },
      global: {
        plugins: [i18n]
      }
    })
  }

  it('shows reset button when widget has default value', () => {
    const widget = createMockWidget()
    const node = createMockNode()

    const wrapper = mountWidgetActions(widget, node)

    const resetButton = wrapper
      .findAll('button')
      .find((b) => b.text().includes('Reset'))
    expect(resetButton).toBeDefined()
  })

  it('emits resetToDefault with default value when reset button clicked', async () => {
    const widget = createMockWidget(100)
    const node = createMockNode()

    const wrapper = mountWidgetActions(widget, node)

    const resetButton = wrapper
      .findAll('button')
      .find((b) => b.text().includes('Reset'))

    await resetButton?.trigger('click')

    expect(wrapper.emitted('resetToDefault')).toHaveLength(1)
    expect(wrapper.emitted('resetToDefault')![0]).toEqual([42])
  })

  it('disables reset button when value equals default', () => {
    const widget = createMockWidget(42)
    const node = createMockNode()

    const wrapper = mountWidgetActions(widget, node)

    const resetButton = wrapper
      .findAll('button')
      .find((b) => b.text().includes('Reset'))

    expect(resetButton?.attributes('disabled')).toBeDefined()
  })

  it('does not show reset button when no default value exists', () => {
    mockGetInputSpecForWidget.mockReturnValue({
      type: 'CUSTOM'
    })

    const widget = createMockWidget(100)
    const node = createMockNode()

    const wrapper = mountWidgetActions(widget, node)

    const resetButton = wrapper
      .findAll('button')
      .find((b) => b.text().includes('Reset'))

    expect(resetButton).toBeUndefined()
  })

  it('uses fallback default for INT type without explicit default', async () => {
    mockGetInputSpecForWidget.mockReturnValue({
      type: 'INT'
    })

    const widget = createMockWidget(100)
    const node = createMockNode()

    const wrapper = mountWidgetActions(widget, node)

    const resetButton = wrapper
      .findAll('button')
      .find((b) => b.text().includes('Reset'))

    await resetButton?.trigger('click')

    expect(wrapper.emitted('resetToDefault')![0]).toEqual([0])
  })

  it('uses first option as default for combo without explicit default', async () => {
    mockGetInputSpecForWidget.mockReturnValue({
      type: 'COMBO',
      options: ['option1', 'option2', 'option3']
    })

    const widget = createMockWidget(100)
    const node = createMockNode()

    const wrapper = mountWidgetActions(widget, node)

    const resetButton = wrapper
      .findAll('button')
      .find((b) => b.text().includes('Reset'))

    await resetButton?.trigger('click')

    expect(wrapper.emitted('resetToDefault')![0]).toEqual(['option1'])
  })
})
