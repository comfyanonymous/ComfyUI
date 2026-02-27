import { createTestingPinia } from '@pinia/testing'
import { mount } from '@vue/test-utils'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'

import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'
import WidgetItem from './WidgetItem.vue'

const { mockGetInputSpecForWidget, StubWidgetComponent } = vi.hoisted(() => ({
  mockGetInputSpecForWidget: vi.fn(),
  StubWidgetComponent: {
    name: 'StubWidget',
    props: ['widget', 'modelValue', 'nodeId', 'nodeType'],
    template: '<div class="stub-widget" />'
  }
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

vi.mock('@/composables/graph/useGraphNodeManager', () => ({
  getControlWidget: vi.fn(() => undefined)
}))

vi.mock('@/core/graph/subgraph/resolvePromotedWidgetSource', () => ({
  resolvePromotedWidgetSource: vi.fn(() => undefined)
}))

vi.mock(
  '@/renderer/extensions/vueNodes/widgets/registry/widgetRegistry',
  () => ({
    getComponent: () => StubWidgetComponent,
    shouldExpand: () => false
  })
)

vi.mock(
  '@/renderer/extensions/vueNodes/widgets/components/WidgetLegacy.vue',
  () => ({
    default: StubWidgetComponent
  })
)

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {
      rightSidePanel: {
        fallbackNodeTitle: 'Untitled'
      }
    }
  }
})

function createMockNode(overrides: Partial<LGraphNode> = {}): LGraphNode {
  return {
    id: 1,
    type: 'TestNode',
    isSubgraphNode: () => false,
    graph: { rootGraph: { id: 'test-graph-id' } },
    ...overrides
  } as unknown as LGraphNode
}

function createMockWidget(overrides: Partial<IBaseWidget> = {}): IBaseWidget {
  return {
    name: 'test_widget',
    type: 'combo',
    value: 'option_a',
    y: 0,
    options: {
      values: ['option_a', 'option_b', 'option_c']
    },
    ...overrides
  } as IBaseWidget
}

/**
 * Creates a mock PromotedWidgetView that mirrors the real class:
 * properties like name, type, value, options are prototype getters,
 * NOT own properties — so object spread loses them.
 */
function createMockPromotedWidgetView(
  sourceOptions: IBaseWidget['options'] = {
    values: ['model_a.safetensors', 'model_b.safetensors']
  }
): IBaseWidget {
  class MockPromotedWidgetView {
    readonly sourceNodeId = '42'
    readonly sourceWidgetName = 'ckpt_name'
    readonly serialize = false

    get name(): string {
      return 'ckpt_name'
    }
    get type(): string {
      return 'combo'
    }
    get value(): unknown {
      return 'model_a.safetensors'
    }
    get options(): IBaseWidget['options'] {
      return sourceOptions
    }
    get label(): string | undefined {
      return undefined
    }
    get y(): number {
      return 0
    }
  }
  return new MockPromotedWidgetView() as unknown as IBaseWidget
}

function mountWidgetItem(
  widget: IBaseWidget,
  node: LGraphNode = createMockNode()
) {
  return mount(WidgetItem, {
    props: { widget, node },
    global: {
      plugins: [i18n],
      stubs: {
        EditableText: { template: '<span />' },
        WidgetActions: { template: '<span />' }
      }
    }
  })
}

describe('WidgetItem', () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    vi.clearAllMocks()
  })

  describe('promoted widget options', () => {
    it('passes options from a regular widget to the widget component', () => {
      const widget = createMockWidget({
        options: { values: ['a', 'b', 'c'] }
      })
      const wrapper = mountWidgetItem(widget)
      const stub = wrapper.findComponent(StubWidgetComponent)

      expect(stub.props('widget').options).toEqual({
        values: ['a', 'b', 'c']
      })
    })

    it('passes options from a PromotedWidgetView to the widget component', () => {
      const expectedOptions = {
        values: ['model_a.safetensors', 'model_b.safetensors']
      }
      const widget = createMockPromotedWidgetView(expectedOptions)
      const wrapper = mountWidgetItem(widget)
      const stub = wrapper.findComponent(StubWidgetComponent)

      expect(stub.props('widget').options).toEqual(expectedOptions)
    })

    it('passes type from a PromotedWidgetView to the widget component', () => {
      const widget = createMockPromotedWidgetView()
      const wrapper = mountWidgetItem(widget)
      const stub = wrapper.findComponent(StubWidgetComponent)

      expect(stub.props('widget').type).toBe('combo')
    })

    it('passes name from a PromotedWidgetView to the widget component', () => {
      const widget = createMockPromotedWidgetView()
      const wrapper = mountWidgetItem(widget)
      const stub = wrapper.findComponent(StubWidgetComponent)

      expect(stub.props('widget').name).toBe('ckpt_name')
    })

    it('passes value from a PromotedWidgetView to the widget component', () => {
      const widget = createMockPromotedWidgetView()
      const wrapper = mountWidgetItem(widget)
      const stub = wrapper.findComponent(StubWidgetComponent)

      expect(stub.props('widget').value).toBe('model_a.safetensors')
    })
  })
})
