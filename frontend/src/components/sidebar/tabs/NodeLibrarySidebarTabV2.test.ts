import { mount } from '@vue/test-utils'
import { createTestingPinia } from '@pinia/testing'
import { TabsContent, TabsList, TabsRoot, TabsTrigger } from 'reka-ui'
import { ref } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'

import NodeLibrarySidebarTabV2 from './NodeLibrarySidebarTabV2.vue'

vi.mock('@vueuse/core', async () => {
  const actual = await vi.importActual('@vueuse/core')
  return {
    ...actual,
    useLocalStorage: vi.fn((_key: string, defaultValue: unknown) =>
      ref(defaultValue)
    )
  }
})

vi.mock('@/composables/node/useNodeDragToCanvas', () => ({
  useNodeDragToCanvas: () => ({
    isDragging: { value: false },
    draggedNode: { value: null },
    cursorPosition: { value: { x: 0, y: 0 } },
    startDrag: vi.fn(),
    cancelDrag: vi.fn(),
    setupGlobalListeners: vi.fn(),
    cleanupGlobalListeners: vi.fn()
  })
}))

vi.mock('@/services/nodeOrganizationService', () => ({
  DEFAULT_TAB_ID: 'essentials',
  DEFAULT_SORTING_ID: 'alphabetical',
  nodeOrganizationService: {
    organizeNodesByTab: vi.fn(() => []),
    getSortingStrategies: vi.fn(() => [])
  }
}))

vi.mock('./nodeLibrary/AllNodesPanel.vue', () => ({
  default: {
    name: 'AllNodesPanel',
    template: '<div data-testid="all-panel"><slot /></div>',
    props: ['sections', 'expandedKeys', 'fillNodeInfo']
  }
}))

vi.mock('./nodeLibrary/CustomNodesPanel.vue', () => ({
  default: {
    name: 'CustomNodesPanel',
    template: '<div data-testid="custom-panel"><slot /></div>',
    props: ['sections', 'expandedKeys']
  }
}))

vi.mock('./nodeLibrary/EssentialNodesPanel.vue', () => ({
  default: {
    name: 'EssentialNodesPanel',
    template: '<div data-testid="essential-panel"><slot /></div>',
    props: ['root', 'expandedKeys']
  }
}))

vi.mock('./nodeLibrary/NodeDragPreview.vue', () => ({
  default: {
    name: 'NodeDragPreview',
    template: '<div />'
  }
}))

vi.mock('@/components/common/SearchBoxV2.vue', () => ({
  default: {
    name: 'SearchBox',
    template: '<input data-testid="search-box" />',
    props: ['modelValue', 'placeholder'],
    setup() {
      return { focus: vi.fn() }
    },
    expose: ['focus']
  }
}))

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: { en: {} }
})

describe('NodeLibrarySidebarTabV2', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  function mountComponent() {
    return mount(NodeLibrarySidebarTabV2, {
      global: {
        plugins: [createTestingPinia({ stubActions: false }), i18n],
        components: {
          TabsRoot,
          TabsList,
          TabsTrigger,
          TabsContent
        },
        stubs: {
          teleport: true
        }
      }
    })
  }

  it('should render with tabs', () => {
    const wrapper = mountComponent()

    const triggers = wrapper.findAllComponents(TabsTrigger)
    expect(triggers).toHaveLength(3)
  })

  it('should render search box', () => {
    const wrapper = mountComponent()

    expect(wrapper.find('[data-testid="search-box"]').exists()).toBe(true)
  })

  it('should render only the selected panel', () => {
    const wrapper = mountComponent()

    expect(wrapper.find('[data-testid="essential-panel"]').exists()).toBe(true)
    expect(wrapper.find('[data-testid="all-panel"]').exists()).toBe(false)
    expect(wrapper.find('[data-testid="custom-panel"]').exists()).toBe(false)
  })
})
