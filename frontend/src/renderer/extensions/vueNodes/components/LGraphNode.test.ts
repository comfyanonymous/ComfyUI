import { createTestingPinia } from '@pinia/testing'
import { mount } from '@vue/test-utils'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { computed, toValue } from 'vue'
import type { ComponentProps } from 'vue-component-type-helpers'
import { createI18n } from 'vue-i18n'

import type { VueNodeData } from '@/composables/graph/useGraphNodeManager'
import LGraphNode from '@/renderer/extensions/vueNodes/components/LGraphNode.vue'
import { useVueElementTracking } from '@/renderer/extensions/vueNodes/composables/useVueNodeResizeTracking'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import { setActivePinia } from 'pinia'

const mockData = vi.hoisted(() => ({
  mockExecuting: false
}))

vi.mock('@/renderer/core/layout/transform/useTransformState', () => {
  return {
    useTransformState: () => ({
      screenToCanvas: vi.fn(),
      canvasToScreen: vi.fn(),
      camera: { z: 1 },
      isNodeInViewport: vi.fn()
    })
  }
})

vi.mock(
  '@/renderer/extensions/vueNodes/composables/useNodeEventHandlers',
  () => {
    const handleNodeSelect = vi.fn()
    return { useNodeEventHandlers: () => ({ handleNodeSelect }) }
  }
)

vi.mock(
  '@/renderer/extensions/vueNodes/composables/useVueNodeResizeTracking',
  () => ({
    useVueElementTracking: vi.fn()
  })
)

vi.mock('@/composables/useErrorHandling', () => ({
  useErrorHandling: () => ({
    toastErrorHandler: vi.fn()
  })
}))

vi.mock('@/renderer/extensions/vueNodes/layout/useNodeLayout', () => ({
  useNodeLayout: () => ({
    position: { x: 100, y: 50 },
    size: computed(() => ({ width: 200, height: 100 })),
    zIndex: 0,
    startDrag: vi.fn(),
    handleDrag: vi.fn(),
    endDrag: vi.fn(),
    moveTo: vi.fn()
  })
}))

vi.mock(
  '@/renderer/extensions/vueNodes/execution/useNodeExecutionState',
  () => ({
    useNodeExecutionState: vi.fn(() => ({
      executing: computed(() => mockData.mockExecuting),
      progress: computed(() => undefined),
      progressPercentage: computed(() => undefined),
      progressState: computed(() => undefined),
      executionState: computed(() => 'idle' as const)
    }))
  })
)

vi.mock('@/renderer/extensions/vueNodes/preview/useNodePreviewState', () => ({
  useNodePreviewState: vi.fn(() => ({
    latestPreviewUrl: computed(() => ''),
    shouldShowPreviewImg: computed(() => false)
  }))
}))

vi.mock(
  '@/renderer/extensions/vueNodes/interactions/resize/useNodeResize',
  () => ({
    useNodeResize: vi.fn(() => ({
      startResize: vi.fn(),
      isResizing: computed(() => false)
    }))
  })
)

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {
      'Node Render Error': 'Node Render Error'
    }
  }
})

const pinia = createTestingPinia({
  createSpy: vi.fn
})

function mountLGraphNode(props: ComponentProps<typeof LGraphNode>) {
  return mount(LGraphNode, {
    props,
    global: {
      plugins: [pinia, i18n],
      stubs: {
        NodeHeader: true,
        NodeSlots: true,
        NodeWidgets: true,
        NodeContent: true,
        SlotConnectionDot: true
      }
    }
  })
}
const mockNodeData: VueNodeData = {
  id: 'test-node-123',
  title: 'Test Node',
  type: 'TestNode',
  mode: 0,
  flags: {},
  inputs: [],
  outputs: [],
  widgets: [],
  selected: false,
  executing: false
}

describe('LGraphNode', () => {
  beforeEach(() => {
    vi.resetAllMocks()
    mockData.mockExecuting = false

    setActivePinia(pinia)
    const canvasStore = useCanvasStore()
    canvasStore.selectedNodeIds.clear()
  })

  it('should call resize tracking composable with node ID', () => {
    mountLGraphNode({ nodeData: mockNodeData })

    expect(useVueElementTracking).toHaveBeenCalledWith(
      expect.any(Function),
      'node'
    )
    const idArg = vi.mocked(useVueElementTracking).mock.calls[0]?.[0]
    const id = toValue(idArg)
    expect(id).toEqual('test-node-123')
  })

  it('should render with data-node-id attribute', () => {
    const wrapper = mountLGraphNode({ nodeData: mockNodeData })

    expect(wrapper.attributes('data-node-id')).toBe('test-node-123')
  })

  it('should render node title', () => {
    // Don't stub NodeHeader for this test so we can see the title
    const wrapper = mount(LGraphNode, {
      props: { nodeData: mockNodeData },
      global: {
        plugins: [pinia, i18n],
        stubs: {
          NodeSlots: true,
          NodeWidgets: true,
          NodeContent: true,
          SlotConnectionDot: true
        }
      }
    })

    expect(wrapper.text()).toContain('Test Node')
  })

  it('should apply selected styling when selected prop is true', async () => {
    const canvasStore = useCanvasStore()
    canvasStore.selectedNodeIds.clear()
    canvasStore.selectedNodeIds.add('test-node-123')

    const wrapper = mountLGraphNode({ nodeData: mockNodeData })

    expect(wrapper.classes()).toContain('outline-3')
    expect(wrapper.classes()).toContain('outline-node-component-outline')
  })

  it('should render progress indicator when executing prop is true', () => {
    mockData.mockExecuting = true

    const wrapper = mountLGraphNode({ nodeData: mockNodeData })

    expect(wrapper.classes()).toContain('outline-node-stroke-executing')
  })

  it('should initialize height CSS vars for collapsed nodes', () => {
    const wrapper = mountLGraphNode({
      nodeData: {
        ...mockNodeData,
        flags: { collapsed: true }
      }
    })

    expect(wrapper.element.style.getPropertyValue('--node-height')).toBe('')
    expect(wrapper.element.style.getPropertyValue('--node-height-x')).toBe(
      '130px'
    )
  })

  it('should initialize height CSS vars for expanded nodes', () => {
    const wrapper = mountLGraphNode({
      nodeData: {
        ...mockNodeData,
        flags: { collapsed: false }
      }
    })

    expect(wrapper.element.style.getPropertyValue('--node-height')).toBe(
      '130px'
    )
    expect(wrapper.element.style.getPropertyValue('--node-height-x')).toBe('')
  })
})
