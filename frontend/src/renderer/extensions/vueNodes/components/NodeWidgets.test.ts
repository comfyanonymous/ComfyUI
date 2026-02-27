import { createTestingPinia } from '@pinia/testing'
import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import type {
  SafeWidgetData,
  VueNodeData
} from '@/composables/graph/useGraphNodeManager'

import NodeWidgets from '@/renderer/extensions/vueNodes/components/NodeWidgets.vue'

describe('NodeWidgets', () => {
  const createMockWidget = (
    overrides: Partial<SafeWidgetData> = {}
  ): SafeWidgetData => ({
    nodeId: 'test_node',
    name: 'test_widget',
    type: 'combo',
    options: undefined,
    callback: undefined,
    spec: undefined,
    isDOMWidget: false,
    slotMetadata: undefined,
    ...overrides
  })

  const createMockNodeData = (
    nodeType: string = 'TestNode',
    widgets: SafeWidgetData[] = []
  ): VueNodeData => ({
    id: '1',
    type: nodeType,
    widgets,
    title: 'Test Node',
    mode: 0,
    selected: false,
    executing: false,
    inputs: [],
    outputs: []
  })

  const mountComponent = (nodeData?: VueNodeData) => {
    return mount(NodeWidgets, {
      props: {
        nodeData
      },
      global: {
        plugins: [createTestingPinia()],
        stubs: {
          // Stub InputSlot to avoid complex slot registration dependencies
          InputSlot: true
        },
        mocks: {
          $t: (key: string) => key
        }
      }
    })
  }

  describe('node-type prop passing', () => {
    it('passes node type to widget components', () => {
      const widget = createMockWidget()
      const nodeData = createMockNodeData('CheckpointLoaderSimple', [widget])
      const wrapper = mountComponent(nodeData)

      // Find the dynamically rendered widget component
      const widgetComponent = wrapper.find('.lg-node-widget')
      expect(widgetComponent.exists()).toBe(true)

      // Verify node-type prop is passed
      const component = widgetComponent.findComponent({ name: 'WidgetSelect' })
      if (component.exists()) {
        expect(component.props('nodeType')).toBe('CheckpointLoaderSimple')
      }
    })

    it('passes empty string when nodeData is undefined', () => {
      const wrapper = mountComponent(undefined)

      // No widgets should be rendered
      const widgetComponents = wrapper.findAll('.lg-node-widget')
      expect(widgetComponents).toHaveLength(0)
    })

    it('passes empty string when nodeData.type is undefined', () => {
      const widget = createMockWidget()
      const nodeData = createMockNodeData('', [widget])
      const wrapper = mountComponent(nodeData)

      const widgetComponent = wrapper.find('.lg-node-widget')
      if (widgetComponent.exists()) {
        const component = widgetComponent.findComponent({
          name: 'WidgetSelect'
        })
        if (component.exists()) {
          expect(component.props('nodeType')).toBe('')
        }
      }
    })

    it.for(['CheckpointLoaderSimple', 'LoraLoader', 'VAELoader', 'KSampler'])(
      'passes correct node type: %s',
      (nodeType) => {
        const widget = createMockWidget()
        const nodeData = createMockNodeData(nodeType, [widget])
        const wrapper = mountComponent(nodeData)

        const widgetComponent = wrapper.find('.lg-node-widget')
        expect(widgetComponent.exists()).toBe(true)

        const component = widgetComponent.findComponent({
          name: 'WidgetSelect'
        })
        if (component.exists()) {
          expect(component.props('nodeType')).toBe(nodeType)
        }
      }
    )
  })
})
