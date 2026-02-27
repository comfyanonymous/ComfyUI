import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { LGraph, LGraphNode } from '@/lib/litegraph/src/litegraph'
import type { INumericWidget } from '@/lib/litegraph/src/types/widgets'
import { NumberWidget } from '@/lib/litegraph/src/widgets/NumberWidget'
import { useWidgetValueStore } from '@/stores/widgetValueStore'

function createTestWidget(
  node: LGraphNode,
  overrides: Partial<INumericWidget> = {}
): NumberWidget {
  return new NumberWidget(
    {
      type: 'number',
      name: 'testWidget',
      value: 42,
      options: { min: 0, max: 100 },
      y: 0,
      ...overrides
    },
    node
  )
}

describe('BaseWidget store integration', () => {
  let graph: LGraph
  let node: LGraphNode
  let store: ReturnType<typeof useWidgetValueStore>

  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    store = useWidgetValueStore()
    graph = new LGraph()
    node = new LGraphNode('TestNode')
    node.id = 1
    graph.add(node)
  })

  describe('metadata properties before registration', () => {
    it('uses internal values when not registered', () => {
      const widget = createTestWidget(node, {
        label: 'My Label',
        hidden: true,
        disabled: true,
        advanced: true
      })

      expect(widget.label).toBe('My Label')
      expect(widget.hidden).toBe(true)
      expect(widget.disabled).toBe(true)
      expect(widget.advanced).toBe(true)
    })

    it('allows setting properties without store', () => {
      const widget = createTestWidget(node)

      widget.label = 'New Label'
      widget.hidden = true
      widget.disabled = true
      widget.advanced = true

      expect(widget.label).toBe('New Label')
      expect(widget.hidden).toBe(true)
      expect(widget.disabled).toBe(true)
      expect(widget.advanced).toBe(true)
    })
  })

  describe('metadata properties after registration', () => {
    it('reads from store when registered', () => {
      const widget = createTestWidget(node, {
        name: 'storeWidget',
        label: 'Store Label',
        hidden: true,
        disabled: true,
        advanced: true
      })
      widget.setNodeId(1)

      expect(widget.label).toBe('Store Label')
      expect(widget.hidden).toBe(true)
      expect(widget.disabled).toBe(true)
      expect(widget.advanced).toBe(true)
    })

    it('writes to store when registered', () => {
      const widget = createTestWidget(node, { name: 'writeWidget' })
      widget.setNodeId(1)

      widget.label = 'Updated Label'
      widget.hidden = true
      widget.disabled = true
      widget.advanced = true

      const state = store.getWidget(graph.id, 1, 'writeWidget')
      expect(state?.label).toBe('Updated Label')
      expect(state?.disabled).toBe(true)

      expect(widget.hidden).toBe(true)
      expect(widget.advanced).toBe(true)
    })

    it('syncs value with store', () => {
      const widget = createTestWidget(node, { name: 'valueWidget', value: 42 })
      widget.setNodeId(1)

      widget.value = 99
      expect(store.getWidget(graph.id, 1, 'valueWidget')?.value).toBe(99)

      const state = store.getWidget(graph.id, 1, 'valueWidget')!
      state.value = 55
      expect(widget.value).toBe(55)
    })
  })

  describe('automatic registration via setNodeId', () => {
    it('registers widget with all metadata', () => {
      const widget = createTestWidget(node, {
        name: 'autoRegWidget',
        value: 100,
        label: 'Auto Label',
        hidden: true,
        disabled: true,
        advanced: true
      })
      widget.setNodeId(1)

      const state = store.getWidget(graph.id, 1, 'autoRegWidget')
      expect(state).toBeDefined()
      expect(state?.nodeId).toBe(1)
      expect(state?.name).toBe('autoRegWidget')
      expect(state?.type).toBe('number')
      expect(state?.value).toBe(100)
      expect(state?.label).toBe('Auto Label')
      expect(state?.disabled).toBe(true)
      expect(state?.options).toEqual({ min: 0, max: 100 })

      expect(widget.hidden).toBe(true)
      expect(widget.advanced).toBe(true)
    })

    it('registers widget with default metadata values', () => {
      const widget = createTestWidget(node, { name: 'defaultsWidget' })
      widget.setNodeId(1)

      const state = store.getWidget(graph.id, 1, 'defaultsWidget')
      expect(state).toBeDefined()
      expect(state?.disabled).toBe(false)
      expect(state?.label).toBeUndefined()

      expect(widget.hidden).toBeUndefined()
      expect(widget.advanced).toBeUndefined()
    })

    it('registers widget value accessible via getWidget', () => {
      const widget = createTestWidget(node, { name: 'valuesWidget', value: 77 })
      widget.setNodeId(1)

      expect(store.getWidget(graph.id, 1, 'valuesWidget')?.value).toBe(77)
    })
  })

  describe('DOM widget value registration', () => {
    it('registers value from getter when value property is overridden', () => {
      const defaultValue = 'You are an expert image-generation engine.'
      const widget = createTestWidget(node, {
        name: 'system_prompt',
        value: undefined as unknown as number
      })

      // Simulate what addDOMWidget does: override value with getter/setter
      // that falls back to a default (like inputEl.value for textarea widgets)
      Object.defineProperty(widget, 'value', {
        get() {
          const graphId = widget.node.graph?.rootGraph.id
          if (!graphId) return defaultValue
          const state = store.getWidget(graphId, node.id, 'system_prompt')
          return (state?.value as string) ?? defaultValue
        },
        set(v: string) {
          const graphId = widget.node.graph?.rootGraph.id
          if (!graphId) return
          const state = store.getWidget(graphId, node.id, 'system_prompt')
          if (state) state.value = v
        }
      })

      widget.setNodeId(node.id)

      const state = store.getWidget(graph.id, node.id, 'system_prompt')
      expect(state?.value).toBe(defaultValue)
    })
  })

  describe('fallback behavior', () => {
    it('uses internal value before registration', () => {
      const widget = createTestWidget(node, {
        name: 'fallbackWidget',
        label: 'Internal'
      })
      // Widget not yet registered - should use internal value
      expect(widget.label).toBe('Internal')
    })

    it('handles undefined values correctly', () => {
      const widget = createTestWidget(node)
      widget.setNodeId(1)

      widget.disabled = undefined

      const state = store.getWidget(graph.id, 1, 'testWidget')
      expect(state?.disabled).toBe(false)
    })
  })
})
