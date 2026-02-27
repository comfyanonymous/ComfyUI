import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import type { UUID } from '@/lib/litegraph/src/utils/uuid'
import type { WidgetState } from './widgetValueStore'
import { useWidgetValueStore } from './widgetValueStore'

function widget<T>(
  nodeId: string,
  name: string,
  type: string,
  value: T,
  extra: Partial<
    Omit<WidgetState<T>, 'nodeId' | 'name' | 'type' | 'value'>
  > = {}
): WidgetState<T> {
  return { nodeId, name, type, value, options: {}, ...extra }
}

describe('useWidgetValueStore', () => {
  const graphA = 'graph-a' as UUID
  const graphB = 'graph-b' as UUID
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
  })

  describe('widgetState.value access', () => {
    it('getWidget returns undefined for unregistered widget', () => {
      const store = useWidgetValueStore()
      expect(store.getWidget(graphA, 'missing', 'widget')).toBeUndefined()
    })

    it('widgetState.value can be read and written directly', () => {
      const store = useWidgetValueStore()
      const state = store.registerWidget(
        graphA,
        widget('node-1', 'seed', 'number', 100)
      )
      expect(state.value).toBe(100)

      state.value = 200
      expect(store.getWidget(graphA, 'node-1', 'seed')?.value).toBe(200)
    })

    it('stores different value types', () => {
      const store = useWidgetValueStore()
      store.registerWidget(graphA, widget('node-1', 'text', 'string', 'hello'))
      store.registerWidget(graphA, widget('node-1', 'number', 'number', 42))
      store.registerWidget(graphA, widget('node-1', 'boolean', 'toggle', true))
      store.registerWidget(
        graphA,
        widget('node-1', 'array', 'combo', [1, 2, 3])
      )

      expect(store.getWidget(graphA, 'node-1', 'text')?.value).toBe('hello')
      expect(store.getWidget(graphA, 'node-1', 'number')?.value).toBe(42)
      expect(store.getWidget(graphA, 'node-1', 'boolean')?.value).toBe(true)
      expect(store.getWidget(graphA, 'node-1', 'array')?.value).toEqual([
        1, 2, 3
      ])
    })
  })

  describe('widget registration', () => {
    it('registers a widget with minimal properties', () => {
      const store = useWidgetValueStore()
      const state = store.registerWidget(
        graphA,
        widget('node-1', 'seed', 'number', 12345)
      )

      expect(state.nodeId).toBe('node-1')
      expect(state.name).toBe('seed')
      expect(state.type).toBe('number')
      expect(state.value).toBe(12345)
      expect(state.disabled).toBeUndefined()
      expect(state.serialize).toBeUndefined()
      expect(state.options).toEqual({})
    })

    it('registers a widget with all properties', () => {
      const store = useWidgetValueStore()
      const state = store.registerWidget(
        graphA,
        widget('node-1', 'prompt', 'string', 'test', {
          label: 'Prompt Text',
          disabled: true,
          serialize: false,
          options: { multiline: true }
        })
      )

      expect(state.label).toBe('Prompt Text')
      expect(state.disabled).toBe(true)
      expect(state.serialize).toBe(false)
      expect(state.options).toEqual({ multiline: true })
    })
  })

  describe('widget getters', () => {
    it('getWidget returns widget state', () => {
      const store = useWidgetValueStore()
      store.registerWidget(graphA, widget('node-1', 'seed', 'number', 100))

      const state = store.getWidget(graphA, 'node-1', 'seed')
      expect(state).toBeDefined()
      expect(state?.name).toBe('seed')
      expect(state?.value).toBe(100)
    })

    it('getWidget returns undefined for missing widget', () => {
      const store = useWidgetValueStore()
      expect(store.getWidget(graphA, 'missing', 'widget')).toBeUndefined()
    })

    it('getNodeWidgets returns all widgets for a node', () => {
      const store = useWidgetValueStore()
      store.registerWidget(graphA, widget('node-1', 'seed', 'number', 1))
      store.registerWidget(graphA, widget('node-1', 'steps', 'number', 20))
      store.registerWidget(graphA, widget('node-2', 'cfg', 'number', 7))

      const widgets = store.getNodeWidgets(graphA, 'node-1')
      expect(widgets).toHaveLength(2)
      expect(widgets.map((w) => w.name).sort()).toEqual(['seed', 'steps'])
    })
  })

  describe('direct property mutation', () => {
    it('disabled can be set directly via getWidget', () => {
      const store = useWidgetValueStore()
      const state = store.registerWidget(
        graphA,
        widget('node-1', 'seed', 'number', 100)
      )

      state.disabled = true
      expect(store.getWidget(graphA, 'node-1', 'seed')?.disabled).toBe(true)
    })

    it('label can be set directly via getWidget', () => {
      const store = useWidgetValueStore()
      const state = store.registerWidget(
        graphA,
        widget('node-1', 'seed', 'number', 100)
      )

      state.label = 'Random Seed'
      expect(store.getWidget(graphA, 'node-1', 'seed')?.label).toBe(
        'Random Seed'
      )

      state.label = undefined
      expect(store.getWidget(graphA, 'node-1', 'seed')?.label).toBeUndefined()
    })
  })

  describe('graph isolation', () => {
    it('isolates widget states by graph', () => {
      const store = useWidgetValueStore()
      store.registerWidget(graphA, widget('node-1', 'seed', 'number', 1))
      store.registerWidget(graphB, widget('node-1', 'seed', 'number', 2))

      expect(store.getWidget(graphA, 'node-1', 'seed')?.value).toBe(1)
      expect(store.getWidget(graphB, 'node-1', 'seed')?.value).toBe(2)
    })

    it('clearGraph only removes one graph namespace', () => {
      const store = useWidgetValueStore()
      store.registerWidget(graphA, widget('node-1', 'seed', 'number', 1))
      store.registerWidget(graphB, widget('node-1', 'seed', 'number', 2))

      store.clearGraph(graphA)

      expect(store.getWidget(graphA, 'node-1', 'seed')).toBeUndefined()
      expect(store.getWidget(graphB, 'node-1', 'seed')?.value).toBe(2)
    })
  })
})
