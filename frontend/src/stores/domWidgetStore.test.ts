import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import type { LGraphNode } from '@/lib/litegraph/src/litegraph'
import { useDomWidgetStore } from '@/stores/domWidgetStore'
import { createTestingPinia } from '@pinia/testing'

// Mock DOM widget for testing
const createMockDOMWidget = (id: string) => {
  const element = document.createElement('input')
  return {
    id,
    element,
    node: {
      id: 'node-1',
      title: 'Test Node',
      pos: [0, 0],
      size: [200, 100]
    } as Partial<LGraphNode> as LGraphNode,
    name: 'test_widget',
    type: 'text',
    value: 'test',
    options: {},
    y: 0,
    margin: 10,
    isVisible: () => true,
    containerNode: undefined
  }
}

describe('domWidgetStore', () => {
  let store: ReturnType<typeof useDomWidgetStore>

  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    store = useDomWidgetStore()
  })

  describe('widget registration', () => {
    it('should register a widget with default state', () => {
      const widget = createMockDOMWidget('widget-1')

      store.registerWidget(widget)

      expect(store.widgetStates.has('widget-1')).toBe(true)
      const state = store.widgetStates.get('widget-1')
      expect(state).toBeDefined()
      expect(state!.widget).toBe(widget)
      expect(state!.visible).toBe(true)
      expect(state!.active).toBe(true)
      expect(state!.readonly).toBe(false)
      expect(state!.zIndex).toBe(0)
      expect(state!.pos).toEqual([0, 0])
      expect(state!.size).toEqual([0, 0])
    })

    it('should not register the same widget twice', () => {
      const widget = createMockDOMWidget('widget-1')

      store.registerWidget(widget)
      store.registerWidget(widget)

      // Should still only have one entry
      const states = Array.from(store.widgetStates.values())
      expect(states.length).toBe(1)
    })
  })

  describe('widget unregistration', () => {
    it('should unregister a widget by id', () => {
      const widget = createMockDOMWidget('widget-1')

      store.registerWidget(widget)
      expect(store.widgetStates.has('widget-1')).toBe(true)

      store.unregisterWidget('widget-1')
      expect(store.widgetStates.has('widget-1')).toBe(false)
    })

    it('should handle unregistering non-existent widget gracefully', () => {
      // Should not throw
      expect(() => {
        store.unregisterWidget('non-existent')
      }).not.toThrow()
    })
  })

  describe('widget state management', () => {
    it('should activate a widget', () => {
      const widget = createMockDOMWidget('widget-1')
      store.registerWidget(widget)

      // Set to inactive first
      const state = store.widgetStates.get('widget-1')!
      state.active = false

      store.activateWidget('widget-1')
      expect(state.active).toBe(true)
    })

    it('should deactivate a widget', () => {
      const widget = createMockDOMWidget('widget-1')
      store.registerWidget(widget)

      store.deactivateWidget('widget-1')
      const state = store.widgetStates.get('widget-1')
      expect(state!.active).toBe(false)
    })

    it('should handle activating non-existent widget gracefully', () => {
      expect(() => {
        store.activateWidget('non-existent')
      }).not.toThrow()
    })
  })

  describe('computed states', () => {
    it('should separate active and inactive widget states', () => {
      const widget1 = createMockDOMWidget('widget-1')
      const widget2 = createMockDOMWidget('widget-2')

      store.registerWidget(widget1)
      store.registerWidget(widget2)

      // Deactivate widget2
      store.deactivateWidget('widget-2')

      expect(store.activeWidgetStates.length).toBe(1)
      expect(store.activeWidgetStates[0].widget.id).toBe('widget-1')

      expect(store.inactiveWidgetStates.length).toBe(1)
      expect(store.inactiveWidgetStates[0].widget.id).toBe('widget-2')
    })
  })

  describe('clear functionality', () => {
    it('should clear all widget states', () => {
      const widget1 = createMockDOMWidget('widget-1')
      const widget2 = createMockDOMWidget('widget-2')

      store.registerWidget(widget1)
      store.registerWidget(widget2)

      expect(store.widgetStates.size).toBe(2)

      store.clear()

      expect(store.widgetStates.size).toBe(0)
      expect(store.activeWidgetStates.length).toBe(0)
      expect(store.inactiveWidgetStates.length).toBe(0)
    })
  })
})
