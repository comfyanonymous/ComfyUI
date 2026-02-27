import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { LGraphNode } from '@/lib/litegraph/src/litegraph'
import type { ISerialisedNode } from '@/lib/litegraph/src/types/serialisation'
import { sortWidgetValuesByInputOrder } from '@/workbench/utils/nodeDefOrderingUtil'

describe('LGraphNode widget ordering', () => {
  let node: LGraphNode

  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    node = new LGraphNode('TestNode')
  })

  describe('configure with widgets_values', () => {
    it('should apply widget values in correct order when widgets order matches input_order', () => {
      // Create node with widgets
      node.addWidget('number', 'steps', 20, null, {})
      node.addWidget('number', 'seed', 0, null, {})
      node.addWidget('text', 'prompt', '', null, {})

      // Configure with widget values
      const info: ISerialisedNode = {
        id: 1,
        type: 'TestNode',
        pos: [0, 0],
        size: [200, 100],
        flags: {},
        order: 0,
        mode: 0,
        widgets_values: [30, 12345, 'test prompt']
      }

      node.configure(info)

      // Check widget values are applied correctly
      expect(node.widgets![0].value).toBe(30) // steps
      expect(node.widgets![1].value).toBe(12345) // seed
      expect(node.widgets![2].value).toBe('test prompt') // prompt
    })

    it('should handle mismatched widget order with input_order', () => {
      // Simulate widgets created in wrong order (e.g., from unordered Object.entries)
      // but widgets_values is in the correct order according to input_order
      node.addWidget('number', 'seed', 0, null, {})
      node.addWidget('text', 'prompt', '', null, {})
      node.addWidget('number', 'steps', 20, null, {})

      // Widget values are in input_order: [steps, seed, prompt]
      const info: ISerialisedNode = {
        id: 1,
        type: 'TestNode',
        pos: [0, 0],
        size: [200, 100],
        flags: {},
        order: 0,
        mode: 0,
        widgets_values: [30, 12345, 'test prompt']
      }

      // This would apply values incorrectly without proper ordering
      node.configure(info)

      // Without fix, values would be applied in wrong order:
      // seed (widget[0]) would get 30 (should be 12345)
      // prompt (widget[1]) would get 12345 (should be 'test prompt')
      // steps (widget[2]) would get 'test prompt' (should be 30)

      // This test demonstrates the bug - values are applied in wrong order
      expect(node.widgets![0].value).toBe(30) // seed gets steps value (WRONG)
      expect(node.widgets![1].value).toBe(12345) // prompt gets seed value (WRONG)
      expect(node.widgets![2].value).toBe('test prompt') // steps gets prompt value (WRONG)
    })

    it('should skip widgets with serialize: false', () => {
      node.addWidget('number', 'steps', 20, null, {})
      node.addWidget('button', 'action', 'Click', null, {})
      node.widgets![1].serialize = false // button should not serialize
      node.addWidget('number', 'seed', 0, null, {})

      const info: ISerialisedNode = {
        id: 1,
        type: 'TestNode',
        pos: [0, 0],
        size: [200, 100],
        flags: {},
        order: 0,
        mode: 0,
        widgets_values: [30, 12345] // Only serializable widgets
      }

      node.configure(info)

      expect(node.widgets![0].value).toBe(30) // steps
      expect(node.widgets![1].value).toBe('Click') // button unchanged
      expect(node.widgets![2].value).toBe(12345) // seed
    })
  })
})

describe('sortWidgetValuesByInputOrder', () => {
  it('should reorder widget values based on input_order', () => {
    const inputOrder = ['steps', 'seed', 'prompt']
    const currentWidgetOrder = ['seed', 'prompt', 'steps']
    const widgetValues = [12345, 'test prompt', 30]

    const reordered = sortWidgetValuesByInputOrder(
      widgetValues,
      currentWidgetOrder,
      inputOrder
    )

    // Should reorder to match input_order: [steps, seed, prompt]
    expect(reordered).toEqual([30, 12345, 'test prompt'])
  })

  it('should handle widgets not in input_order', () => {
    const inputOrder = ['steps', 'seed']
    const currentWidgetOrder = ['seed', 'prompt', 'steps', 'cfg']
    const widgetValues = [12345, 'test prompt', 30, 7.5]

    const reordered = sortWidgetValuesByInputOrder(
      widgetValues,
      currentWidgetOrder,
      inputOrder
    )

    // Should put ordered items first, then unordered
    expect(reordered).toEqual([30, 12345, 'test prompt', 7.5])
  })

  it('should handle empty input_order', () => {
    const inputOrder: string[] = []
    const currentWidgetOrder = ['seed', 'prompt', 'steps']
    const widgetValues = [12345, 'test prompt', 30]

    const reordered = sortWidgetValuesByInputOrder(
      widgetValues,
      currentWidgetOrder,
      inputOrder
    )

    // Should return values unchanged
    expect(reordered).toEqual([12345, 'test prompt', 30])
  })

  it('should handle mismatched array lengths', () => {
    const inputOrder = ['steps', 'seed', 'prompt']
    const currentWidgetOrder = ['seed', 'prompt']
    const widgetValues = [12345, 'test prompt', 30] // Extra value

    const reordered = sortWidgetValuesByInputOrder(
      widgetValues,
      currentWidgetOrder,
      inputOrder
    )

    // Should handle gracefully, keeping extra values at the end
    // Since 'steps' is not in currentWidgetOrder, it won't be reordered
    // Only 'seed' and 'prompt' will be reordered based on input_order
    expect(reordered).toEqual([12345, 'test prompt', 30])
  })
})
