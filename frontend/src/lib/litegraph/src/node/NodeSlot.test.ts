import { describe, expect, it } from 'vitest'

import type {
  INodeInputSlot,
  INodeOutputSlot
} from '@/lib/litegraph/src/litegraph'
import {
  inputAsSerialisable,
  outputAsSerialisable
} from '@/lib/litegraph/src/litegraph'
import type { ReadOnlyRect } from '@/lib/litegraph/src/interfaces'

const boundingRect: ReadOnlyRect = [0, 0, 10, 10]

describe('NodeSlot', () => {
  describe('inputAsSerialisable', () => {
    it('removes _data from serialized slot', () => {
      const slot: INodeOutputSlot = {
        _data: 'test data',
        name: 'test-id',
        type: 'STRING',
        links: [],
        boundingRect
      }
      // @ts-expect-error Argument type mismatch for test
      const serialized = outputAsSerialisable(slot)
      expect(serialized).not.toHaveProperty('_data')
    })

    it('removes pos from widget input slots', () => {
      // Minimal slot for serialization test - boundingRect is calculated at runtime, not serialized
      const widgetInputSlot: INodeInputSlot = {
        name: 'test-id',
        pos: [10, 20],
        type: 'STRING',
        link: null,
        widget: { name: 'test-widget', type: 'combo' },
        boundingRect
      }

      const serialized = inputAsSerialisable(widgetInputSlot)
      expect(serialized).not.toHaveProperty('pos')
    })

    it('preserves pos for non-widget input slots', () => {
      const normalSlot: INodeInputSlot = {
        name: 'test-id',
        type: 'STRING',
        pos: [10, 20],
        link: null,
        boundingRect
      }
      const serialized = inputAsSerialisable(normalSlot)
      expect(serialized).toHaveProperty('pos')
    })

    it('preserves only widget name during serialization', () => {
      // Extra widget properties simulate real data that should be stripped during serialization
      const widgetInputSlot: INodeInputSlot = {
        name: 'test-id',
        type: 'STRING',
        link: null,
        boundingRect,
        widget: {
          name: 'test-widget',
          type: 'combo'
        }
      }

      const serialized = inputAsSerialisable(widgetInputSlot)
      expect(serialized.widget).toEqual({ name: 'test-widget' })
      expect(serialized.widget).not.toHaveProperty('type')
      expect(serialized.widget).not.toHaveProperty('value')
      expect(serialized.widget).not.toHaveProperty('options')
    })
  })
})
