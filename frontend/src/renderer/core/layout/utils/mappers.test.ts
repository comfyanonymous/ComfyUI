import { describe, expect, it } from 'vitest'
import * as Y from 'yjs'

import {
  NODE_LAYOUT_DEFAULTS,
  yNodeToLayout
} from '@/renderer/core/layout/utils/mappers'
import type { NodeLayoutMap } from '@/renderer/core/layout/utils/mappers'

describe('mappers', () => {
  it('yNodeToLayout reads from Yjs-attached map', () => {
    const layout = {
      id: 'node-1',
      position: { x: 12, y: 34 },
      size: { width: 111, height: 222 },
      zIndex: 5,
      visible: true,
      bounds: { x: 12, y: 34, width: 111, height: 222 }
    }

    const doc = new Y.Doc()
    const ynode = doc.getMap('node') as NodeLayoutMap
    ynode.set('id', layout.id)
    ynode.set('position', layout.position)
    ynode.set('size', layout.size)
    ynode.set('zIndex', layout.zIndex)
    ynode.set('visible', layout.visible)
    ynode.set('bounds', layout.bounds)

    const back = yNodeToLayout(ynode)
    expect(back).toEqual(layout)
  })

  it('yNodeToLayout applies defaults for missing fields', () => {
    const doc = new Y.Doc()
    const ynode = doc.getMap('node') as NodeLayoutMap
    // Don't set any fields - they should all use defaults

    const back = yNodeToLayout(ynode)
    expect(back.id).toBe(NODE_LAYOUT_DEFAULTS.id)
    expect(back.position).toEqual(NODE_LAYOUT_DEFAULTS.position)
    expect(back.size).toEqual(NODE_LAYOUT_DEFAULTS.size)
    expect(back.zIndex).toEqual(NODE_LAYOUT_DEFAULTS.zIndex)
    expect(back.visible).toEqual(NODE_LAYOUT_DEFAULTS.visible)
    expect(back.bounds).toEqual(NODE_LAYOUT_DEFAULTS.bounds)
  })
})
