import * as Y from 'yjs'

import type { NodeLayout } from '@/renderer/core/layout/types'

export type NodeLayoutMap = Y.Map<NodeLayout[keyof NodeLayout]>

export const NODE_LAYOUT_DEFAULTS: NodeLayout = {
  id: 'unknown-node',
  position: { x: 0, y: 0 },
  size: { width: 100, height: 50 },
  zIndex: 0,
  visible: true,
  bounds: { x: 0, y: 0, width: 100, height: 50 }
}

export function layoutToYNode(layout: NodeLayout): NodeLayoutMap {
  const ynode = new Y.Map<NodeLayout[keyof NodeLayout]>() as NodeLayoutMap
  ynode.set('id', layout.id)
  ynode.set('position', layout.position)
  ynode.set('size', layout.size)
  ynode.set('zIndex', layout.zIndex)
  ynode.set('visible', layout.visible)
  ynode.set('bounds', layout.bounds)
  return ynode
}

function getOr<K extends keyof NodeLayout>(
  map: NodeLayoutMap,
  key: K,
  fallback: NodeLayout[K]
): NodeLayout[K] {
  const v = map.get(key)
  return (v ?? fallback) as NodeLayout[K]
}

export function yNodeToLayout(ynode: NodeLayoutMap): NodeLayout {
  return {
    id: getOr(ynode, 'id', NODE_LAYOUT_DEFAULTS.id),
    position: getOr(ynode, 'position', NODE_LAYOUT_DEFAULTS.position),
    size: getOr(ynode, 'size', NODE_LAYOUT_DEFAULTS.size),
    zIndex: getOr(ynode, 'zIndex', NODE_LAYOUT_DEFAULTS.zIndex),
    visible: getOr(ynode, 'visible', NODE_LAYOUT_DEFAULTS.visible),
    bounds: getOr(ynode, 'bounds', NODE_LAYOUT_DEFAULTS.bounds)
  }
}
