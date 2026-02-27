import { describe, expect, it } from 'vitest'

import { LGraph, LGraphNode } from '@/lib/litegraph/src/litegraph'

import { getDomWidgetZIndex } from './domWidgetZIndex'

describe('getDomWidgetZIndex', () => {
  it('follows graph node ordering when node.order is stale', () => {
    const graph = new LGraph()
    const first = new LGraphNode('first')
    const second = new LGraphNode('second')
    graph.add(first)
    graph.add(second)

    first.order = 0
    second.order = 1

    const nodes = (graph as unknown as { _nodes: LGraphNode[] })._nodes
    nodes.splice(nodes.indexOf(first), 1)
    nodes.push(first)

    expect(first.order).toBe(0)
    expect(second.order).toBe(1)

    expect(getDomWidgetZIndex(first, graph)).toBe(1)
    expect(getDomWidgetZIndex(second, graph)).toBe(0)
  })

  it('falls back to node.order when node is not in current graph', () => {
    const graph = new LGraph()
    const node = new LGraphNode('orphan')
    node.order = 7

    expect(getDomWidgetZIndex(node, graph)).toBe(7)
    expect(getDomWidgetZIndex(node, undefined)).toBe(7)
  })
})
