import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import { LGraphNode } from '@/lib/litegraph/src/litegraph'
import {
  createTestSubgraph,
  createTestSubgraphNode
} from '@/lib/litegraph/src/subgraph/__fixtures__/subgraphHelpers'
import { usePromotionStore } from '@/stores/promotionStore'

import { hasUnpromotedWidgets } from './unpromotedWidgetUtils'

describe('hasUnpromotedWidgets', () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
  })

  it('returns true when subgraph has at least one enabled unpromoted widget', () => {
    const subgraph = createTestSubgraph()
    const subgraphNode = createTestSubgraphNode(subgraph)
    const interiorNode = new LGraphNode('InnerNode')
    subgraph.add(interiorNode)
    interiorNode.addWidget('text', 'seed', '123', () => {})

    expect(hasUnpromotedWidgets(subgraphNode)).toBe(true)
  })

  it('returns false when all enabled widgets are already promoted', () => {
    const subgraph = createTestSubgraph()
    const subgraphNode = createTestSubgraphNode(subgraph)
    const interiorNode = new LGraphNode('InnerNode')
    subgraph.add(interiorNode)
    interiorNode.addWidget('text', 'seed', '123', () => {})

    usePromotionStore().promote(
      subgraphNode.rootGraph.id,
      subgraphNode.id,
      String(interiorNode.id),
      'seed'
    )

    expect(hasUnpromotedWidgets(subgraphNode)).toBe(false)
  })

  it('ignores computed-disabled widgets', () => {
    const subgraph = createTestSubgraph()
    const subgraphNode = createTestSubgraphNode(subgraph)
    const interiorNode = new LGraphNode('InnerNode')
    subgraph.add(interiorNode)
    const widget = interiorNode.addWidget('text', 'seed', '123', () => {})
    widget.computedDisabled = true

    expect(hasUnpromotedWidgets(subgraphNode)).toBe(false)
  })
})
