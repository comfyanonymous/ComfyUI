import { describe, expect, it, vi } from 'vitest'

import { resolveSubgraphPseudoWidgetCache } from '@/services/subgraphPseudoWidgetCache'
import type {
  SubgraphPseudoWidget,
  SubgraphPseudoWidgetCache,
  SubgraphPseudoWidgetNode,
  SubgraphPromotionEntry
} from '@/services/subgraphPseudoWidgetCache'

interface TestWidget extends SubgraphPseudoWidget {
  isPseudo?: boolean
}

interface TestNode extends SubgraphPseudoWidgetNode<TestWidget> {
  widgets?: TestWidget[]
}

function widget(name: string, isPseudo = false): TestWidget {
  return { name, isPseudo }
}

function node(id: string, widgets: TestWidget[] = []): TestNode {
  return { id, widgets }
}

describe('resolveSubgraphPseudoWidgetCache', () => {
  it('builds update targets from pseudo widget promotions', () => {
    const interiorNode = node('n1', [widget('preview', true)])
    const getNodeById = vi.fn((id: string) =>
      id === 'n1' ? interiorNode : undefined
    )
    const promotions: readonly SubgraphPromotionEntry[] = [
      { interiorNodeId: 'n1', widgetName: 'preview' }
    ]

    const result = resolveSubgraphPseudoWidgetCache<TestNode, TestWidget>({
      cache: null,
      promotions,
      getNodeById,
      isPreviewPseudoWidget: (candidate) => candidate.isPseudo === true
    })

    expect(result.nodes).toEqual([interiorNode])
    expect(result.cache.entries).toHaveLength(1)
    expect(getNodeById).toHaveBeenCalledWith('n1')
  })

  it('keeps $$ fallback behavior when the backing widget is missing', () => {
    const interiorNode = node('n1', [widget('other')])
    const promotions: readonly SubgraphPromotionEntry[] = [
      { interiorNodeId: 'n1', widgetName: '$$canvas-image-preview' }
    ]

    const result = resolveSubgraphPseudoWidgetCache<TestNode, TestWidget>({
      cache: null,
      promotions,
      getNodeById: (id) => (id === 'n1' ? interiorNode : undefined),
      isPreviewPseudoWidget: (candidate) => candidate.isPseudo === true
    })

    expect(result.nodes).toEqual([interiorNode])
  })

  it('reuses cache when promotions and node identities are unchanged', () => {
    const interiorNode = node('n1', [widget('preview', true)])
    const promotions: readonly SubgraphPromotionEntry[] = [
      { interiorNodeId: 'n1', widgetName: 'preview' }
    ]
    const base = resolveSubgraphPseudoWidgetCache<TestNode, TestWidget>({
      cache: null,
      promotions,
      getNodeById: (id) => (id === 'n1' ? interiorNode : undefined),
      isPreviewPseudoWidget: (candidate) => candidate.isPseudo === true
    })
    const getNodeById = vi.fn((id: string) =>
      id === 'n1' ? interiorNode : undefined
    )

    const result = resolveSubgraphPseudoWidgetCache<TestNode, TestWidget>({
      cache: base.cache,
      promotions,
      getNodeById,
      isPreviewPseudoWidget: (candidate) => candidate.isPseudo === true
    })

    expect(result.cache).toBe(base.cache)
    expect(result.nodes).toBe(base.nodes)
    expect(getNodeById).toHaveBeenCalledTimes(1)
  })

  it('rebuilds cache when promotions reference changes', () => {
    const interiorNode = node('n1', [widget('preview', true)])
    const promotionsA: readonly SubgraphPromotionEntry[] = [
      { interiorNodeId: 'n1', widgetName: 'preview' }
    ]
    const base = resolveSubgraphPseudoWidgetCache<TestNode, TestWidget>({
      cache: null,
      promotions: promotionsA,
      getNodeById: (id) => (id === 'n1' ? interiorNode : undefined),
      isPreviewPseudoWidget: (candidate) => candidate.isPseudo === true
    })
    const promotionsB: readonly SubgraphPromotionEntry[] = [
      { interiorNodeId: 'n1', widgetName: 'preview' }
    ]

    const result = resolveSubgraphPseudoWidgetCache<TestNode, TestWidget>({
      cache: base.cache,
      promotions: promotionsB,
      getNodeById: (id) => (id === 'n1' ? interiorNode : undefined),
      isPreviewPseudoWidget: (candidate) => candidate.isPseudo === true
    })

    expect(result.cache).not.toBe(base.cache)
  })

  it('falls back to rebuild when a cached node reference goes stale', () => {
    const oldNode = node('n1', [widget('preview', true)])
    const promotions: readonly SubgraphPromotionEntry[] = [
      { interiorNodeId: 'n1', widgetName: 'preview' }
    ]
    const initial = resolveSubgraphPseudoWidgetCache<TestNode, TestWidget>({
      cache: null,
      promotions,
      getNodeById: () => oldNode,
      isPreviewPseudoWidget: (candidate) => candidate.isPseudo === true
    })
    const newNode = node('n1', [widget('preview', true)])

    const result = resolveSubgraphPseudoWidgetCache<TestNode, TestWidget>({
      cache: initial.cache,
      promotions,
      getNodeById: () => newNode,
      isPreviewPseudoWidget: (candidate) => candidate.isPseudo === true
    })

    expect(result.cache).not.toBe(initial.cache)
    expect(result.nodes).toEqual([newNode])
  })

  it('drops cached entries when node no longer resolves', () => {
    const promotions: readonly SubgraphPromotionEntry[] = [
      { interiorNodeId: 'missing', widgetName: '$$canvas-image-preview' }
    ]
    const cache: SubgraphPseudoWidgetCache<TestNode, TestWidget> = {
      promotions,
      entries: [
        {
          interiorNodeId: 'missing',
          widgetName: '$$canvas-image-preview',
          node: node('missing')
        }
      ],
      nodes: [node('missing')]
    }

    const result = resolveSubgraphPseudoWidgetCache<TestNode, TestWidget>({
      cache,
      promotions,
      getNodeById: () => undefined,
      isPreviewPseudoWidget: (candidate) => candidate.isPseudo === true
    })

    expect(result.nodes).toEqual([])
    expect(result.cache.entries).toEqual([])
  })
})
