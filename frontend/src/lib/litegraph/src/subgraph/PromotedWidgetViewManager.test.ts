import { describe, expect, test } from 'vitest'

import { PromotedWidgetViewManager } from '@/lib/litegraph/src/subgraph/PromotedWidgetViewManager'
import type { SubgraphPromotionEntry } from '@/services/subgraphPseudoWidgetCache'

function makeView(entry: SubgraphPromotionEntry) {
  return {
    key: `${entry.interiorNodeId}:${entry.widgetName}`
  }
}

describe('PromotedWidgetViewManager', () => {
  test('returns memoized array when entries reference is unchanged', () => {
    const manager = new PromotedWidgetViewManager<{ key: string }>()
    const entries = [{ interiorNodeId: '1', widgetName: 'widgetA' }]

    const first = manager.reconcile(entries, makeView)
    const second = manager.reconcile(entries, makeView)

    expect(second).toBe(first)
    expect(second[0]).toBe(first[0])
  })

  test('preserves view identity while reflecting order changes', () => {
    const manager = new PromotedWidgetViewManager<{ key: string }>()

    const firstPass = manager.reconcile(
      [
        { interiorNodeId: '1', widgetName: 'widgetA' },
        { interiorNodeId: '1', widgetName: 'widgetB' }
      ],
      makeView
    )

    const reordered = manager.reconcile(
      [
        { interiorNodeId: '1', widgetName: 'widgetB' },
        { interiorNodeId: '1', widgetName: 'widgetA' }
      ],
      makeView
    )

    expect(reordered[0]).toBe(firstPass[1])
    expect(reordered[1]).toBe(firstPass[0])
  })

  test('deduplicates by first occurrence and clears stale cache entries', () => {
    const manager = new PromotedWidgetViewManager<{ key: string }>()

    const first = manager.reconcile(
      [
        { interiorNodeId: '1', widgetName: 'widgetA' },
        { interiorNodeId: '1', widgetName: 'widgetB' },
        { interiorNodeId: '1', widgetName: 'widgetA' }
      ],
      makeView
    )
    expect(first.map((view) => view.key)).toStrictEqual([
      '1:widgetA',
      '1:widgetB'
    ])

    manager.reconcile(
      [{ interiorNodeId: '1', widgetName: 'widgetB' }],
      makeView
    )

    const restored = manager.reconcile(
      [
        { interiorNodeId: '1', widgetName: 'widgetB' },
        { interiorNodeId: '1', widgetName: 'widgetA' }
      ],
      makeView
    )

    expect(restored[0]).toBe(first[1])
    expect(restored[1]).not.toBe(first[0])
  })
})
