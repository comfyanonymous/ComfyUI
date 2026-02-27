import { describe, expect } from 'vitest'

import { LGraph, LGraphGroup } from '@/lib/litegraph/src/litegraph'

import { test } from './__fixtures__/testExtensions'

describe('LGraphGroup', () => {
  test('serializes to the existing format', () => {
    const link = new LGraphGroup('title', 929)
    expect(link.serialize()).toMatchSnapshot('Basic')
  })

  describe('recomputeInsideNodes', () => {
    test('uses visited set to avoid redundant computation', () => {
      const graph = new LGraph()

      // Create 4 nested groups: outer -> mid1 -> mid2 -> inner
      const outer = new LGraphGroup('outer')
      outer.pos = [0, 0]
      outer.size = [400, 400]
      graph.add(outer)

      const mid1 = new LGraphGroup('mid1')
      mid1.pos = [10, 10]
      mid1.size = [300, 300]
      graph.add(mid1)

      const mid2 = new LGraphGroup('mid2')
      mid2.pos = [20, 20]
      mid2.size = [200, 200]
      graph.add(mid2)

      const inner = new LGraphGroup('inner')
      inner.pos = [30, 30]
      inner.size = [100, 100]
      graph.add(inner)

      // Track the visited set to verify each group is only fully processed once
      const visited = new Set<number>()
      outer.recomputeInsideNodes(100, visited)

      // All nested groups should be in the visited set
      expect(visited.has(outer.id)).toBe(true)
      expect(visited.has(mid1.id)).toBe(true)
      expect(visited.has(mid2.id)).toBe(true)
      expect(visited.has(inner.id)).toBe(true)
      expect(visited.size).toBe(4)

      // Verify children relationships are correct
      expect(outer.children.has(mid1)).toBe(true)
      expect(outer.children.has(mid2)).toBe(true)
      expect(outer.children.has(inner)).toBe(true)
      expect(mid1.children.has(mid2)).toBe(true)
      expect(mid1.children.has(inner)).toBe(true)
      expect(mid2.children.has(inner)).toBe(true)
    })

    test('respects maxDepth limit', () => {
      const graph = new LGraph()

      const outer = new LGraphGroup('outer')
      outer.pos = [0, 0]
      outer.size = [300, 300]
      graph.add(outer)

      const inner = new LGraphGroup('inner')
      inner.pos = [10, 10]
      inner.size = [100, 100]
      graph.add(inner)

      // With maxDepth=1, inner group is added as child but not processed
      outer.recomputeInsideNodes(1)

      // outer should have inner as a child
      expect(outer.children.has(inner)).toBe(true)
      // inner should not have computed its own children (it was never processed)
      expect(inner.children.size).toBe(0)
    })
  })
})
