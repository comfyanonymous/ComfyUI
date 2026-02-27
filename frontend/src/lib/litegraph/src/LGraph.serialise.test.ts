import { describe } from 'vitest'

import { LGraph, LGraphGroup, LGraphNode } from '@/lib/litegraph/src/litegraph'
import type { ISerialisedGraph } from '@/lib/litegraph/src/litegraph'

import { test } from './__fixtures__/testExtensions'

describe('LGraph Serialisation', () => {
  test('can (de)serialise node / group titles', ({ expect, minimalGraph }) => {
    const nodeTitle = 'Test Node'
    const groupTitle = 'Test Group'

    minimalGraph.add(new LGraphNode(nodeTitle))
    minimalGraph.add(new LGraphGroup(groupTitle))

    expect(minimalGraph.nodes.length).toBe(1)
    expect(minimalGraph.nodes[0].title).toEqual(nodeTitle)

    expect(minimalGraph.groups.length).toBe(1)
    expect(minimalGraph.groups[0].title).toEqual(groupTitle)

    const serialised = JSON.stringify(minimalGraph.serialize())
    const deserialised = JSON.parse(serialised) as ISerialisedGraph

    const copied = new LGraph(deserialised)
    expect(copied.nodes.length).toBe(1)
    expect(copied.groups.length).toBe(1)
  })
})
