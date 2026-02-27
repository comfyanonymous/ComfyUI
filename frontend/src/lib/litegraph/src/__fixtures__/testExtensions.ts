// oxlint-disable no-empty-pattern
import { test as baseTest } from 'vitest'

import { LGraph } from '@/lib/litegraph/src/LGraph'
import { LiteGraph } from '@/lib/litegraph/src/litegraph'
import type {
  ISerialisedGraph,
  SerialisableGraph
} from '@/lib/litegraph/src/types/serialisation'

import floatingBranch from './assets/floatingBranch.json' with { type: 'json' }
import floatingLink from './assets/floatingLink.json' with { type: 'json' }
import linkedNodes from './assets/linkedNodes.json' with { type: 'json' }
import reroutesComplex from './assets/reroutesComplex.json' with { type: 'json' }
import {
  basicSerialisableGraph,
  minimalSerialisableGraph,
  oldSchemaGraph
} from './assets/testGraphs'

interface LitegraphFixtures {
  minimalGraph: LGraph
  minimalSerialisableGraph: SerialisableGraph
  oldSchemaGraph: ISerialisedGraph
  floatingLinkGraph: ISerialisedGraph
  linkedNodesGraph: ISerialisedGraph
  floatingBranchGraph: LGraph
  reroutesComplexGraph: LGraph
}

/** These fixtures alter global state, and are difficult to reset. Relies on a single test per-file to reset state. */
interface DirtyFixtures {
  basicSerialisableGraph: SerialisableGraph
}

export const test = baseTest.extend<LitegraphFixtures>({
  minimalGraph: async ({}, use) => {
    // Before each test function
    const serialisable = structuredClone(minimalSerialisableGraph)
    const lGraph = new LGraph(serialisable)

    // use the fixture value
    await use(lGraph)
  },
  minimalSerialisableGraph: structuredClone(minimalSerialisableGraph),
  oldSchemaGraph: structuredClone(oldSchemaGraph),
  floatingLinkGraph: structuredClone(
    floatingLink as unknown as ISerialisedGraph
  ),
  linkedNodesGraph: structuredClone(linkedNodes as unknown as ISerialisedGraph),

  floatingBranchGraph: async ({}, use) => {
    const cloned = structuredClone(
      floatingBranch as unknown as ISerialisedGraph
    )
    const graph = new LGraph(cloned)
    await use(graph)
  },

  reroutesComplexGraph: async ({}, use) => {
    const cloned = structuredClone(
      reroutesComplex as unknown as ISerialisedGraph
    )
    const graph = new LGraph(cloned)
    await use(graph)
  }
})

/** Test that use {@link DirtyFixtures}. One test per file. */
export const dirtyTest = test.extend<DirtyFixtures>({
  basicSerialisableGraph: async ({}, use) => {
    if (!basicSerialisableGraph.nodes) throw new Error('Invalid test object')

    // Register node types
    for (const node of basicSerialisableGraph.nodes) {
      LiteGraph.registerNodeType(node.type!, LiteGraph.LGraphNode)
    }

    await use(structuredClone(basicSerialisableGraph))
  }
})
