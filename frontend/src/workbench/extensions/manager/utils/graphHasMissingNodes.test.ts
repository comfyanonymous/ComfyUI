import { describe, expect, it } from 'vitest'

import type {
  LGraph,
  LGraphNode,
  Subgraph
} from '@/lib/litegraph/src/litegraph'
import {
  collectMissingNodes,
  graphHasMissingNodes
} from '@/workbench/extensions/manager/utils/graphHasMissingNodes'
import type { NodeDefLookup } from '@/workbench/extensions/manager/utils/graphHasMissingNodes'
import type { ComfyNodeDefImpl } from '@/stores/nodeDefStore'

type NodeDefs = NodeDefLookup

let nodeIdCounter = 0
const mockNodeDef = {} as ComfyNodeDefImpl

const createGraph = (nodes: LGraphNode[] = []): LGraph => {
  return { nodes } as Partial<LGraph> as LGraph
}

const createSubgraph = (nodes: LGraphNode[]): Subgraph => {
  return { nodes } as Partial<Subgraph> as Subgraph
}

const createNode = (
  type?: string,
  subgraphNodes?: LGraphNode[]
): LGraphNode => {
  return {
    id: nodeIdCounter++,
    type,
    isSubgraphNode: subgraphNodes ? () => true : undefined,
    subgraph: subgraphNodes ? createSubgraph(subgraphNodes) : undefined
  } as unknown as LGraphNode
}

describe('graphHasMissingNodes', () => {
  it('returns false when graph is null', () => {
    expect(graphHasMissingNodes(null, {})).toBe(false)
  })

  it('returns false when graph is undefined', () => {
    expect(graphHasMissingNodes(undefined, {})).toBe(false)
  })

  it('returns false when graph has no nodes', () => {
    expect(graphHasMissingNodes(createGraph(), {})).toBe(false)
  })

  it('returns false when every node has a definition', () => {
    const graph = createGraph([createNode('FooNode'), createNode('BarNode')])
    const nodeDefs: NodeDefs = {
      FooNode: mockNodeDef,
      BarNode: mockNodeDef
    }

    expect(graphHasMissingNodes(graph, nodeDefs)).toBe(false)
  })

  it('returns true when at least one node is missing', () => {
    const graph = createGraph([
      createNode('FooNode'),
      createNode('MissingNode')
    ])
    const nodeDefs: NodeDefs = {
      FooNode: mockNodeDef
    }

    expect(graphHasMissingNodes(graph, nodeDefs)).toBe(true)
  })

  it('checks nodes nested in subgraphs', () => {
    const graph = createGraph([
      createNode('ContainerNode', [createNode('InnerMissing')])
    ])
    const nodeDefs: NodeDefs = {
      ContainerNode: mockNodeDef
    }

    const missingNodes = collectMissingNodes(graph, nodeDefs)
    expect(missingNodes).toHaveLength(1)
    expect(missingNodes[0]?.type).toBe('InnerMissing')
  })

  it('ignores nodes without a type', () => {
    const graph = createGraph([createNode(undefined), createNode(null!)])

    expect(graphHasMissingNodes(graph, {})).toBe(false)
  })

  it('traverses deeply nested subgraphs', () => {
    const deepGraph = createGraph([
      createNode('Layer1', [
        createNode('Layer2', [
          createNode('Layer3', [createNode('MissingDeep')])
        ])
      ])
    ])
    const nodeDefs: NodeDefs = {
      Layer1: mockNodeDef,
      Layer2: mockNodeDef,
      Layer3: mockNodeDef
    }

    const missingNodes = collectMissingNodes(deepGraph, nodeDefs)
    expect(missingNodes).toHaveLength(1)
    expect(missingNodes[0]?.type).toBe('MissingDeep')
  })
})
