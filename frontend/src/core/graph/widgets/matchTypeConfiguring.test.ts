import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, test, vi } from 'vitest'

import { LGraph, LGraphNode, LiteGraph } from '@/lib/litegraph/src/litegraph'
import { transformInputSpecV1ToV2 } from '@/schemas/nodeDef/migration'
import { app } from '@/scripts/app'
import { useLitegraphService } from '@/services/litegraphService'

setActivePinia(createTestingPinia())

const { addNodeInput } = useLitegraphService()

function createMatchTypeNode(graph: LGraph) {
  const node = new LGraphNode('switch')
  ;(node.constructor as { nodeData: unknown }).nodeData = {
    name: 'ComfySwitchAny',
    output_matchtypes: ['a']
  }
  node.addOutput('out', '*')
  graph.add(node)

  addNodeInput(
    node,
    transformInputSpecV1ToV2(
      [
        'COMFY_MATCHTYPE_V3',
        { template: { allowed_types: '*', template_id: 'a' } }
      ],
      { name: 'on_true', isOptional: false }
    )
  )
  addNodeInput(
    node,
    transformInputSpecV1ToV2(
      [
        'COMFY_MATCHTYPE_V3',
        { template: { allowed_types: '*', template_id: 'a' } }
      ],
      { name: 'on_false', isOptional: false }
    )
  )

  return node
}

function createSourceNode(graph: LGraph, type: string) {
  const node = new LGraphNode('source')
  node.addOutput('out', type)
  graph.add(node)
  return node
}

describe('MatchType during configure', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  test('skips type recalculation when configuringGraph is true', () => {
    const graph = new LGraph()
    const switchNode = createMatchTypeNode(graph)
    const source1 = createSourceNode(graph, 'IMAGE')
    const source2 = createSourceNode(graph, 'IMAGE')

    source1.connect(0, switchNode, 0)
    source2.connect(0, switchNode, 1)

    expect(switchNode.inputs[0].link).not.toBeNull()
    expect(switchNode.inputs[1].link).not.toBeNull()

    const link1Id = switchNode.inputs[0].link!
    const link2Id = switchNode.inputs[1].link!

    const outputTypeBefore = switchNode.outputs[0].type
    ;(
      app as unknown as { configuringGraphLevel: number }
    ).configuringGraphLevel = 1

    try {
      const link1 = graph.links[link1Id]
      switchNode.onConnectionsChange?.(
        LiteGraph.INPUT,
        0,
        true,
        link1,
        switchNode.inputs[0]
      )

      expect(switchNode.inputs[0].link).toBe(link1Id)
      expect(switchNode.inputs[1].link).toBe(link2Id)
      expect(graph.links[link1Id]).toBeDefined()
      expect(graph.links[link2Id]).toBeDefined()
      expect(switchNode.outputs[0].type).toBe(outputTypeBefore)
    } finally {
      ;(
        app as unknown as { configuringGraphLevel: number }
      ).configuringGraphLevel = 0
    }
  })

  test('performs type recalculation during normal operation', () => {
    const graph = new LGraph()
    const switchNode = createMatchTypeNode(graph)
    const source1 = createSourceNode(graph, 'IMAGE')

    expect(app.configuringGraph).toBe(false)

    source1.connect(0, switchNode, 0)

    expect(switchNode.inputs[0].link).not.toBeNull()
    expect(switchNode.outputs[0].type).toBe('IMAGE')
  })

  test('connects both inputs with same type', () => {
    const graph = new LGraph()
    const switchNode = createMatchTypeNode(graph)
    const source1 = createSourceNode(graph, 'IMAGE')
    const source2 = createSourceNode(graph, 'IMAGE')

    expect(app.configuringGraph).toBe(false)

    source1.connect(0, switchNode, 0)
    source2.connect(0, switchNode, 1)

    expect(switchNode.inputs[0].link).not.toBeNull()
    expect(switchNode.inputs[1].link).not.toBeNull()
    expect(switchNode.outputs[0].type).toBe('IMAGE')
  })
})
