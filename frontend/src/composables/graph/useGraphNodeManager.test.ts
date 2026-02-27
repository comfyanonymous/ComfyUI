import { setActivePinia } from 'pinia'
import { createTestingPinia } from '@pinia/testing'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { computed, nextTick, watch } from 'vue'

import { useGraphNodeManager } from '@/composables/graph/useGraphNodeManager'
import { createPromotedWidgetView } from '@/core/graph/subgraph/promotedWidgetView'
import { BaseWidget, LGraph, LGraphNode } from '@/lib/litegraph/src/litegraph'
import {
  createTestSubgraph,
  createTestSubgraphNode
} from '@/lib/litegraph/src/subgraph/__fixtures__/subgraphHelpers'
import { NodeSlotType } from '@/lib/litegraph/src/types/globalEnums'
import { usePromotionStore } from '@/stores/promotionStore'
import { useWidgetValueStore } from '@/stores/widgetValueStore'

describe('Node Reactivity', () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
  })

  function createTestGraph() {
    const graph = new LGraph()
    const node = new LGraphNode('test')
    node.addInput('input', 'INT')
    node.addWidget('number', 'testnum', 2, () => undefined, {})
    graph.add(node)

    const { vueNodeData } = useGraphNodeManager(graph)

    return { node, graph, vueNodeData }
  }

  it('widget values are reactive through the store', async () => {
    const { node, graph } = createTestGraph()
    const store = useWidgetValueStore()
    const widget = node.widgets![0]

    // Verify widget is a BaseWidget with correct value and node assignment
    expect(widget).toBeInstanceOf(BaseWidget)
    expect(widget.value).toBe(2)
    expect((widget as BaseWidget).node.id).toBe(node.id)

    // Initial value should be in store after setNodeId was called
    expect(store.getWidget(graph.id, node.id, 'testnum')?.value).toBe(2)

    const state = store.getWidget(graph.id, node.id, 'testnum')
    if (!state) throw new Error('Expected widget state to exist')

    const onValueChange = vi.fn()
    const widgetValue = computed(() => state.value)
    watch(widgetValue, onValueChange)

    widget.value = 42
    await nextTick()

    expect(widgetValue.value).toBe(42)
    expect(onValueChange).toHaveBeenCalledTimes(1)
  })

  it('widget values remain reactive after a connection is made', async () => {
    const { node, graph } = createTestGraph()
    const store = useWidgetValueStore()
    const onValueChange = vi.fn()

    graph.trigger('node:slot-links:changed', {
      nodeId: String(node.id),
      slotType: NodeSlotType.INPUT
    })
    await nextTick()

    const state = store.getWidget(graph.id, node.id, 'testnum')
    if (!state) throw new Error('Expected widget state to exist')

    const widgetValue = computed(() => state.value)
    watch(widgetValue, onValueChange)

    node.widgets![0].value = 99
    await nextTick()

    expect(onValueChange).toHaveBeenCalledTimes(1)
    expect(widgetValue.value).toBe(99)
  })
})

describe('Widget slotMetadata reactivity on link disconnect', () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
  })

  function createWidgetInputGraph() {
    const graph = new LGraph()
    const node = new LGraphNode('test')

    // Add a widget and an associated input slot (simulates "widget converted to input")
    node.addWidget('string', 'prompt', 'hello', () => undefined, {})
    const input = node.addInput('prompt', 'STRING')
    // Associate the input slot with the widget (as widgetInputs extension does)
    input.widget = { name: 'prompt' }

    // Start with a connected link
    input.link = 42

    graph.add(node)
    return { graph, node }
  }

  it('sets slotMetadata.linked to true when input has a link', () => {
    const { graph, node } = createWidgetInputGraph()
    const { vueNodeData } = useGraphNodeManager(graph)

    const nodeData = vueNodeData.get(String(node.id))
    const widgetData = nodeData?.widgets?.find((w) => w.name === 'prompt')

    expect(widgetData?.slotMetadata).toBeDefined()
    expect(widgetData?.slotMetadata?.linked).toBe(true)
  })

  it('updates slotMetadata.linked to false after link disconnect event', async () => {
    const { graph, node } = createWidgetInputGraph()
    const { vueNodeData } = useGraphNodeManager(graph)

    const nodeData = vueNodeData.get(String(node.id))
    const widgetData = nodeData?.widgets?.find((w) => w.name === 'prompt')

    // Verify initially linked
    expect(widgetData?.slotMetadata?.linked).toBe(true)

    // Simulate link disconnection (as LiteGraph does before firing the event)
    node.inputs[0].link = null

    // Fire the trigger event that LiteGraph fires on disconnect
    graph.trigger('node:slot-links:changed', {
      nodeId: node.id,
      slotType: NodeSlotType.INPUT,
      slotIndex: 0,
      connected: false,
      linkId: 42
    })

    await nextTick()

    // slotMetadata.linked should now be false
    expect(widgetData?.slotMetadata?.linked).toBe(false)
  })

  it('reactively updates disabled state in a derived computed after disconnect', async () => {
    const { graph, node } = createWidgetInputGraph()
    const { vueNodeData } = useGraphNodeManager(graph)

    const nodeData = vueNodeData.get(String(node.id))!

    // Mimic what processedWidgets does in NodeWidgets.vue:
    // derive disabled from slotMetadata.linked
    const derivedDisabled = computed(() => {
      const widgets = nodeData.widgets ?? []
      const widget = widgets.find((w) => w.name === 'prompt')
      return widget?.slotMetadata?.linked ? true : false
    })

    // Initially linked → disabled
    expect(derivedDisabled.value).toBe(true)

    // Track changes
    const onChange = vi.fn()
    watch(derivedDisabled, onChange)

    // Simulate disconnect
    node.inputs[0].link = null
    graph.trigger('node:slot-links:changed', {
      nodeId: node.id,
      slotType: NodeSlotType.INPUT,
      slotIndex: 0,
      connected: false,
      linkId: 42
    })

    await nextTick()

    // The derived computed should now return false
    expect(derivedDisabled.value).toBe(false)
    expect(onChange).toHaveBeenCalledTimes(1)
  })

  it('updates slotMetadata for promoted widgets where SafeWidgetData.name differs from input.widget.name', async () => {
    // Set up a subgraph with an interior node that has a "prompt" widget.
    // createPromotedWidgetView resolves against this interior node.
    const subgraph = createTestSubgraph()
    const interiorNode = new LGraphNode('interior')
    interiorNode.id = 10
    interiorNode.addWidget('string', 'prompt', 'hello', () => undefined, {})
    subgraph.add(interiorNode)

    const subgraphNode = createTestSubgraphNode(subgraph, { id: 123 })

    // Create a PromotedWidgetView with displayName="value" (subgraph input
    // slot name) and sourceWidgetName="prompt" (interior widget name).
    // PromotedWidgetView.name returns "value", but safeWidgetMapper sets
    // SafeWidgetData.name to sourceWidgetName ("prompt").
    const promotedView = createPromotedWidgetView(
      subgraphNode,
      '10',
      'prompt',
      'value'
    )

    // Host the promoted view on a regular node so we can control widgets
    // directly (SubgraphNode.widgets is a synthetic getter).
    const graph = new LGraph()
    const hostNode = new LGraphNode('host')
    hostNode.widgets = [promotedView]
    const input = hostNode.addInput('value', 'STRING')
    input.widget = { name: 'value' }
    input.link = 42
    graph.add(hostNode)

    const { vueNodeData } = useGraphNodeManager(graph)
    const nodeData = vueNodeData.get(String(hostNode.id))

    // SafeWidgetData.name is "prompt" (sourceWidgetName), but the
    // input slot widget name is "value" — slotName bridges this gap.
    const widgetData = nodeData?.widgets?.find((w) => w.name === 'prompt')
    expect(widgetData).toBeDefined()
    expect(widgetData?.slotName).toBe('value')
    expect(widgetData?.slotMetadata?.linked).toBe(true)

    // Disconnect
    hostNode.inputs[0].link = null
    graph.trigger('node:slot-links:changed', {
      nodeId: hostNode.id,
      slotType: NodeSlotType.INPUT,
      slotIndex: 0,
      connected: false,
      linkId: 42
    })

    await nextTick()

    expect(widgetData?.slotMetadata?.linked).toBe(false)
  })
})

describe('Subgraph Promoted Pseudo Widgets', () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
  })

  it('marks promoted $$ widgets as canvasOnly for Vue widget rendering', () => {
    const subgraph = createTestSubgraph()
    const interiorNode = new LGraphNode('interior')
    interiorNode.id = 10
    subgraph.add(interiorNode)

    const subgraphNode = createTestSubgraphNode(subgraph, { id: 123 })
    const graph = subgraphNode.graph as LGraph
    graph.add(subgraphNode)

    usePromotionStore().promote(
      subgraphNode.rootGraph.id,
      subgraphNode.id,
      '10',
      '$$canvas-image-preview'
    )

    const { vueNodeData } = useGraphNodeManager(graph)
    const vueNode = vueNodeData.get(String(subgraphNode.id))
    const promotedWidget = vueNode?.widgets?.find(
      (widget) => widget.name === '$$canvas-image-preview'
    )

    expect(promotedWidget).toBeDefined()
    expect(promotedWidget?.options?.canvasOnly).toBe(true)
  })
})
