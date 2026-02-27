import { mount } from '@vue/test-utils'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import DomWidgets from '@/components/graph/DomWidgets.vue'
import { Rectangle } from '@/lib/litegraph/src/infrastructure/Rectangle'
import { LGraph, LGraphNode } from '@/lib/litegraph/src/litegraph'
import type { LGraphCanvas } from '@/lib/litegraph/src/LGraphCanvas'
import { useCanvasStore } from '@/renderer/core/canvas/canvasStore'
import type { BaseDOMWidget } from '@/scripts/domWidget'
import { useDomWidgetStore } from '@/stores/domWidgetStore'
import { createTestingPinia } from '@pinia/testing'

type TestWidget = BaseDOMWidget<object | string>

function createNode(
  graph: LGraph,
  id: number,
  title: string,
  pos: [number, number]
) {
  const node = new LGraphNode(title)
  node.id = id
  node.pos = [...pos]
  node.size = [240, 120]
  graph.add(node)
  return node
}

function createWidget(id: string, node: LGraphNode, y = 12): TestWidget {
  return {
    id,
    node,
    name: 'test_widget',
    type: 'custom',
    value: '',
    options: {},
    y,
    width: 120,
    computedHeight: 40,
    margin: 10,
    isVisible: () => true
  } as unknown as TestWidget
}

function createCanvas(graph: LGraph): LGraphCanvas {
  return {
    graph,
    low_quality: false,
    read_only: false,
    isNodeVisible: vi.fn(() => true)
  } as unknown as LGraphCanvas
}

function drawFrame(canvas: LGraphCanvas) {
  canvas.onDrawForeground?.({} as CanvasRenderingContext2D, new Rectangle())
}

describe('DomWidgets transition grace characterization', () => {
  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
  })

  it('applies transition grace for exactly one frame when override exists but is not active', () => {
    const canvasStore = useCanvasStore()
    const domWidgetStore = useDomWidgetStore()

    const graphA = new LGraph()
    const graphB = new LGraph()
    const interiorNode = createNode(graphA, 1, 'interior', [100, 200])
    const overrideNode = createNode(graphB, 2, 'override', [600, 700])

    const widget = createWidget('widget-transition', interiorNode, 14)
    const overrideWidget = createWidget('override-widget', overrideNode, 22)

    domWidgetStore.registerWidget(widget)
    domWidgetStore.setPositionOverride(widget.id, {
      node: overrideNode,
      widget: overrideWidget
    })
    domWidgetStore.deactivateWidget(widget.id)

    const widgetState = domWidgetStore.widgetStates.get(widget.id)
    if (!widgetState) throw new Error('Widget state not registered')
    widgetState.visible = true
    widgetState.pos = [321, 654]

    const canvas = createCanvas(graphA)
    canvasStore.canvas = canvas

    mount(DomWidgets, {
      global: {
        stubs: {
          DomWidget: true
        }
      }
    })

    drawFrame(canvas)
    expect(widgetState.visible).toBe(true)
    expect(widgetState.pos).toEqual([321, 654])

    drawFrame(canvas)
    expect(widgetState.visible).toBe(false)
  })

  it('uses override positioning while override node is in current graph even when widget is inactive', () => {
    const canvasStore = useCanvasStore()
    const domWidgetStore = useDomWidgetStore()

    const graphA = new LGraph()
    const graphB = new LGraph()
    const interiorNode = createNode(graphA, 1, 'interior', [10, 20])
    const overrideNode = createNode(graphB, 2, 'override', [300, 400])

    const widget = createWidget('widget-override-active', interiorNode, 8)
    const overrideWidget = createWidget(
      'override-position-source',
      overrideNode,
      18
    )

    domWidgetStore.registerWidget(widget)
    domWidgetStore.setPositionOverride(widget.id, {
      node: overrideNode,
      widget: overrideWidget
    })
    domWidgetStore.deactivateWidget(widget.id)

    const widgetState = domWidgetStore.widgetStates.get(widget.id)
    if (!widgetState) throw new Error('Widget state not registered')

    const canvas = createCanvas(graphB)
    canvasStore.canvas = canvas

    mount(DomWidgets, {
      global: {
        stubs: {
          DomWidget: true
        }
      }
    })

    drawFrame(canvas)

    expect(widgetState.visible).toBe(true)
    expect(widgetState.pos).toEqual([310, 428])
  })

  it('cleans orphaned transition-grace ids after widget removal', () => {
    const canvasStore = useCanvasStore()
    const domWidgetStore = useDomWidgetStore()

    const graphA = new LGraph()
    const graphB = new LGraph()
    const interiorNode = createNode(graphA, 1, 'interior', [0, 0])
    const overrideNode = createNode(graphB, 2, 'override', [200, 200])

    const canvas = createCanvas(graphA)
    canvasStore.canvas = canvas

    mount(DomWidgets, {
      global: {
        stubs: {
          DomWidget: true
        }
      }
    })

    const oldWidget = createWidget('shared-widget-id', interiorNode, 10)
    const overrideWidget = createWidget(
      'shared-override-widget',
      overrideNode,
      14
    )

    domWidgetStore.registerWidget(oldWidget)
    domWidgetStore.setPositionOverride(oldWidget.id, {
      node: overrideNode,
      widget: overrideWidget
    })
    domWidgetStore.deactivateWidget(oldWidget.id)

    drawFrame(canvas)
    domWidgetStore.unregisterWidget(oldWidget.id)

    drawFrame(canvas)

    const replacementWidget = createWidget('shared-widget-id', interiorNode, 10)
    domWidgetStore.registerWidget(replacementWidget)
    domWidgetStore.setPositionOverride(replacementWidget.id, {
      node: overrideNode,
      widget: overrideWidget
    })
    domWidgetStore.deactivateWidget(replacementWidget.id)

    const replacementState = domWidgetStore.widgetStates.get(
      replacementWidget.id
    )
    if (!replacementState) throw new Error('Replacement widget missing state')
    replacementState.visible = true
    replacementState.pos = [999, 999]

    drawFrame(canvas)

    expect(replacementState.visible).toBe(true)
    expect(replacementState.pos).toEqual([999, 999])
  })
})
