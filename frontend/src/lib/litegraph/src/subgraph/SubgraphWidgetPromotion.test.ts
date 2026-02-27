// TODO: Fix these tests after migration
import { describe, expect, it } from 'vitest'

import type {
  ISlotType,
  Subgraph,
  TWidgetType
} from '@/lib/litegraph/src/litegraph'
import { BaseWidget, LGraphNode } from '@/lib/litegraph/src/litegraph'

import {
  createEventCapture,
  createTestSubgraph,
  createTestSubgraphNode
} from './__fixtures__/subgraphHelpers'

// Helper to create a node with a widget
function createNodeWithWidget(
  title: string,
  widgetType: TWidgetType = 'number',
  widgetValue: unknown = 42,
  slotType: ISlotType = 'number',
  tooltip?: string
) {
  const node = new LGraphNode(title)
  const input = node.addInput('value', slotType)
  node.addOutput('out', slotType)

  // @ts-expect-error Abstract class instantiation
  const widget = new BaseWidget({
    name: 'widget',
    type: widgetType,
    value: widgetValue,
    y: 0,
    options: widgetType === 'number' ? { min: 0, max: 100, step: 1 } : {},
    node,
    tooltip
  })
  node.widgets = [widget]
  input.widget = { name: widget.name }

  return { node, widget, input }
}

// Helper to connect subgraph input to node and create SubgraphNode
function setupPromotedWidget(
  subgraph: Subgraph,
  node: LGraphNode,
  slotIndex = 0
) {
  subgraph.add(node)
  subgraph.inputNode.slots[slotIndex].connect(node.inputs[slotIndex], node)
  return createTestSubgraphNode(subgraph)
}

describe.skip('SubgraphWidgetPromotion', () => {
  describe.skip('Widget Promotion Functionality', () => {
    it('should promote widgets when connecting node to subgraph input', () => {
      const subgraph = createTestSubgraph({
        inputs: [{ name: 'value', type: 'number' }]
      })

      const { node } = createNodeWithWidget('Test Node')
      const subgraphNode = setupPromotedWidget(subgraph, node)

      // The widget should be promoted to the subgraph node
      expect(subgraphNode.widgets).toHaveLength(1)
      expect(subgraphNode.widgets[0].name).toBe('value') // Uses subgraph input name
      expect(subgraphNode.widgets[0].type).toBe('number')
      expect(subgraphNode.widgets[0].value).toBe(42)
    })

    it('should promote all widget types', () => {
      const subgraph = createTestSubgraph({
        inputs: [
          { name: 'numberInput', type: 'number' },
          { name: 'stringInput', type: 'string' },
          { name: 'toggleInput', type: 'boolean' }
        ]
      })

      // Create nodes with different widget types
      const { node: numberNode } = createNodeWithWidget(
        'Number Node',
        'number',
        100
      )
      const { node: stringNode } = createNodeWithWidget(
        'String Node',
        'string',
        'test',
        'string'
      )
      const { node: toggleNode } = createNodeWithWidget(
        'Toggle Node',
        'toggle',
        true,
        'boolean'
      )

      // Setup all nodes
      subgraph.add(numberNode)
      subgraph.add(stringNode)
      subgraph.add(toggleNode)

      subgraph.inputNode.slots[0].connect(numberNode.inputs[0], numberNode)
      subgraph.inputNode.slots[1].connect(stringNode.inputs[0], stringNode)
      subgraph.inputNode.slots[2].connect(toggleNode.inputs[0], toggleNode)

      const subgraphNode = createTestSubgraphNode(subgraph)

      // All widgets should be promoted
      expect(subgraphNode.widgets).toHaveLength(3)

      // Check specific widget values
      expect(subgraphNode.widgets[0].value).toBe(100)
      expect(subgraphNode.widgets[1].value).toBe('test')
      expect(subgraphNode.widgets[2].value).toBe(true)
    })

    it('should fire widget-promoted event when widget is promoted', () => {
      const subgraph = createTestSubgraph({
        inputs: [{ name: 'input', type: 'number' }]
      })

      const eventCapture = createEventCapture(subgraph.events, [
        'widget-promoted',
        'widget-demoted'
      ])

      const { node } = createNodeWithWidget('Test Node')
      const subgraphNode = setupPromotedWidget(subgraph, node)

      // Check event was fired
      const promotedEvents = eventCapture.getEventsByType('widget-promoted')
      expect(promotedEvents).toHaveLength(1)
      expect(promotedEvents[0].detail.widget).toBeDefined()
      expect(promotedEvents[0].detail.subgraphNode).toBe(subgraphNode)

      eventCapture.cleanup()
    })

    it('should fire widget-demoted event when removing promoted widget', () => {
      const subgraph = createTestSubgraph({
        inputs: [{ name: 'input', type: 'number' }]
      })

      const { node } = createNodeWithWidget('Test Node')
      const subgraphNode = setupPromotedWidget(subgraph, node)
      expect(subgraphNode.widgets).toHaveLength(1)

      const eventCapture = createEventCapture(subgraph.events, [
        'widget-demoted'
      ])

      // Remove the widget
      subgraphNode.removeWidgetByName('input')

      // Check event was fired
      const demotedEvents = eventCapture.getEventsByType('widget-demoted')
      expect(demotedEvents).toHaveLength(1)
      expect(demotedEvents[0].detail.widget).toBeDefined()
      expect(demotedEvents[0].detail.subgraphNode).toBe(subgraphNode)

      // Widget should be removed
      expect(subgraphNode.widgets).toHaveLength(0)

      eventCapture.cleanup()
    })

    it('should handle multiple widgets on same node', () => {
      const subgraph = createTestSubgraph({
        inputs: [
          { name: 'input1', type: 'number' },
          { name: 'input2', type: 'string' }
        ]
      })

      // Create node with multiple widgets
      const multiWidgetNode = new LGraphNode('Multi Widget Node')
      const numInput = multiWidgetNode.addInput('num', 'number')
      const strInput = multiWidgetNode.addInput('str', 'string')

      // @ts-expect-error Abstract class instantiation
      const widget1 = new BaseWidget({
        name: 'widget1',
        type: 'number',
        value: 10,
        y: 0,
        options: {},
        node: multiWidgetNode
      })

      // @ts-expect-error Abstract class instantiation
      const widget2 = new BaseWidget({
        name: 'widget2',
        type: 'string',
        value: 'hello',
        y: 40,
        options: {},
        node: multiWidgetNode
      })

      multiWidgetNode.widgets = [widget1, widget2]
      numInput.widget = { name: widget1.name }
      strInput.widget = { name: widget2.name }
      subgraph.add(multiWidgetNode)

      // Connect both inputs
      subgraph.inputNode.slots[0].connect(
        multiWidgetNode.inputs[0],
        multiWidgetNode
      )
      subgraph.inputNode.slots[1].connect(
        multiWidgetNode.inputs[1],
        multiWidgetNode
      )

      // Create SubgraphNode
      const subgraphNode = createTestSubgraphNode(subgraph)

      // Both widgets should be promoted
      expect(subgraphNode.widgets).toHaveLength(2)
      expect(subgraphNode.widgets[0].name).toBe('input1')
      expect(subgraphNode.widgets[0].value).toBe(10)

      expect(subgraphNode.widgets[1].name).toBe('input2')
      expect(subgraphNode.widgets[1].value).toBe('hello')
    })

    it('should fire widget-demoted events when node is removed', () => {
      const subgraph = createTestSubgraph({
        inputs: [{ name: 'input', type: 'number' }]
      })

      const { node } = createNodeWithWidget('Test Node')
      const subgraphNode = setupPromotedWidget(subgraph, node)

      expect(subgraphNode.widgets).toHaveLength(1)

      const eventCapture = createEventCapture(subgraph.events, [
        'widget-demoted'
      ])

      // Remove the subgraph node
      subgraphNode.onRemoved()

      // Should fire demoted events for all widgets
      const demotedEvents = eventCapture.getEventsByType('widget-demoted')
      expect(demotedEvents).toHaveLength(1)

      eventCapture.cleanup()
    })

    it('should not promote widget if input is not connected', () => {
      const subgraph = createTestSubgraph({
        inputs: [{ name: 'input', type: 'number' }]
      })

      const { node } = createNodeWithWidget('Test Node')
      subgraph.add(node)

      // Don't connect - just create SubgraphNode
      const subgraphNode = createTestSubgraphNode(subgraph)

      // No widgets should be promoted
      expect(subgraphNode.widgets).toHaveLength(0)
    })

    it('should handle disconnection of promoted widget', () => {
      const subgraph = createTestSubgraph({
        inputs: [{ name: 'input', type: 'number' }]
      })

      const { node } = createNodeWithWidget('Test Node')
      const subgraphNode = setupPromotedWidget(subgraph, node)
      expect(subgraphNode.widgets).toHaveLength(1)

      // Disconnect the link
      subgraph.inputNode.slots[0].disconnect()

      // Widget should be removed (through event listeners)
      expect(subgraphNode.widgets).toHaveLength(0)
    })
  })

  describe.skip('Tooltip Promotion', () => {
    it('should preserve widget tooltip when promoting', () => {
      const subgraph = createTestSubgraph({
        inputs: [{ name: 'value', type: 'number' }]
      })

      const originalTooltip = 'This is a test tooltip'
      const { node } = createNodeWithWidget(
        'Test Node',
        'number',
        42,
        'number',
        originalTooltip
      )
      const subgraphNode = setupPromotedWidget(subgraph, node)

      // The promoted widget should preserve the original tooltip
      expect(subgraphNode.widgets).toHaveLength(1)
      expect(subgraphNode.widgets[0].tooltip).toBe(originalTooltip)
    })

    it('should handle widgets with no tooltip', () => {
      const subgraph = createTestSubgraph({
        inputs: [{ name: 'value', type: 'number' }]
      })

      const { node } = createNodeWithWidget('Test Node', 'number', 42, 'number')
      const subgraphNode = setupPromotedWidget(subgraph, node)

      // The promoted widget should have undefined tooltip
      expect(subgraphNode.widgets).toHaveLength(1)
      expect(subgraphNode.widgets[0].tooltip).toBeUndefined()
    })

    it('should preserve tooltips for multiple promoted widgets', () => {
      const subgraph = createTestSubgraph({
        inputs: [
          { name: 'input1', type: 'number' },
          { name: 'input2', type: 'string' }
        ]
      })

      // Create node with multiple widgets with different tooltips
      const multiWidgetNode = new LGraphNode('Multi Widget Node')
      const numInput = multiWidgetNode.addInput('num', 'number')
      const strInput = multiWidgetNode.addInput('str', 'string')

      // @ts-expect-error Abstract class instantiation
      const widget1 = new BaseWidget({
        name: 'widget1',
        type: 'number',
        value: 10,
        y: 0,
        options: {},
        node: multiWidgetNode,
        tooltip: 'Number widget tooltip'
      })

      // @ts-expect-error Abstract class instantiation
      const widget2 = new BaseWidget({
        name: 'widget2',
        type: 'string',
        value: 'hello',
        y: 40,
        options: {},
        node: multiWidgetNode,
        tooltip: 'String widget tooltip'
      })

      multiWidgetNode.widgets = [widget1, widget2]
      numInput.widget = { name: widget1.name }
      strInput.widget = { name: widget2.name }
      subgraph.add(multiWidgetNode)

      // Connect both inputs
      subgraph.inputNode.slots[0].connect(
        multiWidgetNode.inputs[0],
        multiWidgetNode
      )
      subgraph.inputNode.slots[1].connect(
        multiWidgetNode.inputs[1],
        multiWidgetNode
      )

      // Create SubgraphNode
      const subgraphNode = createTestSubgraphNode(subgraph)

      // Both widgets should preserve their tooltips
      expect(subgraphNode.widgets).toHaveLength(2)
      expect(subgraphNode.widgets[0].tooltip).toBe('Number widget tooltip')
      expect(subgraphNode.widgets[1].tooltip).toBe('String widget tooltip')
    })

    it('should preserve original tooltip after promotion', () => {
      const subgraph = createTestSubgraph({
        inputs: [{ name: 'value', type: 'number' }]
      })

      const originalTooltip = 'Original tooltip'
      const { node } = createNodeWithWidget(
        'Test Node',
        'number',
        42,
        'number',
        originalTooltip
      )
      const subgraphNode = setupPromotedWidget(subgraph, node)

      const promotedWidget = subgraphNode.widgets[0]

      // The promoted widget should preserve the original tooltip
      expect(promotedWidget.tooltip).toBe(originalTooltip)

      // The promoted widget should still function normally
      expect(promotedWidget.name).toBe('value') // Uses subgraph input name
      expect(promotedWidget.type).toBe('number')
      expect(promotedWidget.value).toBe(42)
    })
  })
})
