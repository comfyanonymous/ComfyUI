import { shallowReactive } from 'vue'

import { useChainCallback } from '@/composables/functional/useChainCallback'
import type { CanvasPointerEvent } from '@/lib/litegraph/src/types/events'
import { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import { LLink } from '@/lib/litegraph/src/litegraph'
import type { ComfyNodeDef } from '@/schemas/nodeDefSchema'
import { app } from '@/scripts/app'

function applyToGraph(this: LGraphNode, extraLinks: LLink[] = []) {
  if (!this.outputs[0].links?.length || !this.graph) return

  const links = [
    ...this.outputs[0].links.map((l) => this.graph!.links[l]),
    ...extraLinks
  ]
  let v = this.widgets?.[0].value
  // For each output link copy our value over the original widget value
  for (const linkInfo of links) {
    const node = this.graph?.getNodeById(linkInfo.target_id)
    const input = node?.inputs[linkInfo.target_slot]
    if (!input) {
      console.warn('Unable to resolve node or input for link', linkInfo)
      continue
    }

    const widgetName = input.widget?.name
    if (!widgetName) {
      console.warn('Invalid widget or widget name', input.widget)
      continue
    }

    const widget = node.widgets?.find((w) => w.name === widgetName)
    if (!widget) {
      console.warn(`Unable to find widget "${widgetName}" on node [${node.id}]`)
      continue
    }

    widget.value = v
    widget.callback?.(
      widget.value,
      app.canvas,
      node,
      app.canvas.graph_mouse,
      {} as CanvasPointerEvent
    )
  }
}

function onCustomComboCreated(this: LGraphNode) {
  this.applyToGraph = applyToGraph

  const comboWidget = this.widgets![0]
  const values = shallowReactive<string[]>([])
  comboWidget.options.values = values

  const updateCombo = () => {
    values.splice(
      0,
      values.length,
      ...this.widgets!.filter(
        (w) => w.name.startsWith('option') && w.value
      ).map((w) => `${w.value}`)
    )
    if (app.configuringGraph) return
    if (values.includes(`${comboWidget.value}`)) return
    comboWidget.value = values[0] ?? ''
    comboWidget.callback?.(comboWidget.value)
  }
  comboWidget.callback = useChainCallback(comboWidget.callback, () =>
    this.applyToGraph!()
  )

  function addOption(node: LGraphNode) {
    if (!node.widgets) return
    const newCount = node.widgets.length - 1
    node.addWidget('string', `option${newCount}`, '', () => {})
    const widget = node.widgets.at(-1)
    if (!widget) return

    let value = ''
    Object.defineProperty(widget, 'value', {
      get() {
        return value
      },
      set(v) {
        value = v
        updateCombo()
        if (!node.widgets) return
        const lastWidget = node.widgets.at(-1)
        if (lastWidget === this) {
          if (v) addOption(node)
          return
        }
        if (v || node.widgets.at(-2) !== this || lastWidget?.value) return
        node.widgets.pop()
        node.computeSize(node.size)
        this.callback(v)
      }
    })
  }
  const widgets = this.widgets!
  widgets.push({
    name: 'index',
    type: 'hidden',
    get value() {
      return widgets.slice(2).findIndex((w) => w.value === comboWidget.value)
    },
    set value(_) {},
    draw: () => undefined,
    computeSize: () => [0, -4],
    options: { hidden: true },
    y: 0
  })
  addOption(this)
}

function onCustomIntCreated(this: LGraphNode) {
  const valueWidget = this.widgets?.[0]
  if (!valueWidget) return

  Object.defineProperty(valueWidget.options, 'min', {
    get: () => this.properties.min ?? -(2 ** 63),
    set: (v) => {
      this.properties.min = v
      valueWidget.callback?.(valueWidget.value)
    }
  })
  Object.defineProperty(valueWidget.options, 'max', {
    get: () => this.properties.max ?? 2 ** 63,
    set: (v) => {
      this.properties.max = v
      valueWidget.callback?.(valueWidget.value)
    }
  })
  Object.defineProperty(valueWidget.options, 'step2', {
    get: () => this.properties.step ?? 1,
    set: (v) => {
      this.properties.step = v
      valueWidget.callback?.(valueWidget.value) // for vue reactivity
    }
  })
}
const DISPLAY_WIDGET_TYPES = new Set(['gradientslider', 'slider', 'knob'])

function onCustomFloatCreated(this: LGraphNode) {
  const valueWidget = this.widgets?.[0]
  if (!valueWidget) return

  let baseType = valueWidget.type
  Object.defineProperty(valueWidget, 'type', {
    get: () => {
      const display = this.properties.display as string | undefined
      if (display && DISPLAY_WIDGET_TYPES.has(display)) return display
      return baseType
    },
    set: (v: string) => {
      baseType = v
    }
  })

  Object.defineProperty(valueWidget.options, 'gradient_stops', {
    get: () => this.properties.gradient_stops,
    set: (v) => {
      this.properties.gradient_stops = v
    }
  })
  Object.defineProperty(valueWidget.options, 'min', {
    get: () => this.properties.min ?? -Infinity,
    set: (v) => {
      this.properties.min = v
      valueWidget.callback?.(valueWidget.value)
    }
  })
  Object.defineProperty(valueWidget.options, 'max', {
    get: () => this.properties.max ?? Infinity,
    set: (v) => {
      this.properties.max = v
      valueWidget.callback?.(valueWidget.value)
    }
  })
  Object.defineProperty(valueWidget.options, 'precision', {
    get: () => this.properties.precision ?? 1,
    set: (v) => {
      this.properties.precision = v
      valueWidget.callback?.(valueWidget.value)
    }
  })
  Object.defineProperty(valueWidget.options, 'step2', {
    get: () => {
      if (this.properties.step) return this.properties.step

      const { precision } = this.properties
      return typeof precision === 'number' ? 5 * 10 ** -precision : 1
    },
    set: (v) => (this.properties.step = v)
  })
  Object.defineProperty(valueWidget.options, 'round', {
    get: () => {
      if (this.properties.round) return this.properties.round

      const { precision } = this.properties
      return typeof precision === 'number' ? 10 ** -precision : 0.1
    },
    set: (v) => {
      this.properties.round = v
      valueWidget.callback?.(valueWidget.value)
    }
  })
}

app.registerExtension({
  name: 'Comfy.CustomWidgets',
  beforeRegisterNodeDef(nodeType: typeof LGraphNode, nodeData: ComfyNodeDef) {
    if (nodeData?.name === 'CustomCombo')
      nodeType.prototype.onNodeCreated = useChainCallback(
        nodeType.prototype.onNodeCreated,
        onCustomComboCreated
      )
    else if (nodeData?.name === 'PrimitiveInt')
      nodeType.prototype.onNodeCreated = useChainCallback(
        nodeType.prototype.onNodeCreated,
        onCustomIntCreated
      )
    else if (nodeData?.name === 'PrimitiveFloat')
      nodeType.prototype.onNodeCreated = useChainCallback(
        nodeType.prototype.onNodeCreated,
        onCustomFloatCreated
      )
  }
})
