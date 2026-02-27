import { useChainCallback } from '@/composables/functional/useChainCallback'
import { LGraphNode, LiteGraph } from '@/lib/litegraph/src/litegraph'
import type {
  INodeInputSlot,
  INodeOutputSlot,
  ISlotType,
  LLink
} from '@/lib/litegraph/src/litegraph'
import { NodeSlot } from '@/lib/litegraph/src/node/NodeSlot'
import type { CanvasPointerEvent } from '@/lib/litegraph/src/types/events'
import type {
  IBaseWidget,
  TWidgetValue
} from '@/lib/litegraph/src/types/widgets'
import { assetService } from '@/platform/assets/services/assetService'
import { createAssetWidget } from '@/platform/assets/utils/createAssetWidget'
import type { ComfyNodeDef, InputSpec } from '@/schemas/nodeDefSchema'
import { app } from '@/scripts/app'
import {
  ComfyWidgets,
  addValueControlWidgets,
  isValidWidgetType
} from '@/scripts/widgets'
import { isPrimitiveNode } from '@/renderer/utils/nodeTypeGuards'
import { CONFIG, GET_CONFIG } from '@/services/litegraphService'
import { mergeInputSpec } from '@/utils/nodeDefUtil'
import { applyTextReplacements } from '@/utils/searchAndReplace'

const replacePropertyName = 'Run widget replace on values'
export class PrimitiveNode extends LGraphNode {
  controlValues?: TWidgetValue[]
  lastType?: string
  static override category: string
  constructor(title: string) {
    super(title)
    this.addOutput('connect to widget input', '*')
    this.serialize_widgets = true
    this.isVirtualNode = true

    if (!this.properties || !(replacePropertyName in this.properties)) {
      this.addProperty(replacePropertyName, false, 'boolean')
    }
  }

  override applyToGraph(extraLinks: LLink[] = []) {
    if (!this.outputs[0].links?.length || !this.graph) return

    const links = [
      ...this.outputs[0].links.map((l) => this.graph!.links[l]),
      ...extraLinks
    ]
    let v = this.widgets?.[0].value
    if (v && this.properties[replacePropertyName]) {
      v = applyTextReplacements(this.graph, v as string)
    }

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
        console.warn(
          `Unable to find widget "${widgetName}" on node [${node.id}]`
        )
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

  override refreshComboInNode() {
    const widget = this.widgets?.[0]
    if (widget?.type === 'combo') {
      // @ts-expect-error fixme ts strict error
      widget.options.values = this.outputs[0].widget[GET_CONFIG]()[0]

      // @ts-expect-error fixme ts strict error
      if (!widget.options.values.includes(widget.value as string)) {
        // @ts-expect-error fixme ts strict error
        widget.value = widget.options.values[0]
        ;(widget.callback as Function)(widget.value)
      }
    }
  }

  override onAfterGraphConfigured() {
    if (this.outputs[0].links?.length && !this.widgets?.length) {
      this._onFirstConnection()

      // Populate widget values from config data
      if (this.widgets && this.widgets_values) {
        for (let i = 0; i < this.widgets_values.length; i++) {
          const w = this.widgets[i]
          if (w) {
            w.value = this.widgets_values[i]
          }
        }
      }

      // Merge values if required
      this._mergeWidgetConfig()
    }
  }

  override onConnectionsChange(
    _type: ISlotType,
    _index: number,
    connected: boolean
  ) {
    if (app.configuringGraph) {
      // Dont run while the graph is still setting up
      return
    }

    const links = this.outputs[0].links
    if (connected) {
      if (links?.length && !this.widgets?.length) {
        this._onFirstConnection()
      }
    } else {
      // We may have removed a link that caused the constraints to change
      this._mergeWidgetConfig()

      if (!links?.length) {
        this.onLastDisconnect()
      }
    }
  }

  override onConnectOutput(
    slot: number,
    _type: string,
    input: INodeInputSlot,
    target_node: LGraphNode,
    target_slot: number
  ) {
    // Fires before the link is made allowing us to reject it if it isn't valid
    // No widget, we can't connect
    if (!input.widget && !(input.type in ComfyWidgets)) {
      return false
    }

    if (this.outputs[slot].links?.length) {
      const valid = this._isValidConnection(input)
      if (valid) {
        // On connect of additional outputs, copy our value to their widget
        this.applyToGraph([{ target_id: target_node.id, target_slot } as LLink])
      }
      return valid
    }

    return true
  }

  private _onFirstConnection(recreating?: boolean) {
    // First connection can fire before the graph is ready on initial load so random things can be missing
    if (!this.outputs[0].links || !this.graph) {
      this.onLastDisconnect()
      return
    }
    const linkId = this.outputs[0].links[0]
    const link = this.graph.links[linkId]
    if (!link) return

    const theirNode = this.graph.getNodeById(link.target_id)
    if (!theirNode || !theirNode.inputs) return

    const input = theirNode.inputs[link.target_slot]
    if (!input) return

    let widget
    if (!input.widget) {
      if (!(input.type in ComfyWidgets)) return
      widget = { name: input.name, [GET_CONFIG]: () => [input.type, {}] } //fake widget
    } else {
      widget = input.widget
    }

    // @ts-expect-error fixme ts strict error
    const config = widget[GET_CONFIG]?.()
    if (!config) return

    const { type } = getWidgetType(config)
    // Update our output to restrict to the widget type
    this.outputs[0].type = type
    this.outputs[0].name = type
    this.outputs[0].widget = widget

    this._createWidget(
      widget[CONFIG] ?? config,
      theirNode,
      widget.name,
      // @ts-expect-error fixme ts strict error
      recreating
    )
  }

  private _createWidget(
    inputData: InputSpec,
    node: LGraphNode,
    widgetName: string,
    recreating: boolean
  ) {
    let type = inputData[0]

    if (type instanceof Array) {
      type = 'COMBO'
    }

    // Store current size as addWidget resizes the node
    const [oldWidth, oldHeight] = this.size
    let widget: IBaseWidget

    if (
      type === 'COMBO' &&
      assetService.shouldUseAssetBrowser(node.comfyClass, widgetName)
    ) {
      widget = this._createAssetWidget(node, widgetName, inputData)
      const theirWidget = node.widgets?.find((w) => w.name === widgetName)
      if (theirWidget) widget.value = theirWidget.value
      this._finalizeWidget(widget, oldWidth, oldHeight, recreating)
      return
    }

    if (isValidWidgetType(type)) {
      widget = (ComfyWidgets[type](this, 'value', inputData, app) || {}).widget
    } else {
      // @ts-expect-error InputSpec is not typed correctly
      widget = this.addWidget(type, 'value', null, () => {}, {})
    }

    if (node?.widgets && widget) {
      const theirWidget = node.widgets.find((w) => w.name === widgetName)
      if (theirWidget) {
        widget.value = theirWidget.value
      }
    }

    if (
      !inputData?.[1]?.control_after_generate &&
      (widget.type === 'number' || widget.type === 'combo')
    ) {
      let control_value = this.widgets_values?.[1]
      if (!control_value) {
        control_value = 'fixed'
      }
      addValueControlWidgets(
        this,
        widget,
        control_value as string,
        undefined,
        inputData
      )
      if (this.widgets?.[1]) widget.linkedWidgets = [this.widgets[1]]

      let filter = this.widgets_values?.[2]
      if (filter && this.widgets && this.widgets.length === 3) {
        this.widgets[2].value = filter
      }
    }

    // Restore any saved control values
    const controlValues = this.controlValues
    if (
      this.widgets &&
      this.lastType === this.widgets[0]?.type &&
      controlValues?.length === this.widgets.length - 1
    ) {
      for (let i = 0; i < controlValues.length; i++) {
        this.widgets[i + 1].value = controlValues[i]
      }
    }

    this._finalizeWidget(widget, oldWidth, oldHeight, recreating)
  }

  private _createAssetWidget(
    targetNode: LGraphNode,
    targetInputName: string,
    inputData: InputSpec
  ): IBaseWidget {
    const defaultValue = inputData[1]?.default as string | undefined
    return createAssetWidget({
      node: this,
      widgetName: 'value',
      nodeTypeForBrowser: targetNode.comfyClass ?? '',
      inputNameForBrowser: targetInputName,
      defaultValue,
      onValueChange: (widget, newValue, oldValue) => {
        widget.callback?.(
          widget.value,
          app.canvas,
          this,
          app.canvas.graph_mouse,
          {} as CanvasPointerEvent
        )
        this.onWidgetChanged?.(widget.name, newValue, oldValue, widget)
      }
    })
  }

  private _finalizeWidget(
    widget: IBaseWidget,
    oldWidth: number,
    oldHeight: number,
    recreating: boolean
  ) {
    widget.callback = useChainCallback(widget.callback, () => {
      this.applyToGraph()
    })

    this.setSize([
      Math.max(this.size[0], oldWidth),
      Math.max(this.size[1], oldHeight)
    ])

    if (!recreating) {
      const sz = this.computeSize()
      if (this.size[0] < sz[0]) {
        this.size[0] = sz[0]
      }
      if (this.size[1] < sz[1]) {
        this.size[1] = sz[1]
      }

      requestAnimationFrame(() => {
        this.onResize?.(this.size)
      })
    }
  }

  recreateWidget() {
    const values = this.widgets?.map((w) => w.value)
    this._removeWidgets()
    this._onFirstConnection(true)
    if (values?.length && this.widgets) {
      for (let i = 0; i < this.widgets.length; i++)
        this.widgets[i].value = values[i]
    }
    return this.widgets?.[0]
  }

  private _mergeWidgetConfig() {
    // Merge widget configs if the node has multiple outputs
    const output = this.outputs[0]
    const links = output.links ?? []

    const hasConfig = !!output.widget?.[CONFIG]
    if (hasConfig) {
      delete output.widget?.[CONFIG]
    }

    if (links?.length < 2 && hasConfig) {
      // Copy the widget options from the source
      if (links.length) {
        this.recreateWidget()
      }

      return
    }
    const config1 = (output.widget?.[GET_CONFIG] as () => InputSpec)?.()
    if (!config1) return
    const isNumber = config1[0] === 'INT' || config1[0] === 'FLOAT'
    if (!isNumber || !this.graph) return

    for (const linkId of links) {
      const link = this.graph.links[linkId]
      if (!link) continue // Can be null when removing a node

      const theirNode = this.graph.getNodeById(link.target_id)
      if (!theirNode) continue
      const theirInput = theirNode.inputs[link.target_slot]

      // Call is valid connection so it can merge the configs when validating
      this._isValidConnection(theirInput, hasConfig)
    }
  }

  private _isValidConnection(input: INodeInputSlot, forceUpdate?: boolean) {
    // Only allow connections where the configs match
    const output = this.outputs?.[0]
    const config2 = (input.widget?.[GET_CONFIG] as () => InputSpec)?.()
    if (!config2) return false

    return !!mergeIfValid.call(
      this,
      output,
      config2,
      forceUpdate,
      this.recreateWidget
    )
  }

  private _removeWidgets() {
    if (this.widgets) {
      // Allow widgets to cleanup
      for (const w of this.widgets) {
        if (w.onRemove) {
          w.onRemove()
        }
      }

      // Temporarily store the current values in case the node is being recreated
      // e.g. by group node conversion
      this.controlValues = []
      this.lastType = this.widgets[0]?.type
      for (let i = 1; i < this.widgets.length; i++) {
        this.controlValues.push(this.widgets[i].value)
      }
      setTimeout(() => {
        delete this.lastType
        delete this.controlValues
      }, 15)
      this.widgets.length = 0
    }
  }

  onLastDisconnect() {
    // We can't remove + re-add the output here as if you drag a link over the same link
    // it removes, then re-adds, causing it to break
    this.outputs[0].type = '*'
    this.outputs[0].name = 'connect to widget input'
    delete this.outputs[0].widget

    this._removeWidgets()
  }
}

export function getWidgetConfig(
  slot: INodeInputSlot | INodeOutputSlot
): InputSpec {
  return (slot.widget?.[CONFIG] ??
    (slot.widget?.[GET_CONFIG] as () => InputSpec)?.() ?? [
      '*',
      {}
    ]) as InputSpec
}

function getConfig(this: LGraphNode, widgetName: string) {
  const { nodeData } = this.constructor
  return (
    nodeData?.input?.required?.[widgetName] ??
    nodeData?.input?.optional?.[widgetName]
  )
}

/**
 * Convert a widget to an input slot.
 * @deprecated Widget to socket conversion is no longer necessary, as they co-exist now.
 * @param node The node to convert the widget to an input slot for.
 * @param widget The widget to convert to an input slot.
 * @returns The input slot that was converted from the widget or undefined if the widget is not found.
 * @knipIgnoreUnusedButUsedByCustomNodes
 */
export function convertToInput(
  node: LGraphNode,
  widget: IBaseWidget
): INodeInputSlot | undefined {
  console.warn(
    'Please remove call to convertToInput. Widget to socket conversion is no longer necessary, as they co-exist now.'
  )
  return node.inputs.find((slot) => slot.widget?.name === widget.name)
}

function getWidgetType(config: InputSpec) {
  // Special handling for COMBO so we restrict links based on the entries
  let type = config[0]
  if (type instanceof Array) {
    type = 'COMBO'
  }
  return { type }
}

export function setWidgetConfig(slot: INodeInputSlot, config?: InputSpec) {
  if (!slot.widget) return
  if (config) {
    slot.widget[GET_CONFIG] = () => config
  } else {
    delete slot.widget
  }

  if (!(slot instanceof NodeSlot)) return
  const graph = slot.node.graph
  if (!graph) return
  const link = graph.links[slot.link ?? -1]
  if (!link) return
  const originNode = graph.getNodeById(link.origin_id)
  if (!originNode || !isPrimitiveNode(originNode)) return
  if (config) {
    originNode.recreateWidget()
  } else if (!app.configuringGraph) {
    originNode.disconnectOutput(0)
    originNode.onLastDisconnect()
  }
}

export function mergeIfValid(
  output: INodeOutputSlot | INodeInputSlot,
  config2: InputSpec,
  forceUpdate?: boolean,
  recreateWidget?: () => void,
  config1?: InputSpec
): { customConfig: InputSpec[1] } {
  if (!config1) {
    config1 = getWidgetConfig(output)
  }

  const customSpec = mergeInputSpec(config1, config2)

  if (customSpec || forceUpdate) {
    if (customSpec) {
      // @ts-expect-error fixme ts strict error
      output.widget[CONFIG] = customSpec
    }

    // @ts-expect-error fixme ts strict error
    const widget = recreateWidget?.call(this)
    // When deleting a node this can be null
    if (widget) {
      // @ts-expect-error fixme ts strict error
      const min = widget.options.min
      // @ts-expect-error fixme ts strict error
      const max = widget.options.max
      // @ts-expect-error fixme ts strict error
      if (min != null && widget.value < min) widget.value = min
      // @ts-expect-error fixme ts strict error
      if (max != null && widget.value > max) widget.value = max
      // @ts-expect-error fixme ts strict error
      widget.callback(widget.value)
    }
  }

  return { customConfig: customSpec?.[1] ?? {} }
}

app.registerExtension({
  name: 'Comfy.WidgetInputs',
  async beforeRegisterNodeDef(
    nodeType: typeof LGraphNode,
    _nodeData: ComfyNodeDef
  ) {
    // @ts-expect-error adding extra property
    nodeType.prototype.convertWidgetToInput = function (this: LGraphNode) {
      console.warn(
        'Please remove call to convertWidgetToInput. Widget to socket conversion is no longer necessary, as they co-exist now.'
      )
      return false
    }

    nodeType.prototype.onGraphConfigured = useChainCallback(
      nodeType.prototype.onGraphConfigured,
      function (this: LGraphNode) {
        if (!this.inputs) return
        this.widgets ??= []

        for (const input of this.inputs) {
          if (input.widget) {
            const name = input.widget.name
            if (!input.widget[GET_CONFIG]) {
              input.widget[GET_CONFIG] = () => getConfig.call(this, name)
            }

            const w = this.widgets?.find((w) => w.name === name)
            if (!w) {
              this.removeInput(this.inputs.findIndex((i) => i === input))
            }
          }
        }
      }
    )

    nodeType.prototype.onConfigure = useChainCallback(
      nodeType.prototype.onConfigure,
      function (this: LGraphNode) {
        if (!app.configuringGraph && this.inputs) {
          // On copy + paste of nodes, ensure that widget configs are set up
          for (const input of this.inputs) {
            if (input.widget && !input.widget[GET_CONFIG]) {
              const name = input.widget.name
              input.widget[GET_CONFIG] = () => getConfig.call(this, name)
            }
          }
        }
      }
    )

    // Double click a widget input to automatically attach a primitive
    const origOnInputDblClick = nodeType.prototype.onInputDblClick
    nodeType.prototype.onInputDblClick = function (
      this: LGraphNode,
      ...[slot, ...args]: Parameters<NonNullable<typeof origOnInputDblClick>>
    ) {
      const r = origOnInputDblClick?.apply(this, [slot, ...args])

      const input = this.inputs[slot]
      if (!input.widget) {
        // Not a widget input or already handled input
        if (
          !(input.type in ComfyWidgets) &&
          !(
            (
              input.widget?.[GET_CONFIG] as (() => InputSpec) | undefined
            )?.()?.[0] instanceof Array
          )
        ) {
          return r //also Not a ComfyWidgets input or combo (do nothing)
        }
      }

      // Create a primitive node
      const node = LiteGraph.createNode('PrimitiveNode')
      const graph = app.canvas.graph
      if (!node || !graph) return r

      graph?.add(node)

      // Calculate a position that won't directly overlap another node
      const pos: [number, number] = [
        this.pos[0] - node.size[0] - 30,
        this.pos[1]
      ]
      while (graph.getNodeOnPos(pos[0], pos[1], graph.nodes))
        pos[1] += LiteGraph.NODE_TITLE_HEIGHT

      node.pos = pos
      node.connect(0, this, slot)
      node.title = input.name

      return r
    }
  },
  registerCustomNodes() {
    LiteGraph.registerNodeType(
      'PrimitiveNode',
      Object.assign(PrimitiveNode, {
        title: 'Primitive'
      })
    )
    PrimitiveNode.category = 'utils'
  }
})
