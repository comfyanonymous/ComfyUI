import { remove } from 'es-toolkit'
import { shallowReactive } from 'vue'

import { useChainCallback } from '@/composables/functional/useChainCallback'
import type {
  ISlotType,
  INodeInputSlot,
  INodeOutputSlot
} from '@/lib/litegraph/src/interfaces'
import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import { LiteGraph } from '@/lib/litegraph/src/litegraph'
import type { LLink } from '@/lib/litegraph/src/LLink'
import { commonType } from '@/lib/litegraph/src/utils/type'
import { transformInputSpecV1ToV2 } from '@/schemas/nodeDef/migration'
import type { ComboInputSpec, InputSpec } from '@/schemas/nodeDefSchema'
import type { InputSpec as InputSpecV2 } from '@/schemas/nodeDef/nodeDefSchemaV2'
import {
  zAutogrowOptions,
  zDynamicComboInputSpec
} from '@/schemas/nodeDefSchema'
import { useLitegraphService } from '@/services/litegraphService'
import { app } from '@/scripts/app'
import type { ComfyApp } from '@/scripts/app'

const INLINE_INPUTS = false

type MatchTypeNode = LGraphNode &
  Pick<Required<LGraphNode>, 'onConnectionsChange'> & {
    comfyDynamic: { matchType: Record<string, Record<string, string>> }
  }
type AutogrowNode = LGraphNode &
  Pick<Required<LGraphNode>, 'onConnectionsChange' | 'widgets'> & {
    comfyDynamic: {
      autogrow: Record<
        string,
        {
          min: number
          max: number
          inputSpecs: InputSpecV2[]
          prefix?: string
          names?: string[]
        }
      >
    }
  }

function ensureWidgetForInput(node: LGraphNode, input: INodeInputSlot) {
  node.widgets ??= []
  const { widget } = input
  if (widget && node.widgets.some((w) => w.name === widget.name)) return
  node.widgets.push({
    draw(ctx, _n, _w, y) {
      ctx.save()
      ctx.fillStyle = LiteGraph.NODE_TEXT_COLOR
      ctx.fillText(input.label ?? input.name, 20, y + 15)
      ctx.restore()
    },
    name: input.name,
    options: {},
    serialize: false,
    type: 'shim',
    y: 0
  })
  input.alwaysVisible = true
  input.widget = { name: input.name }
}

function dynamicComboWidget(
  node: LGraphNode,
  inputName: string,
  untypedInputData: InputSpec,
  appArg: ComfyApp,
  widgetName?: string
) {
  const { addNodeInput } = useLitegraphService()
  const parseResult = zDynamicComboInputSpec.safeParse(untypedInputData)
  if (!parseResult.success) throw new Error('invalid DynamicCombo spec')
  const inputData = parseResult.data
  const options = Object.fromEntries(
    inputData[1].options.map(({ key, inputs }) => [key, inputs])
  )
  const subSpec: ComboInputSpec = [Object.keys(options), {}]
  const { widget, minWidth, minHeight } = app.widgets['COMBO'](
    node,
    inputName,
    subSpec,
    appArg,
    widgetName
  )
  function isInGroup(e: { name: string }): boolean {
    return e.name.startsWith(inputName + '.')
  }
  const updateWidgets = (value?: string) => {
    if (!node.widgets) throw new Error('Not Reachable')
    const newSpec = value ? options[value] : undefined

    const removedInputs = remove(node.inputs, isInGroup)
    for (const widget of remove(node.widgets, isInGroup)) widget.onRemove?.()

    if (!newSpec) return

    const insertionPoint = node.widgets.findIndex((w) => w === widget) + 1
    const startingLength = node.widgets.length
    const startingInputLength = node.inputs.length

    if (insertionPoint === 0)
      throw new Error("Dynamic widget doesn't exist on node")
    const inputTypes: (Record<string, InputSpec> | undefined)[] = [
      newSpec.required,
      newSpec.optional
    ]
    inputTypes.forEach((inputType, idx) => {
      for (const key in inputType ?? {}) {
        const name = `${widget.name}.${key}`
        const specToAdd = transformInputSpecV1ToV2(inputType![key], {
          name,
          isOptional: idx !== 0
        })
        specToAdd.display_name = key
        addNodeInput(node, specToAdd)
        const newInputs = node.inputs
          .slice(startingInputLength)
          .filter((inp) => inp.name.startsWith(name))
        for (const newInput of newInputs) {
          if (INLINE_INPUTS && !newInput.widget)
            ensureWidgetForInput(node, newInput)
        }
      }
    })

    const inputInsertionPoint =
      node.inputs.findIndex((i) => i.name === widget.name) + 1
    const addedWidgets = node.widgets.splice(startingLength)
    node.widgets.splice(insertionPoint, 0, ...addedWidgets)
    if (inputInsertionPoint === 0) {
      if (
        addedWidgets.length === 0 &&
        node.inputs.length !== startingInputLength
      )
        //input is inputOnly, but lacks an insertion point
        throw new Error('Failed to find input socket for ' + widget.name)
      return
    }
    const addedInputs = spliceInputs(node, startingInputLength).map(
      (addedInput) => {
        const existingInput = node.inputs.findIndex(
          (existingInput) => addedInput.name === existingInput.name
        )
        return existingInput === -1
          ? addedInput
          : spliceInputs(node, existingInput, 1)[0]
      }
    )
    //assume existing inputs are in correct order
    spliceInputs(node, inputInsertionPoint, 0, ...addedInputs)

    for (const input of removedInputs) {
      const inputIndex = node.inputs.findIndex((inp) => inp.name === input.name)
      if (inputIndex === -1) {
        node.inputs.push(input)
        node.removeInput(node.inputs.length - 1)
      } else {
        node.inputs[inputIndex].link = input.link
        if (!input.link) continue
        const link = node.graph?.links?.[input.link]
        if (!link) continue
        link.target_slot = inputIndex
        node.onConnectionsChange?.(
          LiteGraph.INPUT,
          inputIndex,
          true,
          link,
          node.inputs[inputIndex]
        )
      }
    }

    node.size[1] = node.computeSize([...node.size])[1]
    if (!node.graph) return
    node._setConcreteSlots()
    node.arrange()
    app.canvas?.setDirty(true, true)
  }
  //A little hacky, but onConfigure won't work.
  //It fires too late and is overly disruptive
  let widgetValue = widget.value
  Object.defineProperty(widget, 'value', {
    get() {
      return widgetValue
    },
    set(value) {
      widgetValue = value
      updateWidgets(value)
    }
  })
  widget.value = widgetValue
  return { widget, minWidth, minHeight }
}

export const dynamicWidgets = { COMFY_DYNAMICCOMBO_V3: dynamicComboWidget }
const dynamicInputs: Record<
  string,
  (node: LGraphNode, inputSpec: InputSpecV2) => void
> = {
  COMFY_AUTOGROW_V3: applyAutogrow,
  COMFY_MATCHTYPE_V3: applyMatchType
}

export function applyDynamicInputs(
  node: LGraphNode,
  inputSpec: InputSpecV2
): boolean {
  if (!(inputSpec.type in dynamicInputs)) return false
  //TODO: move parsing/validation of inputSpec here?
  dynamicInputs[inputSpec.type](node, inputSpec)
  return true
}
function spliceInputs(
  node: LGraphNode,
  startIndex: number,
  deleteCount = -1,
  ...toAdd: INodeInputSlot[]
): INodeInputSlot[] {
  if (deleteCount < 0) return node.inputs.splice(startIndex)
  const ret = node.inputs.splice(startIndex, deleteCount, ...toAdd)
  node.inputs.slice(startIndex).forEach((input, index) => {
    const link = input.link && node.graph?.links?.get(input.link)
    if (link) link.target_slot = startIndex + index
  })
  return ret
}

function changeOutputType(
  node: LGraphNode,
  output: INodeOutputSlot,
  combinedType: ISlotType
) {
  if (output.type === combinedType) return
  output.type = combinedType

  //check and potentially remove links
  if (!node.graph) return
  for (const link_id of output.links ?? []) {
    const link = node.graph.links[link_id]
    if (!link) continue
    const { input, inputNode, subgraphOutput } = link.resolve(node.graph)
    const inputType = (input ?? subgraphOutput)?.type
    if (!inputType) continue
    const keep = LiteGraph.isValidConnection(combinedType, inputType)
    if (!keep && subgraphOutput) subgraphOutput.disconnect()
    else if (!keep && inputNode) inputNode.disconnectInput(link.target_slot)
    if (input && inputNode?.onConnectionsChange)
      inputNode.onConnectionsChange(
        LiteGraph.INPUT,
        link.target_slot,
        keep,
        link,
        input
      )
  }
}

function withComfyMatchType(node: LGraphNode): asserts node is MatchTypeNode {
  if (node.comfyDynamic?.matchType) return
  node.comfyDynamic ??= {}
  node.comfyDynamic.matchType = {}

  const outputGroups = node.constructor.nodeData?.output_matchtypes
  node.onConnectionsChange = useChainCallback(
    node.onConnectionsChange,
    function (
      this: MatchTypeNode,
      contype: ISlotType,
      slot: number,
      iscon: boolean,
      linf: LLink | null | undefined
    ) {
      const input = this.inputs[slot]
      if (contype !== LiteGraph.INPUT || !this.graph || !input) return
      if (app.configuringGraph) return
      const [matchKey, matchGroup] = Object.entries(
        this.comfyDynamic.matchType
      ).find(([, group]) => input.name in group) ?? ['', undefined]
      if (!matchGroup) return
      if (iscon && linf) {
        const { output, subgraphInput } = linf.resolve(this.graph)
        const connectingType = (output ?? subgraphInput)?.type
        if (connectingType) linf.type = connectingType
      }
      //NOTE: inputs contains input
      const groupInputs: INodeInputSlot[] = node.inputs.filter(
        (inp) => inp.name in matchGroup
      )
      const connectedTypes = groupInputs.map((inp) => {
        if (!inp.link) return '*'
        const link = this.graph!.links[inp.link]
        if (!link) return '*'
        const { output, subgraphInput } = link.resolve(this.graph!)
        return (output ?? subgraphInput)?.type ?? '*'
      })
      //An input slot can accept a connection that is
      // - Compatible with original type
      // - Compatible with all other input types
      //An output slot can output
      // - Only what every input can output
      groupInputs.forEach((input, idx) => {
        const otherConnected = [
          ...connectedTypes.slice(0, idx),
          ...connectedTypes.slice(idx + 1)
        ]
        const combinedType = commonType(
          ...otherConnected,
          matchGroup[input.name]
        )
        if (!combinedType) throw new Error('invalid connection')
        input.type = combinedType
      })
      const outputType = commonType(...connectedTypes)
      if (!outputType) throw new Error('invalid connection')
      this.outputs.forEach((output, idx) => {
        if (!(outputGroups?.[idx] == matchKey)) return
        changeOutputType(this, output, outputType)
      })
      app.canvas?.setDirty(true, true)
    }
  )
}

function applyMatchType(node: LGraphNode, inputSpec: InputSpecV2) {
  const { addNodeInput } = useLitegraphService()
  const name = inputSpec.name
  const { allowed_types, template_id } = (
    inputSpec as InputSpecV2 & {
      template: { allowed_types: string; template_id: string }
    }
  ).template
  const typedSpec = { ...inputSpec, type: allowed_types }
  addNodeInput(node, typedSpec)
  withComfyMatchType(node)
  node.comfyDynamic.matchType[template_id] ??= {}
  node.comfyDynamic.matchType[template_id][name] = allowed_types

  //TODO: instead apply on output add?
  //ensure outputs get updated
  const index = node.inputs.length - 1
  requestAnimationFrame(() => {
    const input = node.inputs[index]
    if (!input) return
    node.inputs[index] = shallowReactive(input)
    node.onConnectionsChange?.(
      LiteGraph.INPUT,
      index,
      !!input.link,
      input.link ? node.graph?.links?.[input.link] : undefined,
      input
    )
  })
}

function autogrowOrdinalToName(
  ordinal: number,
  key: string,
  groupName: string,
  node: AutogrowNode
) {
  const {
    names,
    prefix = '',
    inputSpecs
  } = node.comfyDynamic.autogrow[groupName]
  const baseName = names
    ? names[ordinal]
    : (inputSpecs.length == 1 ? prefix : key) + ordinal
  return { name: `${groupName}.${baseName}`, display_name: baseName }
}

function addAutogrowGroup(
  ordinal: number,
  groupName: string,
  node: AutogrowNode
) {
  const { addNodeInput } = useLitegraphService()
  const { max, min, inputSpecs } = node.comfyDynamic.autogrow[groupName]
  if (ordinal >= max) return

  const namedSpecs = inputSpecs.map((input) => ({
    ...input,
    isOptional: ordinal >= (min ?? 0) || input.isOptional,
    ...autogrowOrdinalToName(ordinal, input.name, groupName, node)
  }))

  const newInputs = namedSpecs.map((namedSpec) => {
    addNodeInput(node, namedSpec)
    const input = spliceInputs(node, node.inputs.length - 1, 1)[0]
    if (inputSpecs.length !== 1 || (INLINE_INPUTS && !input.widget))
      ensureWidgetForInput(node, input)
    return input
  })

  for (const newInput of newInputs) {
    for (const existingInput of remove(
      node.inputs,
      (inp) => inp.name === newInput.name
    )) {
      //NOTE: link.target_slot is updated on spliceInputs call
      newInput.link ??= existingInput.link
    }
  }

  const targetName = autogrowOrdinalToName(
    ordinal - 1,
    inputSpecs.at(-1)!.name,
    groupName,
    node
  ).name
  const lastIndex = node.inputs.findLastIndex((inp) =>
    inp.name.startsWith(targetName)
  )
  const insertionIndex = lastIndex === -1 ? node.inputs.length : lastIndex + 1
  spliceInputs(node, insertionIndex, 0, ...newInputs)
  app.canvas?.setDirty(true, true)
}

const ORDINAL_REGEX = /\d+$/
function resolveAutogrowOrdinal(
  inputName: string,
  groupName: string,
  node: AutogrowNode
): number | undefined {
  //TODO preslice groupname?
  const name = inputName.slice(groupName.length + 1)
  const { names } = node.comfyDynamic.autogrow[groupName]
  if (names) {
    const ordinal = names.findIndex((s) => s === name)
    return ordinal === -1 ? undefined : ordinal
  }
  const match = name.match(ORDINAL_REGEX)
  if (!match) return undefined
  const ordinal = parseInt(match[0])
  return ordinal !== ordinal ? undefined : ordinal
}
function autogrowInputConnected(index: number, node: AutogrowNode) {
  const input = node.inputs[index]
  const groupName = input.name.slice(0, input.name.lastIndexOf('.'))
  const lastInput = node.inputs.findLast((inp) =>
    inp.name.startsWith(groupName + '.')
  )
  const ordinal = resolveAutogrowOrdinal(input.name, groupName, node)
  if (
    !lastInput ||
    ordinal == undefined ||
    (ordinal !== resolveAutogrowOrdinal(lastInput.name, groupName, node) &&
      !app.configuringGraph)
  )
    return
  addAutogrowGroup(ordinal + 1, groupName, node)
}
function autogrowInputDisconnected(index: number, node: AutogrowNode) {
  const input = node.inputs[index]
  if (!input) return
  const groupName = input.name.slice(0, input.name.lastIndexOf('.'))
  const { min = 1, inputSpecs } = node.comfyDynamic.autogrow[groupName]
  const ordinal = resolveAutogrowOrdinal(input.name, groupName, node)
  if (ordinal == undefined || ordinal + 1 < min) return

  //resolve all inputs in group
  const groupInputs = node.inputs.filter(
    (inp) =>
      inp.name.startsWith(groupName + '.') &&
      inp.name.lastIndexOf('.') === groupName.length
  )
  const stride = inputSpecs.length
  if (stride + index >= node.inputs.length) return
  if (groupInputs.length % stride !== 0) {
    console.error('Failed to group multi-input autogrow inputs')
    return
  }
  app.canvas?.setDirty(true, true)
  //groupBy would be nice here, but may not be supported
  for (let column = 0; column < stride; column++) {
    for (
      let bubbleOrdinal = ordinal * stride + column;
      bubbleOrdinal + stride < groupInputs.length;
      bubbleOrdinal += stride
    ) {
      const curInput = groupInputs[bubbleOrdinal]
      curInput.link = groupInputs[bubbleOrdinal + stride].link
      if (!curInput.link) continue
      const link = node.graph?.links[curInput.link]
      if (!link) continue
      const curIndex = node.inputs.findIndex((inp) => inp === curInput)
      if (curIndex === -1) throw new Error('missing input')
      link.target_slot = curIndex
      node.onConnectionsChange?.(
        LiteGraph.INPUT,
        curIndex,
        true,
        link,
        curInput
      )
    }
    const lastInput = groupInputs.at(column - stride)
    if (!lastInput) continue
    lastInput.link = null
    node.onConnectionsChange?.(
      LiteGraph.INPUT,
      node.inputs.length + column - stride,
      false,
      null,
      lastInput
    )
  }
  const removalChecks = groupInputs.slice((min - 1) * stride)
  let i
  for (i = removalChecks.length - stride; i >= 0; i -= stride) {
    if (removalChecks.slice(i, i + stride).some((inp) => inp.link)) break
  }
  const toRemove = removalChecks.slice(i + stride * 2)
  remove(node.inputs, (inp) => toRemove.includes(inp))
  for (const input of toRemove) {
    const widgetName = input?.widget?.name
    if (!widgetName) continue
    for (const widget of remove(node.widgets, (w) => w.name === widgetName))
      widget.onRemove?.()
  }
  node.size[1] = node.computeSize([...node.size])[1]
}

function withComfyAutogrow(node: LGraphNode): asserts node is AutogrowNode {
  if (node.comfyDynamic?.autogrow) return
  node.comfyDynamic ??= {}
  node.comfyDynamic.autogrow = {}

  let pendingConnection: number | undefined
  let swappingConnection = false

  const originalOnConnectInput = node.onConnectInput
  node.onConnectInput = function (slot: number, ...args) {
    pendingConnection = slot
    requestAnimationFrame(() => (pendingConnection = undefined))
    return originalOnConnectInput?.apply(this, [slot, ...args]) ?? true
  }

  node.onConnectionsChange = useChainCallback(
    node.onConnectionsChange,
    function (
      this: AutogrowNode,
      contype: ISlotType,
      slot: number,
      iscon: boolean,
      linf: LLink | null | undefined
    ) {
      const input = this.inputs[slot]
      if (contype !== LiteGraph.INPUT || !input) return
      //Return if input isn't known autogrow
      const key = input.name.slice(0, input.name.lastIndexOf('.'))
      const autogrowGroup = this.comfyDynamic.autogrow[key]
      if (!autogrowGroup) return
      if (app.configuringGraph && input.widget)
        ensureWidgetForInput(node, input)
      if (iscon) {
        if (swappingConnection || !linf) return
        autogrowInputConnected(slot, this)
      } else {
        if (pendingConnection === slot) {
          swappingConnection = true
          requestAnimationFrame(() => (swappingConnection = false))
          return
        }
        requestAnimationFrame(() => autogrowInputDisconnected(slot, this))
      }
    }
  )
}
function applyAutogrow(node: LGraphNode, inputSpecV2: InputSpecV2) {
  withComfyAutogrow(node)

  const parseResult = zAutogrowOptions.safeParse(inputSpecV2)
  if (!parseResult.success) throw new Error('invalid Autogrow spec')
  const inputSpec = parseResult.data
  const { input, min = 1, names, prefix, max = 100 } = inputSpec.template

  const inputTypes: (Record<string, InputSpec> | undefined)[] = [
    input.required,
    input.optional
  ]
  const inputsV2 = inputTypes.flatMap((inputType, index) =>
    Object.entries(inputType ?? {}).map(([name, v]) =>
      transformInputSpecV1ToV2(v, { name, isOptional: index === 1 })
    )
  )
  node.comfyDynamic.autogrow[inputSpecV2.name] = {
    names,
    min,
    max: names?.length ?? max,
    prefix,
    inputSpecs: inputsV2
  }
  for (let i = 0; i === 0 || i < min; i++)
    addAutogrowGroup(i, inputSpecV2.name, node)
}
