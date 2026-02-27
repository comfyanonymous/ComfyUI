import { PREFIX, SEPARATOR } from '@/constants/groupNodeConstants'
import { t } from '@/i18n'
import type { GroupNodeWorkflowData } from '@/lib/litegraph/src/LGraph'
import type { SerialisedLLinkArray } from '@/lib/litegraph/src/LLink'
import type { NodeId } from '@/lib/litegraph/src/LGraphNode'
import type { IContextMenuValue } from '@/lib/litegraph/src/interfaces'
import {
  type ExecutableLGraphNode,
  type ExecutionId,
  LGraphNode,
  type LGraphNodeConstructor,
  LiteGraph,
  SubgraphNode
} from '@/lib/litegraph/src/litegraph'
import { useToastStore } from '@/platform/updates/common/toastStore'
import {
  type ComfyNode,
  type ComfyWorkflowJSON
} from '@/platform/workflow/validation/schemas/workflowSchema'
import type { ComfyNodeDef, InputSpec } from '@/schemas/nodeDefSchema'
import { useDialogService } from '@/services/dialogService'
import { useExecutionStore } from '@/stores/executionStore'
import { useNodeDefStore } from '@/stores/nodeDefStore'
import { useWidgetStore } from '@/stores/widgetStore'
import { type ComfyExtension } from '@/types/comfy'
import { ExecutableGroupNodeChildDTO } from '@/utils/executableGroupNodeChildDTO'
import { GROUP } from '@/utils/executableGroupNodeDto'
import { deserialiseAndCreate, serialise } from '@/utils/vintageClipboard'

import { api } from '../../scripts/api'
import { app } from '../../scripts/app'
import { ManageGroupDialog } from './groupNodeManage'
import { mergeIfValid } from './widgetInputs'

type GroupNodeLink = SerialisedLLinkArray
type LinksFromMap = Record<number, Record<number, GroupNodeLink[]>>
type LinksToMap = Record<number, Record<number, GroupNodeLink>>
type ExternalFromMap = Record<number, Record<number, string | number>>

interface GroupNodeInput {
  name?: string
  type?: string
  label?: string
  widget?: { name: string }
}

interface GroupNodeOutput {
  name?: string
  type?: string
  label?: string
  widget?: { name: string }
  links?: number[]
}

interface GroupNodeData extends Omit<
  GroupNodeWorkflowData['nodes'][number],
  'inputs' | 'outputs'
> {
  title?: string
  widgets_values?: unknown[]
  inputs?: GroupNodeInput[]
  outputs?: GroupNodeOutput[]
}

interface GroupNodeDef {
  input: {
    required: Record<string, unknown>
    optional?: Record<string, unknown>
  }
  output: unknown[]
  output_name: string[]
  output_is_list: boolean[]
}

interface NodeConfigEntry {
  input?: Record<string, { name?: string; visible?: boolean }>
  output?: Record<number, { name?: string; visible?: boolean }>
}

interface SerializedGroupConfig {
  nodes: unknown[]
  links: GroupNodeLink[]
  external?: (number | string)[][]
}

const Workflow = {
  InUse: {
    Free: 0,
    Registered: 1,
    InWorkflow: 2
  },
  isInUseGroupNode(name: string) {
    const id = `${PREFIX}${SEPARATOR}${name}`
    // Check if lready registered/in use in this workflow
    if (app.rootGraph.extra?.groupNodes?.[name]) {
      if (app.rootGraph.nodes.find((n) => n.type === id)) {
        return Workflow.InUse.InWorkflow
      } else {
        return Workflow.InUse.Registered
      }
    }
    return Workflow.InUse.Free
  },
  storeGroupNode(name: string, data: GroupNodeWorkflowData) {
    let extra = app.rootGraph.extra
    if (!extra) app.rootGraph.extra = extra = {}
    let groupNodes = extra.groupNodes
    if (!groupNodes) extra.groupNodes = groupNodes = {}
    groupNodes[name] = data
  }
}

class GroupNodeBuilder {
  nodes: LGraphNode[]
  nodeData!: GroupNodeWorkflowData

  constructor(nodes: LGraphNode[]) {
    this.nodes = nodes
  }

  async build() {
    const name = await this.getName()
    if (!name) return

    // Sort the nodes so they are in execution order
    // this allows for widgets to be in the correct order when reconstructing
    this.sortNodes()

    this.nodeData = this.getNodeData()
    Workflow.storeGroupNode(name, this.nodeData)

    return { name, nodeData: this.nodeData }
  }

  async getName() {
    const name = await useDialogService().prompt({
      title: t('groupNode.create'),
      message: t('groupNode.enterName'),
      defaultValue: ''
    })
    if (!name) return
    const used = Workflow.isInUseGroupNode(name)
    switch (used) {
      case Workflow.InUse.InWorkflow:
        useToastStore().addAlert(
          'An in use group node with this name already exists embedded in this workflow, please remove any instances or use a new name.'
        )
        return
      case Workflow.InUse.Registered:
        if (
          !confirm(
            'A group node with this name already exists embedded in this workflow, are you sure you want to overwrite it?'
          )
        ) {
          return
        }
        break
    }
    return name
  }

  sortNodes() {
    // Gets the builders nodes in graph execution order
    const nodesInOrder = app.rootGraph.computeExecutionOrder(false)
    this.nodes = this.nodes
      .map((node) => ({ index: nodesInOrder.indexOf(node), node }))
      .sort(
        (a, b) =>
          a.index - b.index ||
          String(a.node.id).localeCompare(String(b.node.id))
      )
      .map(({ node }) => node)
  }

  getNodeData(): GroupNodeWorkflowData {
    const storeLinkTypes = (config: SerializedGroupConfig) => {
      // Store link types for dynamically typed nodes e.g. reroutes
      for (const link of config.links) {
        const origin = app.rootGraph.getNodeById(link[4] as NodeId)
        const type = origin?.outputs?.[Number(link[1])]?.type
        if (type !== undefined) link.push(type)
      }
    }

    const storeExternalLinks = (config: SerializedGroupConfig) => {
      // Store any external links to the group in the config so when rebuilding we add extra slots
      config.external = []
      for (let i = 0; i < this.nodes.length; i++) {
        const node = this.nodes[i]
        if (!node.outputs?.length) continue
        for (let slot = 0; slot < node.outputs.length; slot++) {
          let hasExternal = false
          const output = node.outputs[slot]
          let type = output.type
          if (!output.links?.length) continue
          for (const l of output.links) {
            const link = app.rootGraph.links[l]
            if (!link) continue
            if (type === '*') type = link.type

            if (!app.canvas.selected_nodes[link.target_id]) {
              hasExternal = true
              break
            }
          }
          if (hasExternal) {
            config.external.push([i, slot, String(type)])
          }
        }
      }
    }

    // Use the built in copyToClipboard function to generate the node data we need
    const graph = app.canvas?.graph
    if (!graph) return { nodes: [], links: [], external: [] }
    const serialised = serialise(this.nodes, graph)
    const config = JSON.parse(serialised) as SerializedGroupConfig
    config.external = []

    storeLinkTypes(config)
    storeExternalLinks(config)

    return config as GroupNodeWorkflowData
  }
}

export class GroupNodeConfig {
  name: string
  nodeData: GroupNodeWorkflowData
  inputCount: number
  oldToNewOutputMap: Record<number, Record<number, number>>
  newToOldOutputMap: Record<number, { node: GroupNodeData; slot: number }>
  oldToNewInputMap: Record<number, Record<string, number>>
  oldToNewWidgetMap: Record<number, Record<string, string | null>>
  newToOldWidgetMap: Record<string, { node: GroupNodeData; inputName: string }>
  primitiveDefs: Record<number, GroupNodeDef>
  widgetToPrimitive: Record<number, Record<string, number | number[]>>
  primitiveToWidget: Record<
    number,
    { nodeId: number | string | null; inputName: string }[]
  >
  nodeInputs: Record<number, Record<string, string>>
  outputVisibility: boolean[]
  nodeDef: (ComfyNodeDef & { [GROUP]: GroupNodeConfig }) | undefined
  inputs!: unknown[]
  linksFrom!: LinksFromMap
  linksTo!: LinksToMap
  externalFrom!: ExternalFromMap

  constructor(name: string, nodeData: GroupNodeWorkflowData) {
    this.name = name
    this.nodeData = nodeData
    this.getLinks()

    this.inputCount = 0
    this.oldToNewOutputMap = {}
    this.newToOldOutputMap = {}
    this.oldToNewInputMap = {}
    this.oldToNewWidgetMap = {}
    this.newToOldWidgetMap = {}
    this.primitiveDefs = {}
    this.widgetToPrimitive = {}
    this.primitiveToWidget = {}
    this.nodeInputs = {}
    this.outputVisibility = []
  }

  async registerType(source = PREFIX) {
    this.nodeDef = {
      output: [],
      output_name: [],
      output_is_list: [],
      output_node: false, // This is a lie (to satisfy the interface)
      name: source + SEPARATOR + this.name,
      display_name: this.name,
      category: 'group nodes' + (SEPARATOR + source),
      input: { required: {} },
      description: `Group node combining ${this.nodeData.nodes
        .map((n) => n.type)
        .join(', ')}`,
      python_module: 'custom_nodes.' + this.name,

      [GROUP]: this
    }

    this.inputs = []
    const seenInputs = {}
    const seenOutputs = {}
    for (let i = 0; i < this.nodeData.nodes.length; i++) {
      const node = this.nodeData.nodes[i] as GroupNodeData
      node.index = i
      this.processNode(node, seenInputs, seenOutputs)
    }

    for (const p of this._convertedToProcess) {
      p()
    }
    this._convertedToProcess = []
    if (!this.nodeDef) return
    await app.registerNodeDef(`${PREFIX}${SEPARATOR}` + this.name, this.nodeDef)
    useNodeDefStore().addNodeDef(this.nodeDef)
  }

  getLinks() {
    this.linksFrom = {}
    this.linksTo = {}
    this.externalFrom = {}

    // Extract links for easy lookup
    for (const link of this.nodeData.links) {
      const [sourceNodeId, sourceNodeSlot, targetNodeId, targetNodeSlot] = link

      // Skip links outside the copy config
      if (
        sourceNodeId == null ||
        sourceNodeSlot == null ||
        targetNodeId == null ||
        targetNodeSlot == null
      )
        continue

      const srcId = Number(sourceNodeId)
      const srcSlot = Number(sourceNodeSlot)
      const tgtId = Number(targetNodeId)
      const tgtSlot = Number(targetNodeSlot)

      if (!this.linksFrom[srcId]) {
        this.linksFrom[srcId] = {}
      }
      if (!this.linksFrom[srcId][srcSlot]) {
        this.linksFrom[srcId][srcSlot] = []
      }
      this.linksFrom[srcId][srcSlot].push(link)

      if (!this.linksTo[tgtId]) {
        this.linksTo[tgtId] = {}
      }
      this.linksTo[tgtId][tgtSlot] = link
    }

    if (this.nodeData.external) {
      for (const ext of this.nodeData.external) {
        const nodeIdx = Number(ext[0])
        const slotIdx = Number(ext[1])
        const typeVal = ext[2]
        if (typeVal == null) continue
        if (!this.externalFrom[nodeIdx]) {
          this.externalFrom[nodeIdx] = { [slotIdx]: typeVal }
        } else {
          this.externalFrom[nodeIdx][slotIdx] = typeVal
        }
      }
    }
  }

  processNode(
    node: GroupNodeData,
    seenInputs: Record<string, number>,
    seenOutputs: Record<string, number>
  ) {
    const def = this.getNodeDef(node)
    if (!def) return

    const inputs = { ...def.input?.required, ...def.input?.optional }

    this.inputs.push(this.processNodeInputs(node, seenInputs, inputs))
    if (def.output?.length) this.processNodeOutputs(node, seenOutputs, def)
  }

  getNodeDef(
    node: GroupNodeData | GroupNodeWorkflowData['nodes'][number]
  ): GroupNodeDef | ComfyNodeDef | null | undefined {
    if (node.type) {
      const def = globalDefs[node.type]
      if (def) return def
    }

    const nodeIndex = node.index
    if (nodeIndex == null) return undefined

    const linksFrom = this.linksFrom[nodeIndex]
    if (node.type === 'PrimitiveNode') {
      // Skip as its not linked
      if (!linksFrom) return

      let type: string | number | null = linksFrom[0]?.[0]?.[5] ?? null
      if (type === 'COMBO') {
        // Use the array items
        const output = node.outputs?.[0] as GroupNodeOutput | undefined
        const source = output?.widget?.name
        const nodeIdx = linksFrom[0]?.[0]?.[2]
        if (source && nodeIdx != null) {
          const fromTypeName = this.nodeData.nodes[Number(nodeIdx)]?.type
          if (fromTypeName) {
            const fromType = globalDefs[fromTypeName]
            const input =
              fromType?.input?.required?.[source] ??
              fromType?.input?.optional?.[source]
            const inputType = input?.[0]
            type =
              typeof inputType === 'string' || typeof inputType === 'number'
                ? inputType
                : null
          }
        }
      }

      const def = (this.primitiveDefs[nodeIndex] = {
        input: {
          required: {
            value: [type, {}]
          }
        },
        output: [type],
        output_name: [],
        output_is_list: []
      })
      return def
    } else if (node.type === 'Reroute') {
      const linksTo = this.linksTo[nodeIndex]
      if (linksTo && linksFrom && !this.externalFrom[nodeIndex]?.[0]) {
        // Being used internally
        return null
      }

      let config: Record<string, unknown> = {}
      let rerouteType = '*'
      if (linksFrom) {
        const links = linksFrom[0] ?? []
        for (const link of links) {
          const id = link[2]
          const slot = link[3]
          if (id == null || slot == null) continue
          const targetNode = this.nodeData.nodes[Number(id)]
          const input = targetNode?.inputs?.[Number(slot)] as
            | GroupNodeInput
            | undefined
          if (input?.type && rerouteType === '*') {
            rerouteType = input.type
          }
          if (input?.widget && targetNode?.type) {
            const targetDef = globalDefs[targetNode.type]
            const targetWidget =
              targetDef?.input?.required?.[input.widget.name] ??
              targetDef?.input?.optional?.[input.widget.name]

            if (targetWidget) {
              const widgetSpec = [targetWidget[0], config] as Parameters<
                typeof mergeIfValid
              >[4]
              const res = mergeIfValid(
                { widget: widgetSpec } as unknown as Parameters<
                  typeof mergeIfValid
                >[0],
                targetWidget,
                false,
                undefined,
                widgetSpec
              )
              config = (res?.customConfig as Record<string, unknown>) ?? config
            }
          }
        }
      } else if (linksTo) {
        const link = linksTo[0]
        if (link) {
          const id = link[0]
          const slot = link[1]
          if (id != null && slot != null) {
            const outputType =
              this.nodeData.nodes[Number(id)]?.outputs?.[Number(slot)]
            if (
              outputType &&
              typeof outputType === 'object' &&
              'type' in outputType
            ) {
              rerouteType = String((outputType as GroupNodeOutput).type ?? '*')
            }
          }
        }
      } else {
        // Reroute used as a pipe
        for (const l of this.nodeData.links) {
          if (l[2] === node.index) {
            const linkType = l[5]
            if (linkType != null) rerouteType = String(linkType)
            break
          }
        }
        if (rerouteType === '*') {
          // Check for an external link
          const t = this.externalFrom[nodeIndex]?.[0]
          if (t) {
            rerouteType = String(t)
          }
        }
      }

      config.forceInput = true
      return {
        input: {
          required: {
            [rerouteType]: [rerouteType, config]
          }
        },
        output: [rerouteType],
        output_name: [],
        output_is_list: []
      }
    }

    console.warn(
      'Skipping virtual node ' +
        node.type +
        ' when building group node ' +
        this.name
    )
  }

  getInputConfig(
    node: GroupNodeData,
    inputName: string,
    seenInputs: Record<string, number>,
    config: unknown[],
    extra?: Record<string, unknown>
  ) {
    const nodeConfig = this.nodeData.config?.[node.index ?? -1] as
      | NodeConfigEntry
      | undefined
    const customConfig = nodeConfig?.input?.[inputName]
    let name =
      customConfig?.name ??
      node.inputs?.find((inp) => inp.name === inputName)?.label ??
      inputName
    let key = name
    let prefix = ''
    // Special handling for primitive to include the title if it is set rather than just "value"
    if ((node.type === 'PrimitiveNode' && node.title) || name in seenInputs) {
      prefix = `${node.title ?? node.type} `
      key = name = `${prefix}${inputName}`
      if (name in seenInputs) {
        name = `${prefix}${seenInputs[name]} ${inputName}`
      }
    }
    seenInputs[key] = (seenInputs[key] ?? 1) + 1

    if (inputName === 'seed' || inputName === 'noise_seed') {
      if (!extra) extra = {}
      extra.control_after_generate = `${prefix}control_after_generate`
    }
    if (config[0] === 'IMAGEUPLOAD') {
      if (!extra) extra = {}
      const nodeIndex = node.index ?? -1
      const configOptions =
        typeof config[1] === 'object' && config[1] !== null ? config[1] : {}
      const widgetKey =
        'widget' in configOptions && typeof configOptions.widget === 'string'
          ? configOptions.widget
          : 'image'
      extra.widget = this.oldToNewWidgetMap[nodeIndex]?.[widgetKey] ?? 'image'
    }

    if (extra) {
      const configObj =
        typeof config[1] === 'object' && config[1] ? config[1] : {}
      config = [config[0], { ...configObj, ...extra }]
    }

    return { name, config, customConfig }
  }

  processWidgetInputs(
    inputs: Record<string, unknown>,
    node: GroupNodeData,
    inputNames: string[],
    seenInputs: Record<string, number>
  ) {
    const slots: string[] = []
    const converted = new Map<number, string>()
    const nodeIndex = node.index ?? -1
    const widgetMap: Record<string, string | null> = (this.oldToNewWidgetMap[
      nodeIndex
    ] = {})
    for (const inputName of inputNames) {
      const inputSpec = inputs[inputName]
      const isValidSpec =
        Array.isArray(inputSpec) &&
        inputSpec.length >= 1 &&
        typeof inputSpec[0] === 'string'
      if (
        isValidSpec &&
        useWidgetStore().inputIsWidget(inputSpec as InputSpec)
      ) {
        const convertedIndex =
          node.inputs?.findIndex(
            (inp) => inp.name === inputName && inp.widget?.name === inputName
          ) ?? -1
        if (convertedIndex > -1) {
          // This widget has been converted to a widget
          // We need to store this in the correct position so link ids line up
          converted.set(convertedIndex, inputName)
          widgetMap[inputName] = null
        } else {
          // Normal widget
          const { name, config } = this.getInputConfig(
            node,
            inputName,
            seenInputs,
            inputs[inputName] as unknown[]
          )
          if (this.nodeDef?.input?.required) {
            // @ts-expect-error legacy dynamic input assignment
            this.nodeDef.input.required[name] = config
          }
          widgetMap[inputName] = name
          this.newToOldWidgetMap[name] = { node, inputName }
        }
      } else {
        // Normal input
        slots.push(inputName)
      }
    }
    return { converted, slots }
  }

  checkPrimitiveConnection(
    link: GroupNodeLink,
    inputName: string,
    inputs: Record<string, unknown[]>
  ) {
    const linkSourceIdx = link[0]
    if (linkSourceIdx == null) return
    const sourceNode = this.nodeData.nodes[Number(linkSourceIdx)]
    if (sourceNode?.type === 'PrimitiveNode') {
      // Merge link configurations
      const sourceNodeId = Number(link[0])
      const targetNodeId = Number(link[2])
      const primitiveDef = this.primitiveDefs[sourceNodeId]
      if (!primitiveDef) return
      const targetWidget = inputs[inputName]
      const primitiveConfig = primitiveDef.input.required.value as [
        unknown,
        Record<string, unknown>
      ]
      const output = { widget: primitiveConfig }
      const config = mergeIfValid(
        // @ts-expect-error slot type mismatch - legacy API
        output,
        targetWidget,
        false,
        undefined,
        primitiveConfig
      )
      const inputConfig = inputs[inputName]?.[1]
      primitiveConfig[1] =
        (config?.customConfig ?? inputConfig)
          ? { ...(typeof inputConfig === 'object' ? inputConfig : {}) }
          : {}

      const widgetName = this.oldToNewWidgetMap[sourceNodeId]?.['value']
      if (widgetName) {
        const name = widgetName.substring(0, widgetName.length - 6)
        primitiveConfig[1].control_after_generate = true
        primitiveConfig[1].control_prefix = name
      }

      let toPrimitive = this.widgetToPrimitive[targetNodeId]
      if (!toPrimitive) {
        toPrimitive = this.widgetToPrimitive[targetNodeId] = {}
      }
      const existing = toPrimitive[inputName]
      if (Array.isArray(existing)) {
        existing.push(sourceNodeId)
      } else if (typeof existing === 'number') {
        toPrimitive[inputName] = [existing, sourceNodeId]
      } else {
        toPrimitive[inputName] = sourceNodeId
      }

      let toWidget = this.primitiveToWidget[sourceNodeId]
      if (!toWidget) {
        toWidget = this.primitiveToWidget[sourceNodeId] = []
      }
      toWidget.push({ nodeId: targetNodeId, inputName })
    }
  }

  processInputSlots(
    inputs: Record<string, unknown[]>,
    node: GroupNodeData,
    slots: string[],
    linksTo: Record<number, GroupNodeLink>,
    inputMap: Record<number, number>,
    seenInputs: Record<string, number>
  ) {
    const nodeIdx = node.index ?? -1
    this.nodeInputs[nodeIdx] = {}
    for (let i = 0; i < slots.length; i++) {
      const inputName = slots[i]
      if (linksTo[i]) {
        this.checkPrimitiveConnection(linksTo[i], inputName, inputs)
        // This input is linked so we can skip it
        continue
      }

      const { name, config, customConfig } = this.getInputConfig(
        node,
        inputName,
        seenInputs,
        inputs[inputName]
      )

      this.nodeInputs[nodeIdx][inputName] = name
      if (customConfig?.visible === false) continue

      if (this.nodeDef?.input?.required) {
        // @ts-expect-error legacy dynamic input assignment
        this.nodeDef.input.required[name] = config
      }
      inputMap[i] = this.inputCount++
    }
  }

  processConvertedWidgets(
    inputs: Record<string, unknown>,
    node: GroupNodeData,
    slots: string[],
    converted: Map<number, string>,
    linksTo: Record<number, GroupNodeLink>,
    inputMap: Record<number, number>,
    seenInputs: Record<string, number>
  ) {
    // Add converted widgets sorted into their index order (ordered as they were converted) so link ids match up
    const convertedSlots = [...converted.keys()]
      .sort()
      .map((k) => converted.get(k))
    for (let i = 0; i < convertedSlots.length; i++) {
      const inputName = convertedSlots[i]
      if (!inputName) continue
      if (linksTo[slots.length + i]) {
        this.checkPrimitiveConnection(
          linksTo[slots.length + i],
          inputName,
          inputs as Record<string, unknown[]>
        )
        // This input is linked so we can skip it
        continue
      }

      const { name, config } = this.getInputConfig(
        node,
        inputName,
        seenInputs,
        inputs[inputName] as unknown[],
        {
          defaultInput: true
        }
      )

      if (this.nodeDef?.input?.required) {
        // @ts-expect-error legacy dynamic input assignment
        this.nodeDef.input.required[name] = config
      }
      this.newToOldWidgetMap[name] = { node, inputName }

      const nodeIndex = node.index ?? -1
      if (!this.oldToNewWidgetMap[nodeIndex]) {
        this.oldToNewWidgetMap[nodeIndex] = {}
      }
      this.oldToNewWidgetMap[nodeIndex][inputName] = name

      inputMap[slots.length + i] = this.inputCount++
    }
  }

  private _convertedToProcess: (() => void)[] = []
  processNodeInputs(
    node: GroupNodeData,
    seenInputs: Record<string, number>,
    inputs: Record<string, unknown>
  ) {
    const inputMapping: unknown[] = []

    const inputNames = Object.keys(inputs)
    if (!inputNames.length) return

    const { converted, slots } = this.processWidgetInputs(
      inputs,
      node,
      inputNames,
      seenInputs
    )
    const nodeIndex = node.index ?? -1
    const linksTo = this.linksTo[nodeIndex] ?? {}
    const inputMap: Record<number, number> = (this.oldToNewInputMap[nodeIndex] =
      {})
    this.processInputSlots(
      inputs as unknown as Record<string, unknown[]>,
      node,
      slots,
      linksTo,
      inputMap,
      seenInputs
    )

    // Converted inputs have to be processed after all other nodes as they'll be at the end of the list
    this._convertedToProcess.push(() =>
      this.processConvertedWidgets(
        inputs,
        node,
        slots,
        converted,
        linksTo,
        inputMap,
        seenInputs
      )
    )

    return inputMapping
  }

  processNodeOutputs(
    node: GroupNodeData,
    seenOutputs: Record<string, number>,
    def: GroupNodeDef | ComfyNodeDef
  ) {
    const nodeIndex = node.index ?? -1
    const oldToNew: Record<number, number> = (this.oldToNewOutputMap[
      nodeIndex
    ] = {})

    const defOutput = def.output ?? []
    // Add outputs
    for (let outputId = 0; outputId < defOutput.length; outputId++) {
      const linksFrom = this.linksFrom[nodeIndex]
      // If this output is linked internally we flag it to hide
      const hasLink =
        linksFrom?.[outputId] && !this.externalFrom[nodeIndex]?.[outputId]
      const outputConfig = this.nodeData.config?.[node.index ?? -1] as
        | NodeConfigEntry
        | undefined
      const customConfig = outputConfig?.output?.[outputId]
      const visible = customConfig?.visible ?? !hasLink
      this.outputVisibility.push(visible)
      if (!visible) {
        continue
      }

      if (this.nodeDef?.output) {
        oldToNew[outputId] = this.nodeDef.output.length
        this.newToOldOutputMap[this.nodeDef.output.length] = {
          node,
          slot: outputId
        }
        // @ts-expect-error legacy dynamic output type assignment
        this.nodeDef.output.push(defOutput[outputId])
        this.nodeDef.output_is_list?.push(
          def.output_is_list?.[outputId] ?? false
        )
      }

      let label: string | undefined = customConfig?.name
      if (!label) {
        const outputVal = defOutput[outputId]
        label =
          def.output_name?.[outputId] ??
          (typeof outputVal === 'string' ? outputVal : undefined)
        const output = node.outputs?.find((o) => o.name === label)
        if (output?.label) {
          label = output.label
        }
      }

      let name: string = String(label ?? `output_${outputId}`)
      if (name in seenOutputs) {
        const prefix = `${node.title ?? node.type} `
        name = `${prefix}${label ?? outputId}`
        if (name in seenOutputs) {
          name = `${prefix}${node.index} ${label ?? outputId}`
        }
      }
      seenOutputs[name] = 1

      this.nodeDef?.output_name?.push(name)
    }
  }

  static async registerFromWorkflow(
    groupNodes: Record<string, GroupNodeWorkflowData>,
    missingNodeTypes: (
      | string
      | { type: string; hint?: string; action?: unknown }
    )[]
  ) {
    for (const g in groupNodes) {
      const groupData = groupNodes[g]

      let hasMissing = false
      for (const n of groupData.nodes) {
        // Find missing node types
        if (!n.type || !(n.type in LiteGraph.registered_node_types)) {
          missingNodeTypes.push({
            type: n.type ?? 'unknown',
            hint: ` (In group node '${PREFIX}${SEPARATOR}${g}')`
          })

          missingNodeTypes.push({
            type: `${PREFIX}${SEPARATOR}` + g,
            action: {
              text: 'Remove from workflow',
              callback: (e: MouseEvent) => {
                delete groupNodes[g]
                const target = e.target as HTMLElement
                target.textContent = 'Removed'
                target.style.pointerEvents = 'none'
                target.style.opacity = '0.7'
              }
            }
          })

          hasMissing = true
        }
      }

      if (hasMissing) continue

      const config = new GroupNodeConfig(g, groupData)
      await config.registerType()
    }
  }
}

export class GroupNodeHandler {
  node: LGraphNode
  groupData: GroupNodeConfig
  innerNodes: LGraphNode[] | null = null

  constructor(node: LGraphNode) {
    this.node = node
    this.groupData = node.constructor?.nodeData?.[GROUP] as GroupNodeConfig

    this.node.setInnerNodes = (innerNodes) => {
      this.innerNodes = innerNodes

      for (
        let innerNodeIndex = 0;
        innerNodeIndex < this.innerNodes.length;
        innerNodeIndex++
      ) {
        const innerNode = this.innerNodes[innerNodeIndex]
        innerNode.graph ??= this.node.graph

        for (const w of innerNode.widgets ?? []) {
          if (w.type === 'converted-widget') {
            // @ts-expect-error legacy widget property for converted widgets
            w.serializeValue = w.origSerializeValue
          }
        }

        innerNode.index = innerNodeIndex
        innerNode.getInputNode = (slot: number) => {
          // Check if this input is internal or external
          const nodeIdx = innerNode.index ?? 0
          const externalSlot = this.groupData.oldToNewInputMap[nodeIdx]?.[slot]
          if (externalSlot != null) {
            return this.node.getInputNode(externalSlot)
          }

          // Internal link
          const innerLink = this.groupData.linksTo[nodeIdx]?.[slot]
          if (!innerLink) return null

          const linkSrcIdx = innerLink[0]
          if (linkSrcIdx == null) return null
          const inputNode = innerNodes[Number(linkSrcIdx)]
          // Primitives will already apply their values
          if (inputNode.type === 'PrimitiveNode') return null

          return inputNode
        }

        // @ts-expect-error returns partial link object, not full LLink
        innerNode.getInputLink = (slot: number) => {
          const nodeIdx = innerNode.index ?? 0
          const externalSlot = this.groupData.oldToNewInputMap[nodeIdx]?.[slot]
          if (externalSlot != null) {
            // The inner node is connected via the group node inputs
            const linkId = this.node.inputs[externalSlot].link
            if (linkId == null) return null
            const existingLink = app.rootGraph.links[linkId]
            if (!existingLink) return null

            // Use the outer link, but update the target to the inner node
            return {
              ...existingLink,
              target_id: innerNode.id,
              target_slot: +slot
            }
          }

          const innerLink = this.groupData.linksTo[nodeIdx]?.[slot]
          if (!innerLink) return null
          const linkSrcIdx = innerLink[0]
          if (linkSrcIdx == null) return null
          // Use the inner link, but update the origin node to be inner node id
          return {
            origin_id: innerNodes[Number(linkSrcIdx)].id,
            origin_slot: innerLink[1],
            target_id: innerNode.id,
            target_slot: +slot
          }
        }
      }
    }

    this.node.updateLink = (link) => {
      // Replace the group node reference with the internal node
      // @ts-expect-error Can this be removed?  Or replaced with: LLink.create(link.asSerialisable())
      link = { ...link }
      const output = this.groupData.newToOldOutputMap[link.origin_slot]
      if (!output || !this.innerNodes) return null
      const nodeIdx = output.node.index ?? 0
      let innerNode: LGraphNode | null = this.innerNodes[nodeIdx]
      let l
      while (innerNode?.type === 'Reroute') {
        l = innerNode.getInputLink(0)
        innerNode = innerNode.getInputNode(0)
      }

      if (!innerNode) {
        return null
      }

      if (
        l &&
        GroupNodeHandler.isGroupNode(innerNode) &&
        innerNode.updateLink
      ) {
        return innerNode.updateLink(l)
      }

      link.origin_id = innerNode.id
      link.origin_slot = l?.origin_slot ?? output.slot
      return link
    }

    /** @internal Used to flatten the subgraph before execution. Recursive; call with no args. */
    this.node.getInnerNodes = (
      computedNodeDtos: Map<ExecutionId, ExecutableLGraphNode>,
      /** The path of subgraph node IDs. */
      subgraphNodePath: readonly NodeId[] = [],
      /** The list of nodes to add to. */
      nodes: ExecutableLGraphNode[] = [],
      /** The set of visited nodes. */
      visited = new Set<LGraphNode>()
    ): ExecutableLGraphNode[] => {
      if (visited.has(this.node))
        throw new Error('RecursionError: while flattening subgraph')
      visited.add(this.node)

      if (!this.innerNodes) {
        const createdNodes = this.groupData.nodeData.nodes
          .map((n, i) => {
            if (!n.type) return null
            const innerNode = LiteGraph.createNode(n.type)
            if (!innerNode) return null
            // @ts-expect-error legacy node data format used for configure
            innerNode.configure(n)
            innerNode.id = `${this.node.id}:${i}`
            innerNode.graph = this.node.graph
            return innerNode
          })
          .filter((n): n is LGraphNode => n !== null)
        this.node.setInnerNodes?.(createdNodes)
      }

      this.updateInnerWidgets()

      const subgraphInstanceIdPath = [...subgraphNodePath, this.node.id]

      // Assertion: Deprecated, does not matter.
      const subgraphNode = (this.node.graph?.getNodeById(
        subgraphNodePath.at(-1)
      ) ?? undefined) as SubgraphNode | undefined

      for (const node of this.innerNodes ?? []) {
        node.graph ??= this.node.graph

        // Create minimal DTOs rather than cloning the node
        const currentId = String(node.id)
        // @ts-expect-error temporary id reassignment for DTO creation
        node.id = currentId.split(':').at(-1)
        const aVeryRealNode = new ExecutableGroupNodeChildDTO(
          node,
          subgraphInstanceIdPath,
          computedNodeDtos,
          subgraphNode
        )
        node.id = currentId
        aVeryRealNode.groupNodeHandler = this

        nodes.push(aVeryRealNode)
      }
      return nodes
    }

    // @ts-expect-error recreate returns null if creation fails
    this.node.recreate = async () => {
      const id = this.node.id
      const sz = this.node.size
      const nodes = (
        this.node as LGraphNode & { convertToNodes?: () => LGraphNode[] }
      ).convertToNodes?.()
      if (!nodes) return null

      const groupNode = LiteGraph.createNode(this.node.type)
      if (!groupNode) return null
      groupNode.id = id

      // Reuse the existing nodes for this instance
      groupNode.setInnerNodes?.(nodes)
      const handler = GroupNodeHandler.getHandler(groupNode)
      handler?.populateWidgets()
      app.rootGraph.add(groupNode)
      groupNode.setSize?.([
        Math.max(groupNode.size[0], sz[0]),
        Math.max(groupNode.size[1], sz[1])
      ])

      // Remove all converted nodes and relink them
      const builder = new GroupNodeBuilder(nodes)
      const nodeData = builder.getNodeData()
      if (handler) {
        handler.groupData.nodeData.links = nodeData.links
        handler.replaceNodes(nodes)
      }
      return groupNode
    }
    ;(
      this.node as LGraphNode & { convertToNodes: () => LGraphNode[] }
    ).convertToNodes = () => {
      const addInnerNodes = () => {
        // Clone the node data so we dont mutate it for other nodes
        const c = { ...this.groupData.nodeData }
        c.nodes = [...c.nodes]
        // @ts-expect-error getInnerNodes called without args in legacy conversion code
        const innerNodes = this.node.getInnerNodes?.()
        const ids: (string | number)[] = []
        for (let i = 0; i < c.nodes.length; i++) {
          let id: string | number | undefined = innerNodes?.[i]?.id
          // Use existing IDs if they are set on the inner nodes
          if (id == null || (typeof id === 'number' && isNaN(id))) {
            id = undefined
          } else {
            ids.push(id)
          }
          // @ts-expect-error adding id to node copy for serialization
          c.nodes[i] = { ...c.nodes[i], id }
        }
        deserialiseAndCreate(JSON.stringify(c), app.canvas)

        const [x, y] = this.node.pos
        let top: number | undefined
        let left: number | undefined
        // Configure nodes with current widget data
        const selectedIds = ids.length
          ? ids
          : Object.keys(app.canvas.selected_nodes)
        const newNodes: LGraphNode[] = []
        for (let i = 0; i < selectedIds.length; i++) {
          const id = selectedIds[i]
          const newNode = app.rootGraph.getNodeById(id)
          const innerNode = innerNodes?.[i]
          if (!newNode) continue
          newNodes.push(newNode)

          if (left == null || newNode.pos[0] < left) {
            left = newNode.pos[0]
          }
          if (top == null || newNode.pos[1] < top) {
            top = newNode.pos[1]
          }

          if (!newNode.widgets || !innerNode) continue

          // @ts-expect-error index property access on ExecutableLGraphNode
          const map = this.groupData.oldToNewWidgetMap[innerNode.index ?? 0]
          if (map) {
            const widgets = Object.keys(map)

            for (const oldName of widgets) {
              const newName = map[oldName]
              if (!newName) continue

              const widgetIndex =
                this.node.widgets?.findIndex((w) => w.name === newName) ?? -1
              if (widgetIndex === -1) continue

              // Populate the main and any linked widgets
              if (innerNode.type === 'PrimitiveNode') {
                for (let i = 0; i < newNode.widgets.length; i++) {
                  const srcWidget = this.node.widgets?.[widgetIndex + i]
                  if (srcWidget) {
                    newNode.widgets[i].value = srcWidget.value
                  }
                }
              } else {
                const outerWidget = this.node.widgets?.[widgetIndex]
                const newWidget = newNode.widgets.find(
                  (w) => w.name === oldName
                )
                if (!newWidget || !outerWidget) continue

                newWidget.value = outerWidget.value
                const linkedWidgets = outerWidget.linkedWidgets ?? []
                for (let w = 0; w < linkedWidgets.length; w++) {
                  const newLinked = newWidget.linkedWidgets?.[w]
                  if (newLinked && linkedWidgets[w]) {
                    newLinked.value = linkedWidgets[w].value
                  }
                }
              }
            }
          }
        }

        // Shift each node
        for (const newNode of newNodes) {
          newNode.pos[0] -= (left ?? 0) - x
          newNode.pos[1] -= (top ?? 0) - y
        }

        return { newNodes, selectedIds }
      }

      const reconnectInputs = (selectedIds: (string | number)[]) => {
        for (const innerNodeIndex in this.groupData.oldToNewInputMap) {
          const id = selectedIds[Number(innerNodeIndex)]
          const newNode = app.rootGraph.getNodeById(id)
          if (!newNode) continue
          const map = this.groupData.oldToNewInputMap[Number(innerNodeIndex)]
          for (const innerInputId in map) {
            const groupSlotId = map[Number(innerInputId)]
            if (groupSlotId == null) continue
            const slot = node.inputs[groupSlotId]
            if (slot.link == null) continue
            const link = app.rootGraph.links[slot.link]
            if (!link) continue
            //  connect this node output to the input of another node
            const originNode = app.rootGraph.getNodeById(link.origin_id)
            originNode?.connect(link.origin_slot, newNode, +innerInputId)
          }
        }
      }

      const reconnectOutputs = (selectedIds: (string | number)[]) => {
        for (
          let groupOutputId = 0;
          groupOutputId < node.outputs?.length;
          groupOutputId++
        ) {
          const output = node.outputs[groupOutputId]
          if (!output.links) continue
          const links = [...output.links]
          for (const l of links) {
            const slot = this.groupData.newToOldOutputMap[groupOutputId]
            if (!slot) continue
            const link = app.rootGraph.links[l]
            if (!link) continue
            const targetNode = app.rootGraph.getNodeById(link.target_id)
            const newNode = app.rootGraph.getNodeById(
              selectedIds[slot.node.index ?? 0]
            )
            if (targetNode) {
              newNode?.connect(slot.slot, targetNode, link.target_slot)
            }
          }
        }
      }

      app.canvas.emitBeforeChange()

      try {
        const { newNodes, selectedIds } = addInnerNodes()
        reconnectInputs(selectedIds)
        reconnectOutputs(selectedIds)
        app.rootGraph.remove(this.node)

        return newNodes
      } finally {
        app.canvas.emitAfterChange()
      }
    }

    const getExtraMenuOptions = this.node.getExtraMenuOptions
    const handlerNode = this.node
    this.node.getExtraMenuOptions = function (_canvas, options) {
      getExtraMenuOptions?.call(this, _canvas, options)

      let optionIndex = options.findIndex((o) => o?.content === 'Outputs')
      if (optionIndex === -1) optionIndex = options.length
      else optionIndex++
      options.splice(
        optionIndex,
        0,
        null,
        {
          content: 'Convert to nodes',
          // @ts-expect-error async callback not expected by legacy menu API
          callback: async () => {
            const convertFn = (
              handlerNode as LGraphNode & {
                convertToNodes?: () => LGraphNode[]
              }
            ).convertToNodes
            return convertFn?.()
          }
        },
        {
          content: 'Manage Group Node',
          callback: () => manageGroupNodes(this.type)
        }
      )
      // Return empty array to satisfy type signature without triggering
      // LGraphCanvas concatenation (which only happens when length > 0)
      return []
    }

    // Draw custom collapse icon to identity this as a group
    const onDrawTitleBox = this.node.onDrawTitleBox
    this.node.onDrawTitleBox = function (ctx, height, size, scale) {
      onDrawTitleBox?.call(this, ctx, height, size, scale)

      const fill = ctx.fillStyle
      ctx.beginPath()
      ctx.rect(11, -height + 11, 2, 2)
      ctx.rect(14, -height + 11, 2, 2)
      ctx.rect(17, -height + 11, 2, 2)
      ctx.rect(11, -height + 14, 2, 2)
      ctx.rect(14, -height + 14, 2, 2)
      ctx.rect(17, -height + 14, 2, 2)
      ctx.rect(11, -height + 17, 2, 2)
      ctx.rect(14, -height + 17, 2, 2)
      ctx.rect(17, -height + 17, 2, 2)

      ctx.fillStyle = this.boxcolor || LiteGraph.NODE_DEFAULT_BOXCOLOR
      ctx.fill()
      ctx.fillStyle = fill
    }

    // Draw progress label
    const onDrawForeground = node.onDrawForeground
    const groupData = this.groupData.nodeData
    node.onDrawForeground = function (ctx, canvas, canvasElement) {
      onDrawForeground?.call(this, ctx, canvas, canvasElement)
      const progressState = useExecutionStore().nodeProgressStates[this.id]
      if (
        progressState &&
        progressState.state === 'running' &&
        this.runningInternalNodeId !== null
      ) {
        const nodeIdx =
          typeof this.runningInternalNodeId === 'number'
            ? this.runningInternalNodeId
            : parseInt(String(this.runningInternalNodeId), 10)
        const n = groupData.nodes[nodeIdx] as { title?: string; type?: string }
        if (!n) return
        const message = `Running ${n.title || n.type} (${this.runningInternalNodeId}/${groupData.nodes.length})`
        ctx.save()
        ctx.font = '12px sans-serif'
        const sz = ctx.measureText(message)
        ctx.fillStyle = node.boxcolor || LiteGraph.NODE_DEFAULT_BOXCOLOR
        ctx.beginPath()
        ctx.roundRect(
          0,
          -LiteGraph.NODE_TITLE_HEIGHT - 20,
          sz.width + 12,
          20,
          5
        )
        ctx.fill()

        ctx.fillStyle = '#fff'
        ctx.fillText(message, 6, -LiteGraph.NODE_TITLE_HEIGHT - 6)
        ctx.restore()
      }
    }

    // Flag this node as needing to be reset
    const onExecutionStart = this.node.onExecutionStart
    this.node.onExecutionStart = function () {
      ;(this as LGraphNode & { resetExecution?: boolean }).resetExecution = true
      return onExecutionStart?.call(this)
    }

    const onNodeCreated = this.node.onNodeCreated
    const handlerGroupData = this.groupData
    this.node.onNodeCreated = function () {
      if (!this.widgets) {
        return
      }
      const config = handlerGroupData.nodeData.config as
        | Record<number, NodeConfigEntry>
        | undefined
      if (config) {
        for (const n in config) {
          const inputs = config[n]?.input
          if (!inputs) continue
          for (const w in inputs) {
            if (inputs[w]?.visible !== false) continue
            const widgetName =
              handlerGroupData.oldToNewWidgetMap[Number(n)]?.[w]
            const widget = this.widgets.find((wg) => wg.name === widgetName)
            if (widget) {
              widget.hidden = true
              widget.computeSize = () => [0, -4]
            }
          }
        }
      }

      return onNodeCreated?.call(this)
    }

    type EventDetail = { display_node?: string; node?: string } | string
    const handleEvent = (
      type: string,
      getId: (detail: EventDetail) => string | undefined,
      getEvent: (
        detail: EventDetail,
        id: string,
        node: LGraphNode
      ) => EventDetail
    ) => {
      const handler = ({ detail }: CustomEvent<EventDetail>) => {
        const id = getId(detail)
        if (!id) return
        const existingNode = app.rootGraph.getNodeById(id)
        if (existingNode) return

        const innerNodeIndex =
          this.innerNodes?.findIndex((n) => n.id == id) ?? -1
        if (innerNodeIndex > -1) {
          ;(
            this.node as LGraphNode & { runningInternalNodeId?: number }
          ).runningInternalNodeId = innerNodeIndex
          api.dispatchCustomEvent(
            type as 'executing',
            getEvent(detail, `${this.node.id}`, this.node) as string
          )
        }
      }
      api.addEventListener(
        type as 'executing' | 'executed',
        handler as EventListener
      )
      return handler
    }

    const executing = handleEvent(
      'executing',
      (d) => (typeof d === 'string' ? d : undefined),
      (_d, id) => id
    )

    const executed = handleEvent(
      'executed',
      (d) => (typeof d === 'object' ? d?.display_node || d?.node : undefined),
      (d, id, node) => ({
        ...(typeof d === 'object' ? d : {}),
        node: id,
        display_node: id,
        merge: !(node as LGraphNode & { resetExecution?: boolean })
          .resetExecution
      })
    )

    const onRemoved = node.onRemoved
    this.node.onRemoved = function () {
      onRemoved?.call(this)
      api.removeEventListener('executing', executing as EventListener)
      api.removeEventListener('executed', executed as EventListener)
    }

    this.node.refreshComboInNode = (defs) => {
      // Update combo widget options
      for (const widgetName in this.groupData.newToOldWidgetMap) {
        const widget = this.node.widgets?.find((w) => w.name === widgetName)
        if (widget?.type === 'combo') {
          const old = this.groupData.newToOldWidgetMap[widgetName]
          if (!old.node.type) continue
          const def = defs[old.node.type]
          const input =
            def?.input?.required?.[old.inputName] ??
            def?.input?.optional?.[old.inputName]
          if (!input) continue

          widget.options.values = input[0] as unknown[]

          const values = widget.options.values as unknown[]
          if (old.inputName !== 'image' && !values.includes(widget.value)) {
            widget.value = values[0] as typeof widget.value
            widget.callback?.(widget.value)
          }
        }
      }
    }
  }

  updateInnerWidgets() {
    if (!this.innerNodes) return
    for (const newWidgetName in this.groupData.newToOldWidgetMap) {
      const newWidget = this.node.widgets?.find((w) => w.name === newWidgetName)
      if (!newWidget) continue

      const newValue = newWidget.value
      const old = this.groupData.newToOldWidgetMap[newWidgetName]
      const nodeIdx = old.node.index ?? 0
      const innerNode = this.innerNodes[nodeIdx]
      if (!innerNode) continue

      if (innerNode.type === 'PrimitiveNode') {
        // @ts-expect-error primitiveValue is a custom property on PrimitiveNode
        innerNode.primitiveValue = newValue
        const primitiveLinked = this.groupData.primitiveToWidget[nodeIdx]
        for (const linked of primitiveLinked ?? []) {
          const linkedNodeId =
            typeof linked.nodeId === 'number'
              ? linked.nodeId
              : Number(linked.nodeId)
          const linkedNode = this.innerNodes[linkedNodeId]
          const widget = linkedNode?.widgets?.find(
            (w) => w.name === linked.inputName
          )

          if (widget) {
            widget.value = newValue
          }
        }
        continue
      } else if (innerNode.type === 'Reroute') {
        const rerouteLinks = this.groupData.linksFrom[nodeIdx]
        if (rerouteLinks) {
          for (const [, , targetNodeId, targetSlot] of rerouteLinks[0] ?? []) {
            if (targetNodeId == null || targetSlot == null) continue
            const targetNode = this.innerNodes[Number(targetNodeId)]
            if (!targetNode) continue
            const input = targetNode.inputs?.[Number(targetSlot)]
            if (input?.widget) {
              const widgetName = input.widget.name
              const widget = targetNode.widgets?.find(
                (w) => w.name === widgetName
              )
              if (widget) {
                widget.value = newValue
              }
            }
          }
        }
      }

      const widget = innerNode.widgets?.find((w) => w.name === old.inputName)
      if (widget) {
        widget.value = newValue
      }
    }
  }

  populatePrimitive(
    _node: GroupNodeData,
    nodeId: number,
    oldName: string
  ): boolean {
    // Converted widget, populate primitive if linked
    const primitiveId = this.groupData.widgetToPrimitive[nodeId]?.[oldName]
    if (primitiveId == null) return false
    const targetWidgetName =
      this.groupData.oldToNewWidgetMap[
        Array.isArray(primitiveId) ? primitiveId[0] : primitiveId
      ]?.['value']
    if (!targetWidgetName) return false
    const targetWidgetIndex =
      this.node.widgets?.findIndex((w) => w.name === targetWidgetName) ?? -1
    if (targetWidgetIndex > -1 && this.innerNodes) {
      const primIdx = Array.isArray(primitiveId) ? primitiveId[0] : primitiveId
      const primitiveNode = this.innerNodes[primIdx]
      if (!primitiveNode?.widgets) return true
      let len = primitiveNode.widgets.length
      if (
        len - 1 !==
        (this.node.widgets?.[targetWidgetIndex]?.linkedWidgets?.length ?? 0)
      ) {
        // Fallback handling for if some reason the primitive has a different number of widgets
        // we dont want to overwrite random widgets, better to leave blank
        len = 1
      }
      for (let i = 0; i < len; i++) {
        const targetWidget = this.node.widgets?.[targetWidgetIndex + i]
        const srcWidget = primitiveNode.widgets[i]
        if (targetWidget && srcWidget) {
          targetWidget.value = srcWidget.value
        }
      }
    }
    return true
  }

  populateReroute(
    node: GroupNodeData,
    nodeId: number,
    map: Record<string, string | null>
  ) {
    if (node.type !== 'Reroute') return

    const link = this.groupData.linksFrom[nodeId]?.[0]?.[0]
    if (!link) return
    const targetNodeId = link[2]
    const targetNodeSlot = link[3]
    if (targetNodeId == null || targetNodeSlot == null) return
    const targetNode = this.groupData.nodeData.nodes[Number(targetNodeId)] as
      | GroupNodeData
      | undefined
    const inputs = targetNode?.inputs
    const targetWidget = (inputs as GroupNodeInput[] | undefined)?.[
      Number(targetNodeSlot)
    ]?.widget
    if (!targetWidget) return

    const offset =
      (inputs?.length ?? 0) - (targetNode?.widgets_values?.length ?? 0)
    const v = targetNode?.widgets_values?.[Number(targetNodeSlot) - offset]
    if (v == null) return

    const widgetName = Object.values(map)[0]
    const widget = this.node.widgets?.find((w) => w.name === widgetName)
    if (widget) {
      widget.value = v
    }
  }

  populateWidgets() {
    if (!this.node.widgets) return

    for (
      let nodeId = 0;
      nodeId < this.groupData.nodeData.nodes.length;
      nodeId++
    ) {
      const node = this.groupData.nodeData.nodes[nodeId] as GroupNodeData
      const map = this.groupData.oldToNewWidgetMap[nodeId] ?? {}
      const widgets = Object.keys(map)

      if (!node.widgets_values?.length) {
        // special handling for populating values into reroutes
        // this allows primitives connect to them to pick up the correct value
        this.populateReroute(node, nodeId, map)
        continue
      }

      let linkedShift = 0
      for (let i = 0; i < widgets.length; i++) {
        const oldName = widgets[i]
        const newName = map[oldName]
        const widgetIndex = this.node.widgets.findIndex(
          (w) => w.name === newName
        )
        const mainWidget = this.node.widgets[widgetIndex]
        if (
          this.populatePrimitive(node, nodeId, oldName) ||
          widgetIndex === -1
        ) {
          // Find the inner widget and shift by the number of linked widgets as they will have been removed too
          const innerWidget = this.innerNodes?.[nodeId]?.widgets?.find(
            (w) => w.name === oldName
          )
          linkedShift += innerWidget?.linkedWidgets?.length ?? 0
        }
        if (widgetIndex === -1) {
          continue
        }

        // Populate the main and any linked widget
        mainWidget.value = node.widgets_values?.[
          i + linkedShift
        ] as typeof mainWidget.value
        const linkedWidgets = mainWidget.linkedWidgets ?? []
        for (let w = 0; w < linkedWidgets.length; w++) {
          this.node.widgets[widgetIndex + w + 1].value = node.widgets_values?.[
            i + ++linkedShift
          ] as typeof mainWidget.value
        }
      }
    }
  }

  replaceNodes(nodes: LGraphNode[]) {
    let top: number | undefined
    let left: number | undefined

    for (let i = 0; i < nodes.length; i++) {
      const node = nodes[i]
      if (left == null || node.pos[0] < left) {
        left = node.pos[0]
      }
      if (top == null || node.pos[1] < top) {
        top = node.pos[1]
      }

      this.linkOutputs(node, i)
      app.rootGraph.remove(node)

      // Set internal ID to what is expected after workflow is reloaded
      node.id = `${this.node.id}:${i}`
    }

    this.linkInputs()
    this.node.pos = [left ?? 0, top ?? 0]
  }

  linkOutputs(originalNode: LGraphNode, nodeId: number) {
    if (!originalNode.outputs) return

    for (const output of originalNode.outputs) {
      if (!output.links) continue
      // Clone the links as they'll be changed if we reconnect
      const links = [...output.links]
      for (const l of links) {
        const link = app.rootGraph.links[l]
        if (!link) continue

        const targetNode = app.rootGraph.getNodeById(link.target_id)
        const newSlot =
          this.groupData.oldToNewOutputMap[nodeId]?.[link.origin_slot]
        if (newSlot != null && targetNode) {
          this.node.connect(newSlot, targetNode, link.target_slot)
        }
      }
    }
  }

  linkInputs() {
    for (const link of this.groupData.nodeData.links ?? []) {
      const [, originSlot, targetId, targetSlot, actualOriginId] = link
      if (actualOriginId == null || typeof actualOriginId === 'object') continue
      const originNode = app.rootGraph.getNodeById(actualOriginId)
      if (!originNode) continue // this node is in the group
      if (targetId == null || targetSlot == null) continue
      const mappedSlot =
        this.groupData.oldToNewInputMap[Number(targetId)]?.[Number(targetSlot)]
      if (mappedSlot == null) continue
      if (typeof originSlot === 'number' || typeof originSlot === 'string') {
        originNode.connect(
          originSlot,
          // @ts-expect-error Valid - uses deprecated interface (node ID instead of node reference)
          this.node.id,
          mappedSlot
        )
      }
    }
  }

  static getGroupData(
    node: LGraphNodeConstructor<LGraphNode>
  ): GroupNodeConfig | undefined
  static getGroupData(node: LGraphNode): GroupNodeConfig | undefined
  static getGroupData(
    node: LGraphNode | LGraphNodeConstructor<LGraphNode>
  ): GroupNodeConfig | undefined {
    // Check if this is a constructor (function) or an instance
    if (typeof node === 'function') {
      // Constructor case - access nodeData directly
      const nodeData = (node as LGraphNodeConstructor & { nodeData?: unknown })
        .nodeData as Record<symbol, GroupNodeConfig> | undefined
      return nodeData?.[GROUP]
    }
    // Instance case - check instance property first, then constructor
    const instanceData = (node as LGraphNode & { nodeData?: unknown })
      .nodeData as Record<symbol, GroupNodeConfig> | undefined
    if (instanceData?.[GROUP]) return instanceData[GROUP]
    const ctorData = (
      node.constructor as LGraphNodeConstructor & { nodeData?: unknown }
    )?.nodeData as Record<symbol, GroupNodeConfig> | undefined
    return ctorData?.[GROUP]
  }

  static getHandler(node: LGraphNode): GroupNodeHandler | undefined {
    // @ts-expect-error GROUP symbol indexing on LGraphNode
    let handler = node[GROUP] as GroupNodeHandler | undefined
    // Handler may not be set yet if nodeCreated async hook hasn't run
    // Create it synchronously if needed
    if (!handler && GroupNodeHandler.isGroupNode(node)) {
      handler = new GroupNodeHandler(node)
      ;(node as LGraphNode & { [GROUP]: GroupNodeHandler })[GROUP] = handler
    }
    return handler
  }

  static isGroupNode(node: LGraphNode) {
    return !!node.constructor?.nodeData?.[GROUP]
  }

  static async fromNodes(nodes: LGraphNode[]) {
    // Process the nodes into the stored workflow group node data
    const builder = new GroupNodeBuilder(nodes)
    const res = await builder.build()
    if (!res) return

    const { name, nodeData } = res

    // Convert this data into a LG node definition and register it
    const config = new GroupNodeConfig(name, nodeData)
    await config.registerType()

    const groupNode = LiteGraph.createNode(`${PREFIX}${SEPARATOR}${name}`)
    if (!groupNode) return
    // Reuse the existing nodes for this instance
    groupNode.setInnerNodes?.(builder.nodes)
    const handler = GroupNodeHandler.getHandler(groupNode)
    handler?.populateWidgets()
    app.rootGraph.add(groupNode)

    // Remove all converted nodes and relink them
    handler?.replaceNodes(builder.nodes)
    return groupNode
  }
}

const replaceLegacySeparators = (nodes: ComfyNode[]): void => {
  for (const node of nodes) {
    if (typeof node.type === 'string' && node.type.startsWith('workflow/')) {
      node.type = node.type.replace(/^workflow\//, `${PREFIX}${SEPARATOR}`)
    }
  }
}

/**
 * Convert selected nodes to a group node
 * @throws {Error} if no nodes are selected
 * @throws {Error} if a group node is already selected
 * @throws {Error} if a group node is selected
 *
 * The context menu item should not be available if any of the above conditions are met.
 * The error is automatically handled by the commandStore when the command is executed.
 */
async function convertSelectedNodesToGroupNode() {
  const nodes = Object.values(app.canvas.selected_nodes ?? {})
  if (nodes.length === 0) {
    throw new Error('No nodes selected')
  }
  if (nodes.length === 1) {
    throw new Error('Please select multiple nodes to convert to group node')
  }

  for (const node of nodes) {
    if (node instanceof SubgraphNode) {
      throw new Error('Selected nodes contain a subgraph node')
    }
    if (GroupNodeHandler.isGroupNode(node)) {
      throw new Error('Selected nodes contain a group node')
    }
  }
  return await GroupNodeHandler.fromNodes(nodes)
}

const convertDisabled = (selected: LGraphNode[]) =>
  selected.length < 2 || !!selected.find((n) => GroupNodeHandler.isGroupNode(n))

function ungroupSelectedGroupNodes() {
  const nodes = Object.values(app.canvas.selected_nodes ?? {})
  for (const node of nodes) {
    if (GroupNodeHandler.isGroupNode(node)) {
      node.convertToNodes?.()
    }
  }
}

function manageGroupNodes(type?: string) {
  new ManageGroupDialog(app).show(type)
}

const id = 'Comfy.GroupNode'

/**
 * Global node definitions cache, populated and mutated by extension callbacks.
 *
 * **Initialization**: Set by `addCustomNodeDefs` during extension initialization.
 * This callback runs early in the app lifecycle, before any code that reads
 * `globalDefs` is executed.
 *
 * **Mutation**: `refreshComboInNodes` merges updated definitions into this object
 * when combo options are refreshed (e.g., after model files change).
 *
 * **Usage Notes**:
 * - Functions reading `globalDefs` (e.g., `getNodeDef`, `checkPrimitiveConnection`)
 *   must only be called after `addCustomNodeDefs` has run.
 * - Not thread-safe; assumes single-threaded JS execution model.
 * - The object reference is stable after initialization; only contents are mutated.
 */
let globalDefs: Record<string, ComfyNodeDef>
const ext: ComfyExtension = {
  name: id,
  commands: [
    {
      id: 'Comfy.GroupNode.ConvertSelectedNodesToGroupNode',
      label: 'Convert selected nodes to group node',
      icon: 'pi pi-sitemap',
      versionAdded: '1.3.17',
      function: () => convertSelectedNodesToGroupNode()
    },
    {
      id: 'Comfy.GroupNode.UngroupSelectedGroupNodes',
      label: 'Ungroup selected group nodes',
      icon: 'pi pi-sitemap',
      versionAdded: '1.3.17',
      function: () => ungroupSelectedGroupNodes()
    },
    {
      id: 'Comfy.GroupNode.ManageGroupNodes',
      label: 'Manage group nodes',
      icon: 'pi pi-cog',
      versionAdded: '1.3.17',
      function: (...args: unknown[]) =>
        manageGroupNodes(args[0] as string | undefined)
    }
  ],
  keybindings: [
    {
      commandId: 'Comfy.GroupNode.ConvertSelectedNodesToGroupNode',
      combo: {
        alt: true,
        key: 'g'
      }
    },
    {
      commandId: 'Comfy.GroupNode.UngroupSelectedGroupNodes',
      combo: {
        alt: true,
        shift: true,
        key: 'G'
      }
    }
  ],

  getCanvasMenuItems(canvas): IContextMenuValue[] {
    const items: IContextMenuValue[] = []
    const selected = Object.values(canvas.selected_nodes ?? {})
    const convertEnabled = !convertDisabled(selected)

    items.push({
      content: `Convert to Group Node (Deprecated)`,
      disabled: !convertEnabled,
      // @ts-expect-error async callback - legacy menu API doesn't expect Promise
      callback: async () => convertSelectedNodesToGroupNode()
    })

    const groups = canvas.graph?.extra?.groupNodes
    const manageDisabled = !groups || !Object.keys(groups).length
    items.push({
      content: `Manage Group Nodes`,
      disabled: manageDisabled,
      callback: () => manageGroupNodes()
    })

    return items
  },

  getNodeMenuItems(node): IContextMenuValue[] {
    if (GroupNodeHandler.isGroupNode(node)) {
      return []
    }

    const selected = Object.values(app.canvas.selected_nodes ?? {})
    const convertEnabled = !convertDisabled(selected)

    return [
      {
        content: `Convert to Group Node (Deprecated)`,
        disabled: !convertEnabled,
        // @ts-expect-error async callback - legacy menu API doesn't expect Promise
        callback: async () => convertSelectedNodesToGroupNode()
      }
    ]
  },
  async beforeConfigureGraph(
    graphData: ComfyWorkflowJSON,
    missingNodeTypes: string[]
  ) {
    const nodes = graphData?.extra?.groupNodes as
      | Record<string, GroupNodeWorkflowData>
      | undefined
    if (nodes) {
      replaceLegacySeparators(graphData.nodes)
      await GroupNodeConfig.registerFromWorkflow(nodes, missingNodeTypes)
    }
  },
  addCustomNodeDefs(defs: Record<string, ComfyNodeDef>) {
    // Store this so we can mutate it later with group nodes
    globalDefs = defs
  },
  nodeCreated(node: LGraphNode) {
    if (GroupNodeHandler.isGroupNode(node)) {
      ;(node as LGraphNode & { [GROUP]: GroupNodeHandler })[GROUP] =
        new GroupNodeHandler(node)

      // Ensure group nodes pasted from other workflows are stored
      const handler = GroupNodeHandler.getHandler(node)
      if (node.title && handler?.groupData?.nodeData) {
        Workflow.storeGroupNode(node.title, handler.groupData.nodeData)
      }
    }
  },
  async refreshComboInNodes(defs: Record<string, ComfyNodeDef>) {
    // Re-register group nodes so new ones are created with the correct options
    Object.assign(globalDefs, defs)
    const nodes = app.rootGraph.extra?.groupNodes as
      | Record<string, GroupNodeWorkflowData>
      | undefined
    if (nodes) {
      await GroupNodeConfig.registerFromWorkflow(nodes, [])
    }
  }
}

app.registerExtension(ext)
