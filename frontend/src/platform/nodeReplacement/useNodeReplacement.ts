import type { LGraph, LGraphNode } from '@/lib/litegraph/src/litegraph'
import { LiteGraph } from '@/lib/litegraph/src/litegraph'
import type { ISerialisedNode } from '@/lib/litegraph/src/types/serialisation'
import type { TWidgetValue } from '@/lib/litegraph/src/types/widgets'
import { t } from '@/i18n'
import type { NodeReplacement } from '@/platform/nodeReplacement/types'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import { app, sanitizeNodeName } from '@/scripts/app'
import type { MissingNodeType } from '@/types/comfy'
import { collectAllNodes } from '@/utils/graphTraversalUtil'

/** Compares sanitized type strings to match placeholder → missing node type. */
function findMatchingType(
  node: LGraphNode,
  selectedTypes: MissingNodeType[]
): Extract<MissingNodeType, { type: string }> | undefined {
  const nodeType = node.type
  for (const selected of selectedTypes) {
    if (typeof selected !== 'object' || !selected.isReplaceable) continue
    if (sanitizeNodeName(selected.type) === nodeType) return selected
  }
  return undefined
}

function transferInputConnection(
  oldNode: LGraphNode,
  oldInputName: string,
  newNode: LGraphNode,
  newInputName: string,
  graph: LGraph
): void {
  const oldSlotIdx = oldNode.inputs?.findIndex((i) => i.name === oldInputName)
  const newSlotIdx = newNode.inputs?.findIndex((i) => i.name === newInputName)
  if (oldSlotIdx == null || oldSlotIdx === -1) return
  if (newSlotIdx == null || newSlotIdx === -1) return

  const linkId = oldNode.inputs[oldSlotIdx].link
  if (linkId == null) return

  const link = graph.links.get(linkId)
  if (!link) return

  link.target_id = newNode.id
  link.target_slot = newSlotIdx
  newNode.inputs[newSlotIdx].link = linkId
  oldNode.inputs[oldSlotIdx].link = null
}

function transferOutputConnections(
  oldNode: LGraphNode,
  oldOutputIdx: number,
  newNode: LGraphNode,
  newOutputIdx: number,
  graph: LGraph
): void {
  const oldLinks = oldNode.outputs?.[oldOutputIdx]?.links
  if (!oldLinks?.length) return
  if (!newNode.outputs?.[newOutputIdx]) return

  for (const linkId of oldLinks) {
    const link = graph.links.get(linkId)
    if (!link) continue
    link.origin_id = newNode.id
    link.origin_slot = newOutputIdx
  }
  newNode.outputs[newOutputIdx].links = [...oldLinks]
  oldNode.outputs[oldOutputIdx].links = []
}

/** Uses old_widget_ids as name→index lookup into widgets_values. */
function transferWidgetValue(
  serialized: ISerialisedNode,
  oldWidgetIds: string[] | null,
  oldInputName: string,
  newNode: LGraphNode,
  newInputName: string
): void {
  if (!oldWidgetIds || !serialized.widgets_values) return

  const oldWidgetIdx = oldWidgetIds.indexOf(oldInputName)
  if (oldWidgetIdx === -1) return

  const oldValue = serialized.widgets_values[oldWidgetIdx]
  if (oldValue === undefined) return

  const newWidget = newNode.widgets?.find((w) => w.name === newInputName)
  if (newWidget) {
    newWidget.value = oldValue
    newWidget.callback?.(oldValue)
  }
}

function applySetValue(
  newNode: LGraphNode,
  inputName: string,
  value: unknown
): void {
  const widget = newNode.widgets?.find((w) => w.name === inputName)
  if (widget) {
    widget.value = value as TWidgetValue
    widget.callback?.(widget.value)
  }
}

function isDotNotation(id: string): boolean {
  return id.includes('.')
}

/** Auto-generates identity mapping by name for same-structure replacements without backend mapping. */
function generateDefaultMapping(
  serialized: ISerialisedNode,
  newNode: LGraphNode
): Pick<
  NodeReplacement,
  'input_mapping' | 'output_mapping' | 'old_widget_ids'
> {
  const oldInputNames = new Set(serialized.inputs?.map((i) => i.name) ?? [])

  const inputMapping: { old_id: string; new_id: string }[] = []
  for (const newInput of newNode.inputs ?? []) {
    if (oldInputNames.has(newInput.name)) {
      inputMapping.push({ old_id: newInput.name, new_id: newInput.name })
    }
  }

  const oldWidgetIds = (newNode.widgets ?? []).map((w) => w.name)
  for (const widget of newNode.widgets ?? []) {
    if (!oldInputNames.has(widget.name)) {
      inputMapping.push({ old_id: widget.name, new_id: widget.name })
    }
  }

  const outputMapping: { old_idx: number; new_idx: number }[] = []
  for (const [oldIdx, oldOutput] of (serialized.outputs ?? []).entries()) {
    const newIdx = newNode.outputs?.findIndex((o) => o.name === oldOutput.name)
    if (newIdx != null && newIdx !== -1) {
      outputMapping.push({ old_idx: oldIdx, new_idx: newIdx })
    }
  }

  return {
    input_mapping: inputMapping.length > 0 ? inputMapping : null,
    output_mapping: outputMapping.length > 0 ? outputMapping : null,
    old_widget_ids: oldWidgetIds.length > 0 ? oldWidgetIds : null
  }
}

function replaceWithMapping(
  node: LGraphNode,
  newNode: LGraphNode,
  replacement: NodeReplacement,
  nodeGraph: LGraph,
  idx: number
): void {
  newNode.id = node.id
  newNode.pos = [...node.pos]
  newNode.size = [...node.size]
  newNode.order = node.order
  newNode.mode = node.mode
  if (node.flags) newNode.flags = { ...node.flags }

  nodeGraph._nodes[idx] = newNode
  newNode.graph = nodeGraph
  nodeGraph._nodes_by_id[newNode.id] = newNode

  const serialized = node.last_serialization ?? node.serialize()

  if (serialized.title != null) newNode.title = serialized.title
  if (serialized.properties) {
    newNode.properties = { ...serialized.properties }
    if ('Node name for S&R' in newNode.properties) {
      newNode.properties['Node name for S&R'] = replacement.new_node_id
    }
  }

  if (replacement.input_mapping) {
    for (const inputMap of replacement.input_mapping) {
      if ('old_id' in inputMap) {
        if (isDotNotation(inputMap.new_id)) continue // Autogrow/DynamicCombo
        transferInputConnection(
          node,
          inputMap.old_id,
          newNode,
          inputMap.new_id,
          nodeGraph
        )
        transferWidgetValue(
          serialized,
          replacement.old_widget_ids,
          inputMap.old_id,
          newNode,
          inputMap.new_id
        )
      } else {
        if (!isDotNotation(inputMap.new_id)) {
          applySetValue(newNode, inputMap.new_id, inputMap.set_value)
        }
      }
    }
  }

  if (replacement.output_mapping) {
    for (const outMap of replacement.output_mapping) {
      transferOutputConnections(
        node,
        outMap.old_idx,
        newNode,
        outMap.new_idx,
        nodeGraph
      )
    }
  }

  newNode.has_errors = false
}

export function useNodeReplacement() {
  const toastStore = useToastStore()

  function replaceNodesInPlace(selectedTypes: MissingNodeType[]): string[] {
    const replacedTypes: string[] = []
    const graph = app.rootGraph

    const changeTracker =
      useWorkflowStore().activeWorkflow?.changeTracker ?? null
    changeTracker?.beforeChange()

    try {
      const placeholders = collectAllNodes(
        graph,
        (n) => !!n.has_errors && !!n.last_serialization
      )

      for (const node of placeholders) {
        const match = findMatchingType(node, selectedTypes)
        if (!match?.replacement) continue

        const replacement = match.replacement
        const nodeGraph = node.graph
        if (!nodeGraph) continue

        const idx = nodeGraph._nodes.indexOf(node)
        if (idx === -1) continue

        const newNode = LiteGraph.createNode(replacement.new_node_id)
        if (!newNode) continue

        const hasMapping =
          replacement.input_mapping != null ||
          replacement.output_mapping != null

        const effectiveReplacement = hasMapping
          ? replacement
          : {
              ...replacement,
              ...generateDefaultMapping(
                node.last_serialization ?? node.serialize(),
                newNode
              )
            }
        replaceWithMapping(node, newNode, effectiveReplacement, nodeGraph, idx)

        // Refresh Vue node data — replaceWithMapping bypasses graph.add()
        // so onNodeAdded must be called explicitly to update VueNodeData.
        nodeGraph.onNodeAdded?.(newNode)

        if (!replacedTypes.includes(match.type)) {
          replacedTypes.push(match.type)
        }
      }

      if (replacedTypes.length > 0) {
        graph.updateExecutionOrder()
        graph.setDirtyCanvas(true, true)

        toastStore.add({
          severity: 'success',
          summary: t('g.success'),
          detail: t('nodeReplacement.replacedAllNodes', {
            count: replacedTypes.length
          }),
          life: 3000
        })
      }
    } catch (error) {
      console.error('Failed to replace nodes:', error)
      if (replacedTypes.length > 0) {
        graph.updateExecutionOrder()
        graph.setDirtyCanvas(true, true)
      }
      toastStore.add({
        severity: 'error',
        summary: t('g.error', 'Error'),
        detail: t('nodeReplacement.replaceFailed', 'Failed to replace nodes'),
        life: 5000
      })
      return replacedTypes
    } finally {
      changeTracker?.afterChange()
    }

    return replacedTypes
  }

  return {
    replaceNodesInPlace
  }
}
