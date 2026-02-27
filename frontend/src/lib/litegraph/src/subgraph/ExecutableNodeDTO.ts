import type { LGraph } from '@/lib/litegraph/src/LGraph'
import type { LGraphNode, NodeId } from '@/lib/litegraph/src/LGraphNode'
import { InvalidLinkError } from '@/lib/litegraph/src/infrastructure/InvalidLinkError'
import { NullGraphError } from '@/lib/litegraph/src/infrastructure/NullGraphError'
import { RecursionError } from '@/lib/litegraph/src/infrastructure/RecursionError'
import { SlotIndexError } from '@/lib/litegraph/src/infrastructure/SlotIndexError'
import type { ISlotType } from '@/lib/litegraph/src/interfaces'
import { LGraphEventMode, LiteGraph } from '@/lib/litegraph/src/litegraph'

import type { Subgraph } from './Subgraph'
import type { SubgraphNode } from './SubgraphNode'

export type ExecutionId = string

/**
 * Interface describing the data transfer objects used when compiling a graph for execution.
 */
export type ExecutableLGraphNode = Omit<
  ExecutableNodeDTO,
  'graph' | 'node' | 'subgraphNode'
>

/**
 * The end result of resolving a DTO input.
 * When a widget value is returned, {@link widgetInfo} is present and {@link origin_slot} is `-1`.
 */
type ResolvedInput = {
  /** DTO for the node that the link originates from. */
  node: ExecutableLGraphNode
  /** Full unique execution ID of the node that the link originates from. In the case of a widget value, this is the ID of the subgraph node. */
  origin_id: ExecutionId
  /** The slot index of the output on the node that the link originates from. `-1` when widget value is set. */
  origin_slot: number
  /** Boxed widget value (e.g. for widgets). If this box is `undefined`, then an input link is connected, and widget values from the subgraph node are ignored. */
  widgetInfo?: { value: unknown }
}

/**
 * Concrete implementation of {@link ExecutableLGraphNode}.
 * @remarks This is the class that is used to create the data transfer objects for executable nodes.
 */
export class ExecutableNodeDTO implements ExecutableLGraphNode {
  applyToGraph?(
    ...args: Parameters<NonNullable<typeof this.node.applyToGraph>>
  ): ReturnType<NonNullable<typeof this.node.applyToGraph>>

  /** The graph that this node is a part of. */
  readonly graph: LGraph | Subgraph

  inputs: { linkId: number | null; name: string; type: ISlotType }[]

  /** Backing field for {@link id}. */
  private _id: ExecutionId

  /**
   * The path to the actual node through subgraph instances, represented as a list of all subgraph node IDs (instances),
   * followed by the actual original node ID within the subgraph. Each segment is separated by `:`.
   *
   * e.g. `1:2:3`:
   * - `1` is the node ID of the first subgraph node in the parent workflow
   * - `2` is the node ID of the second subgraph node in the first subgraph
   * - `3` is the node ID of the actual node in the subgraph definition
   */
  get id() {
    return this._id
  }

  get type() {
    return this.node.type
  }

  get title() {
    return this.node.title
  }

  get mode() {
    return this.node.mode
  }

  get comfyClass() {
    return this.node.comfyClass
  }

  get isVirtualNode() {
    return this.node.isVirtualNode
  }

  get widgets() {
    return this.node.widgets
  }

  get subgraphId() {
    return this.subgraphNode?.subgraph.id
  }

  constructor(
    /** The actual node that this DTO wraps. */
    readonly node: LGraphNode | SubgraphNode,
    /** A list of subgraph instance node IDs from the root graph to the containing instance. @see {@link id} */
    readonly subgraphNodePath: readonly NodeId[],
    /** A flattened map of all DTOs in this node network. Subgraph instances have been expanded into their inner nodes. */
    readonly nodesByExecutionId: Map<ExecutionId, ExecutableLGraphNode>,
    /** The actual subgraph instance that contains this node, otherwise undefined. */
    readonly subgraphNode?: SubgraphNode
  ) {
    if (!node.graph) throw new NullGraphError()

    // Set the internal ID of the DTO
    this._id = [...this.subgraphNodePath, this.node.id].join(':')
    this.graph = node.graph
    this.inputs = this.node.inputs.map((x) => ({
      linkId: x.link,
      name: x.name,
      type: x.type
    }))

    // Only create a wrapper if the node has an applyToGraph method
    if (this.node.applyToGraph) {
      this.applyToGraph = (...args) => this.node.applyToGraph?.(...args)
    }
  }

  /** Returns either the DTO itself, or the DTOs of the inner nodes of the subgraph. */
  getInnerNodes(): ExecutableLGraphNode[] {
    return this.node.isSubgraphNode()
      ? this.node.getInnerNodes(this.nodesByExecutionId, this.subgraphNodePath)
      : [this]
  }

  /**
   * Resolves the executable node & link IDs for a given input slot.
   * @param slot The slot index of the input.
   * @param visited Leave empty unless overriding this method.
   * A set of unique IDs, used to guard against infinite recursion.
   * If overriding, ensure that the set is passed on all recursive calls.
   * @returns The node and the origin ID / slot index of the output.
   */
  resolveInput(
    slot: number,
    visited = new Set<string>(),
    type?: ISlotType
  ): ResolvedInput | undefined {
    const uniqueId = `${this.subgraphNode?.subgraph.id}:${this.node.id}[I]${slot}`
    if (visited.has(uniqueId)) {
      const nodeInfo = `${this.node.id}${this.node.title ? ` (${this.node.title})` : ''}`
      const pathInfo =
        this.subgraphNodePath.length > 0
          ? ` at path ${this.subgraphNodePath.join(':')}`
          : ''
      throw new RecursionError(
        `Circular reference detected while resolving input ${slot} of node ${nodeInfo}${pathInfo}. ` +
          `This creates an infinite loop in link resolution. UniqueID: [${uniqueId}]`
      )
    }
    visited.add(uniqueId)

    const input = this.inputs.at(slot)
    if (!input)
      throw new SlotIndexError(
        `No input found for flattened id [${this.id}] slot [${slot}]`
      )

    // Nothing connected
    if (input.linkId == null) return

    const link = this.graph.getLink(input.linkId)
    if (!link)
      throw new InvalidLinkError(
        `No link found in parent graph for id [${this.id}] slot [${slot}] ${input.name}`
      )

    const { subgraphNode } = this

    // Link goes up and out of this subgraph
    if (subgraphNode && link.originIsIoNode) {
      const subgraphNodeInput = subgraphNode.inputs.at(link.origin_slot)
      if (!subgraphNodeInput)
        throw new SlotIndexError(
          `No input found for slot [${link.origin_slot}] ${input.name}`
        )

      // Nothing connected
      const linkId = subgraphNodeInput.link
      if (linkId == null) {
        const widget = subgraphNode.getWidgetFromSlot(subgraphNodeInput)
        if (!widget) return

        // Special case: SubgraphNode widget.
        return {
          node: this,
          origin_id: this.id,
          origin_slot: -1,
          widgetInfo: { value: widget.value }
        }
      }

      if (!subgraphNode.graph)
        throw new NullGraphError(
          `SubgraphNode ${subgraphNode.id} has no graph during input resolution`
        )
      const outerLink = subgraphNode.graph.getLink(linkId)
      if (!outerLink)
        throw new InvalidLinkError(
          `No outer link found for slot [${link.origin_slot}] ${input.name}`
        )

      const subgraphNodeExecutionId = this.subgraphNodePath.join(':')
      const subgraphNodeDto = this.nodesByExecutionId.get(
        subgraphNodeExecutionId
      )
      if (!subgraphNodeDto)
        throw new Error(
          `No subgraph node DTO found for id [${subgraphNodeExecutionId}]`
        )

      return subgraphNodeDto.resolveInput(outerLink.target_slot, visited)
    }

    // Not part of a subgraph; use the original link
    const outputNode = this.graph.getNodeById(link.origin_id)
    if (!outputNode)
      throw new InvalidLinkError(
        `No input node found for id [${this.id}] slot [${slot}] ${input.name}`
      )

    const outputNodeExecutionId = [
      ...this.subgraphNodePath,
      outputNode.id
    ].join(':')
    const outputNodeDto = this.nodesByExecutionId.get(outputNodeExecutionId)
    if (!outputNodeDto)
      throw new Error(
        `No output node DTO found for id [${outputNodeExecutionId}]`
      )

    return outputNodeDto.resolveOutput(
      link.origin_slot,
      type ?? input.type,
      visited
    )
  }

  /**
   * Determines whether this output is a valid endpoint for a link (non-virtual, non-bypass).
   * @param slot The slot index of the output.
   * @param type The type of the input
   * @param visited A set of unique IDs to guard against infinite recursion. See {@link resolveInput}.
   * @returns The node and the origin ID / slot index of the output.
   */
  resolveOutput(
    slot: number,
    type: ISlotType,
    visited: Set<string>
  ): ResolvedInput | undefined {
    const uniqueId = `${this.subgraphNode?.subgraph.id}:${this.node.id}[O]${slot}`
    if (visited.has(uniqueId)) {
      const nodeInfo = `${this.node.id}${this.node.title ? ` (${this.node.title})` : ''}`
      const pathInfo =
        this.subgraphNodePath.length > 0
          ? ` at path ${this.subgraphNodePath.join(':')}`
          : ''
      throw new RecursionError(
        `Circular reference detected while resolving output ${slot} of node ${nodeInfo}${pathInfo}. ` +
          `This creates an infinite loop in link resolution. UniqueID: [${uniqueId}]`
      )
    }
    visited.add(uniqueId)

    // Upstreamed: Bypass nodes are bypassed using the first input with matching type
    if (this.mode === LGraphEventMode.BYPASS) {
      // Bypass nodes by finding first input with matching type
      const matchingIndex = this._getBypassSlotIndex(slot, type)

      // No input types match - bypass not possible
      if (matchingIndex === -1) {
        console.warn(
          `[ExecutableNodeDTO.resolveOutput] No input types match type [${type}] for id [${this.id}] slot [${slot}]`,
          this
        )
        return
      }

      return this.resolveInput(matchingIndex, visited)
    }

    const { node } = this
    if (node.isSubgraphNode())
      return this._resolveSubgraphOutput(slot, type, visited)

    if (node.isVirtualNode) {
      const virtualLink = this.node.getInputLink(slot)
      if (virtualLink) {
        const { inputNode } = virtualLink.resolve(this.graph)
        if (!inputNode)
          throw new InvalidLinkError(
            `Virtual node failed to resolve parent [${this.id}] slot [${slot}]`
          )

        const inputNodeExecutionId = [
          ...this.subgraphNodePath,
          inputNode.id
        ].join(':')
        const inputNodeDto = this.nodesByExecutionId.get(inputNodeExecutionId)
        if (!inputNodeDto)
          throw new Error(`No input node DTO found for id [${inputNode.id}]`)

        return inputNodeDto.resolveInput(virtualLink.target_slot, visited, type)
      }

      // Virtual nodes without a matching input should be discarded.
      return
    }

    return {
      node: this,
      origin_id: this.id,
      origin_slot: slot
    }
  }

  /**
   * Finds the index of the input slot on this node that matches the given output {@link slot} index.
   * Used when bypassing nodes.
   * @param slot The output slot index on this node
   * @param type The type of the final target input (so type list matches are accurate)
   * @returns The index of the input slot on this node, otherwise `-1`.
   */
  private _getBypassSlotIndex(slot: number, type: ISlotType) {
    const { inputs } = this
    const oppositeInput = inputs[slot]
    const outputType = this.node.outputs[slot].type

    // Any type short circuit - match slot ID, fallback to first slot
    if (type === '*' || type === '') {
      return inputs.length > slot ? slot : 0
    }

    // Prefer input with the same slot ID
    if (
      oppositeInput &&
      LiteGraph.isValidConnection(oppositeInput.type, outputType) &&
      LiteGraph.isValidConnection(oppositeInput.type, type)
    ) {
      return slot
    }

    // Preserve legacy behaviour; use exact match first.
    const exactMatch = inputs.findIndex((input) => input.type === type)
    if (exactMatch !== -1) return exactMatch

    // Find first matching slot - prefer exact type
    return inputs.findIndex(
      (input) =>
        LiteGraph.isValidConnection(input.type, outputType) &&
        LiteGraph.isValidConnection(input.type, type)
    )
  }

  /**
   * Resolves the link inside a subgraph node, from the subgraph IO node to the node inside the subgraph.
   * @param slot The slot index of the output on the subgraph node.
   * @param visited A set of unique IDs to guard against infinite recursion. See {@link resolveInput}.
   * @returns A DTO for the node, and the origin ID / slot index of the output.
   */
  private _resolveSubgraphOutput(
    slot: number,
    type: ISlotType,
    visited: Set<string>
  ): ResolvedInput | undefined {
    const { node } = this
    const output = node.outputs.at(slot)

    if (!output)
      throw new SlotIndexError(
        `No output found for flattened id [${this.id}] slot [${slot}]`
      )
    if (!node.isSubgraphNode())
      throw new TypeError(`Node is not a subgraph node: ${node.id}`)

    // Link inside the subgraph
    const innerResolved = node.resolveSubgraphOutputLink(slot)
    if (!innerResolved) return

    const innerNode = innerResolved.outputNode
    if (!innerNode)
      throw new Error(
        `No output node found for id [${this.id}] slot [${slot}] ${output.name}`
      )

    // Recurse into the subgraph
    const innerNodeExecutionId = [
      ...this.subgraphNodePath,
      node.id,
      innerNode.id
    ].join(':')
    const innerNodeDto = this.nodesByExecutionId.get(innerNodeExecutionId)
    if (!innerNodeDto)
      throw new Error(
        `No inner node DTO found for id [${innerNodeExecutionId}]`
      )

    return innerNodeDto.resolveOutput(
      innerResolved.link.origin_slot,
      type,
      visited
    )
  }
}
