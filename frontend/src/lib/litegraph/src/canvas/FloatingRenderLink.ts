import type { LGraphNode, NodeId } from '@/lib/litegraph/src/LGraphNode'
import type { LLink } from '@/lib/litegraph/src/LLink'
import type { Reroute } from '@/lib/litegraph/src/Reroute'
import {
  SUBGRAPH_INPUT_ID,
  SUBGRAPH_OUTPUT_ID
} from '@/lib/litegraph/src/constants'
import type { CustomEventTarget } from '@/lib/litegraph/src/infrastructure/CustomEventTarget'
import type { LinkConnectorEventMap } from '@/lib/litegraph/src/infrastructure/LinkConnectorEventMap'
import type {
  INodeInputSlot,
  INodeOutputSlot,
  LinkNetwork,
  Point
} from '@/lib/litegraph/src/interfaces'
import type { SubgraphInput } from '@/lib/litegraph/src/subgraph/SubgraphInput'
import type { SubgraphOutput } from '@/lib/litegraph/src/subgraph/SubgraphOutput'
import { LinkDirection } from '@/lib/litegraph/src/types/globalEnums'

import type { RenderLink } from './RenderLink'

/**
 * Represents a floating link that is currently being dragged from one slot to another.
 *
 * This is a heavier, but short-lived convenience data structure. All refs to FloatingRenderLinks should be discarded on drop.
 * @remarks
 * At time of writing, Litegraph is using several different styles and methods to handle link dragging.
 *
 * Once the library has undergone more substantial changes to the way links are managed,
 * many properties of this class will be superfluous and removable.
 */
export class FloatingRenderLink implements RenderLink {
  readonly node: LGraphNode
  readonly fromSlot: INodeOutputSlot | INodeInputSlot
  readonly fromPos: Point
  readonly fromDirection: LinkDirection
  readonly fromSlotIndex: number

  readonly outputNodeId: NodeId = -1
  readonly outputNode?: LGraphNode
  readonly outputSlot?: INodeOutputSlot
  readonly outputIndex: number = -1
  readonly outputPos?: Point

  readonly inputNodeId: NodeId = -1
  readonly inputNode?: LGraphNode
  readonly inputSlot?: INodeInputSlot
  readonly inputIndex: number = -1
  readonly inputPos?: Point

  constructor(
    readonly network: LinkNetwork,
    readonly link: LLink,
    readonly toType: 'input' | 'output',
    readonly fromReroute: Reroute,
    readonly dragDirection: LinkDirection = LinkDirection.CENTER
  ) {
    const {
      origin_id: outputNodeId,
      target_id: inputNodeId,
      origin_slot: outputIndex,
      target_slot: inputIndex
    } = link

    if (outputNodeId !== -1) {
      // Output connected
      const outputNode = network.getNodeById(outputNodeId) ?? undefined
      if (!outputNode)
        throw new Error(
          `Creating DraggingRenderLink for link [${link.id}] failed: Output node [${outputNodeId}] not found.`
        )

      const outputSlot = outputNode?.outputs.at(outputIndex)
      if (!outputSlot)
        throw new Error(
          `Creating DraggingRenderLink for link [${link.id}] failed: Output slot [${outputIndex}] not found.`
        )

      this.outputNodeId = outputNodeId
      this.outputNode = outputNode
      this.outputSlot = outputSlot
      this.outputIndex = outputIndex
      this.outputPos = outputNode.getOutputPos(outputIndex)

      // RenderLink props
      this.node = outputNode
      this.fromSlot = outputSlot
      this.fromPos = fromReroute?.pos ?? this.outputPos
      this.fromDirection = LinkDirection.LEFT
      this.dragDirection = LinkDirection.RIGHT
      this.fromSlotIndex = outputIndex
    } else {
      // Input connected
      const inputNode = network.getNodeById(inputNodeId) ?? undefined
      if (!inputNode)
        throw new Error(
          `Creating DraggingRenderLink for link [${link.id}] failed: Input node [${inputNodeId}] not found.`
        )

      const inputSlot = inputNode?.inputs.at(inputIndex)
      if (!inputSlot)
        throw new Error(
          `Creating DraggingRenderLink for link [${link.id}] failed: Input slot [${inputIndex}] not found.`
        )

      this.inputNodeId = inputNodeId
      this.inputNode = inputNode
      this.inputSlot = inputSlot
      this.inputIndex = inputIndex
      this.inputPos = inputNode.getInputPos(inputIndex)

      // RenderLink props
      this.node = inputNode
      this.fromSlot = inputSlot
      this.fromDirection = LinkDirection.RIGHT
      this.fromSlotIndex = inputIndex
    }
    this.fromPos = fromReroute.pos
  }

  canConnectToInput(): boolean {
    return this.toType === 'input'
  }

  canConnectToOutput(): boolean {
    return this.toType === 'output'
  }

  canConnectToReroute(reroute: Reroute): boolean {
    if (this.toType === 'input') {
      if (reroute.origin_id === this.inputNode?.id) return false
    } else {
      if (reroute.origin_id === this.outputNode?.id) return false
    }
    return true
  }

  canConnectToSubgraphInput(input: SubgraphInput): boolean {
    return this.toType === 'output' && input.isValidTarget(this.fromSlot)
  }

  connectToInput(
    node: LGraphNode,
    input: INodeInputSlot,
    _events?: CustomEventTarget<LinkConnectorEventMap>
  ): void {
    const floatingLink = this.link
    floatingLink.target_id = node.id
    floatingLink.target_slot = node.inputs.indexOf(input)

    node.disconnectInput(node.inputs.indexOf(input))

    this.fromSlot._floatingLinks?.delete(floatingLink)
    input._floatingLinks ??= new Set()
    input._floatingLinks.add(floatingLink)
  }

  connectToOutput(
    node: LGraphNode,
    output: INodeOutputSlot,
    _events?: CustomEventTarget<LinkConnectorEventMap>
  ): void {
    const floatingLink = this.link
    floatingLink.origin_id = node.id
    floatingLink.origin_slot = node.outputs.indexOf(output)

    this.fromSlot._floatingLinks?.delete(floatingLink)
    output._floatingLinks ??= new Set()
    output._floatingLinks.add(floatingLink)
  }

  connectToSubgraphInput(
    input: SubgraphInput,
    _events?: CustomEventTarget<LinkConnectorEventMap>
  ): void {
    const floatingLink = this.link
    floatingLink.origin_id = SUBGRAPH_INPUT_ID
    floatingLink.origin_slot = input.parent.slots.indexOf(input)

    this.fromSlot._floatingLinks?.delete(floatingLink)
    input._floatingLinks ??= new Set()
    input._floatingLinks.add(floatingLink)
  }

  connectToSubgraphOutput(
    output: SubgraphOutput,
    _events?: CustomEventTarget<LinkConnectorEventMap>
  ): void {
    const floatingLink = this.link
    floatingLink.origin_id = SUBGRAPH_OUTPUT_ID
    floatingLink.origin_slot = output.parent.slots.indexOf(output)

    this.fromSlot._floatingLinks?.delete(floatingLink)
    output._floatingLinks ??= new Set()
    output._floatingLinks.add(floatingLink)
  }

  connectToRerouteInput(
    // @ts-expect-error - Reroute type needs fixing
    reroute: Reroute,
    { node: inputNode, input }: { node: LGraphNode; input: INodeInputSlot },
    events: CustomEventTarget<LinkConnectorEventMap>
  ) {
    const floatingLink = this.link
    floatingLink.target_id = inputNode.id
    floatingLink.target_slot = inputNode.inputs.indexOf(input)

    this.fromSlot._floatingLinks?.delete(floatingLink)
    input._floatingLinks ??= new Set()
    input._floatingLinks.add(floatingLink)

    events.dispatch('input-moved', this)
  }

  connectToRerouteOutput(
    // @ts-expect-error - Reroute type needs fixing
    reroute: Reroute,
    outputNode: LGraphNode,
    output: INodeOutputSlot,
    events: CustomEventTarget<LinkConnectorEventMap>
  ) {
    const floatingLink = this.link
    floatingLink.origin_id = outputNode.id
    floatingLink.origin_slot = outputNode.outputs.indexOf(output)

    this.fromSlot._floatingLinks?.delete(floatingLink)
    output._floatingLinks ??= new Set()
    output._floatingLinks.add(floatingLink)

    events.dispatch('output-moved', this)
  }
}
