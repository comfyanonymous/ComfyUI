import type { LGraphNode, NodeId } from '@/lib/litegraph/src/LGraphNode'
import type { LLink } from '@/lib/litegraph/src/LLink'
import type { Reroute } from '@/lib/litegraph/src/Reroute'
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
import type { NodeLike } from '@/lib/litegraph/src/types/NodeLike'
import { LinkDirection } from '@/lib/litegraph/src/types/globalEnums'

import type { RenderLink } from './RenderLink'

/**
 * Represents an existing link that is currently being dragged by the user from one slot to another.
 *
 * This is a heavier, but short-lived convenience data structure.
 * All refs to {@link MovingInputLink} and {@link MovingOutputLink} should be discarded on drop.
 * @remarks
 * At time of writing, Litegraph is using several different styles and methods to handle link dragging.
 *
 * Once the library has undergone more substantial changes to the way links are managed,
 * many properties of this class will be superfluous and removable.
 */

export abstract class MovingLinkBase implements RenderLink {
  abstract readonly node: LGraphNode
  abstract readonly fromSlot: INodeOutputSlot | INodeInputSlot
  abstract readonly fromPos: Point
  abstract readonly fromDirection: LinkDirection
  abstract readonly fromSlotIndex: number

  readonly outputNodeId: NodeId
  readonly outputNode: LGraphNode
  readonly outputSlot: INodeOutputSlot
  readonly outputIndex: number
  readonly outputPos: Point

  readonly inputNodeId: NodeId
  readonly inputNode: LGraphNode
  readonly inputSlot: INodeInputSlot
  readonly inputIndex: number
  readonly inputPos: Point

  constructor(
    readonly network: LinkNetwork,
    readonly link: LLink,
    readonly toType: 'input' | 'output',
    readonly fromReroute?: Reroute,
    readonly dragDirection: LinkDirection = LinkDirection.CENTER
  ) {
    const {
      origin_id: outputNodeId,
      target_id: inputNodeId,
      origin_slot: outputIndex,
      target_slot: inputIndex
    } = link

    // Store output info
    const outputNode = network.getNodeById(outputNodeId) ?? undefined
    if (!outputNode)
      throw new Error(
        `Creating MovingRenderLink for link [${link.id}] failed: Output node [${outputNodeId}] not found.`
      )

    const outputSlot = outputNode.outputs.at(outputIndex)
    if (!outputSlot)
      throw new Error(
        `Creating MovingRenderLink for link [${link.id}] failed: Output slot [${outputIndex}] not found.`
      )

    this.outputNodeId = outputNodeId
    this.outputNode = outputNode
    this.outputSlot = outputSlot
    this.outputIndex = outputIndex
    this.outputPos = outputNode.getOutputPos(outputIndex)

    // Store input info
    const inputNode = network.getNodeById(inputNodeId) ?? undefined
    if (!inputNode)
      throw new Error(
        `Creating DraggingRenderLink for link [${link.id}] failed: Input node [${inputNodeId}] not found.`
      )

    const inputSlot = inputNode.inputs.at(inputIndex)
    if (!inputSlot)
      throw new Error(
        `Creating DraggingRenderLink for link [${link.id}] failed: Input slot [${inputIndex}] not found.`
      )

    this.inputNodeId = inputNodeId
    this.inputNode = inputNode
    this.inputSlot = inputSlot
    this.inputIndex = inputIndex
    this.inputPos = inputNode.getInputPos(inputIndex)
  }

  abstract canConnectToInput(
    inputNode: NodeLike,
    input: INodeInputSlot
  ): boolean
  abstract canConnectToOutput(
    outputNode: NodeLike,
    output: INodeOutputSlot
  ): boolean
  abstract connectToInput(
    node: LGraphNode,
    input: INodeInputSlot,
    events?: CustomEventTarget<LinkConnectorEventMap>
  ): void
  abstract connectToOutput(
    node: LGraphNode,
    output: INodeOutputSlot,
    events?: CustomEventTarget<LinkConnectorEventMap>
  ): void
  abstract connectToSubgraphInput(
    input: SubgraphInput,
    events?: CustomEventTarget<LinkConnectorEventMap>
  ): void
  abstract connectToSubgraphOutput(
    output: SubgraphOutput,
    events?: CustomEventTarget<LinkConnectorEventMap>
  ): void
  abstract connectToRerouteInput(
    reroute: Reroute,
    {
      node,
      input,
      link
    }: { node: LGraphNode; input: INodeInputSlot; link: LLink },
    events: CustomEventTarget<LinkConnectorEventMap>,
    originalReroutes: Reroute[]
  ): void
  abstract connectToRerouteOutput(
    reroute: Reroute,
    outputNode: LGraphNode,
    output: INodeOutputSlot,
    events: CustomEventTarget<LinkConnectorEventMap>
  ): void

  abstract disconnect(): boolean
}
