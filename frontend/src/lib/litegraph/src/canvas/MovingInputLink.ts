import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import { LiteGraph } from '@/lib/litegraph/src/litegraph'
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
import type { SubgraphOutput } from '@/lib/litegraph/src/subgraph/SubgraphOutput'
import type { NodeLike } from '@/lib/litegraph/src/types/NodeLike'
import { LinkDirection } from '@/lib/litegraph/src/types/globalEnums'
import type { SubgraphIO } from '@/lib/litegraph/src/types/serialisation'

import { MovingLinkBase } from './MovingLinkBase'

export class MovingInputLink extends MovingLinkBase {
  override readonly toType = 'input'

  readonly node: LGraphNode
  readonly fromSlot: INodeOutputSlot
  readonly fromPos: Point
  readonly fromDirection: LinkDirection
  readonly fromSlotIndex: number
  disconnectOnDrop: boolean
  readonly disconnectOrigin: Point

  constructor(
    network: LinkNetwork,
    link: LLink,
    fromReroute?: Reroute,
    dragDirection: LinkDirection = LinkDirection.CENTER,
    startPoint?: Point
  ) {
    super(network, link, 'input', fromReroute, dragDirection)

    this.node = this.outputNode
    this.fromSlot = this.outputSlot
    this.fromPos = fromReroute?.pos ?? this.outputPos
    this.fromDirection = LinkDirection.NONE
    this.fromSlotIndex = this.outputIndex
    this.disconnectOnDrop = true
    this.disconnectOrigin = startPoint ?? this.inputPos
  }

  canConnectToInput(
    inputNode: NodeLike,
    input: INodeInputSlot | SubgraphIO
  ): boolean {
    return this.node.canConnectTo(inputNode, input, this.outputSlot)
  }

  canConnectToOutput(): false {
    return false
  }

  canConnectToReroute(reroute: Reroute): boolean {
    return reroute.origin_id !== this.inputNode.id
  }

  connectToInput(
    inputNode: LGraphNode,
    input: INodeInputSlot,
    events: CustomEventTarget<LinkConnectorEventMap>
  ): LLink | null | undefined {
    if (input === this.inputSlot) return

    this.inputNode.disconnectInput(this.inputIndex, true)
    const link = this.outputNode.connectSlots(
      this.outputSlot,
      inputNode,
      input,
      this.fromReroute?.id
    )
    if (link) events.dispatch('input-moved', this)
    return link
  }

  connectToOutput(): never {
    throw new Error('MovingInputLink cannot connect to an output.')
  }

  connectToSubgraphInput(): void {
    throw new Error('MovingInputLink cannot connect to a subgraph input.')
  }

  connectToSubgraphOutput(
    output: SubgraphOutput,
    events?: CustomEventTarget<LinkConnectorEventMap>
  ): void {
    const newLink = output.connect(
      this.fromSlot,
      this.node,
      this.fromReroute?.id
    )
    events?.dispatch('link-created', newLink)
  }

  connectToRerouteInput(
    reroute: Reroute,
    {
      node: inputNode,
      input,
      link: existingLink
    }: { node: LGraphNode; input: INodeInputSlot; link: LLink },
    events: CustomEventTarget<LinkConnectorEventMap>,
    originalReroutes: Reroute[]
  ): void {
    const { outputNode, outputSlot, fromReroute } = this

    // Clean up reroutes
    for (const reroute of originalReroutes) {
      if (reroute.id === this.link.parentId) break

      if (reroute.totalLinks === 1) reroute.remove()
    }
    // Set the parentId of the reroute we dropped on, to the reroute we dragged from
    reroute.parentId = fromReroute?.id

    const newLink = outputNode.connectSlots(
      outputSlot,
      inputNode,
      input,
      existingLink.parentId
    )
    if (newLink) events.dispatch('input-moved', this)
  }

  connectToRerouteOutput(): never {
    throw new Error('MovingInputLink cannot connect to an output.')
  }

  disconnect(): boolean {
    return this.inputNode.disconnectInput(this.inputIndex, true)
  }

  drawConnectionCircle(ctx: CanvasRenderingContext2D, to: Readonly<Point>) {
    if (!this.disconnectOnDrop) return

    const [originX, originY] = this.disconnectOrigin
    const radius = 35
    const distSquared = (originX - to[0]) ** 2 + (originY - to[1]) ** 2

    ctx.save()
    ctx.strokeStyle = LiteGraph.WIDGET_OUTLINE_COLOR
    ctx.lineWidth = 2
    ctx.beginPath()
    ctx.moveTo(originX + radius, originY)
    ctx.arc(originX, originY, radius, 0, Math.PI * 2)
    ctx.stroke()
    ctx.restore()

    this.disconnectOnDrop = distSquared < radius ** 2
  }
}
