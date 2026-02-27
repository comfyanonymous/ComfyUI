import type { CanvasPointer } from '@/lib/litegraph/src/CanvasPointer'
import type { LGraphNode, NodeId } from '@/lib/litegraph/src/LGraphNode'
import type { LLink } from '@/lib/litegraph/src/LLink'
import type { RerouteId } from '@/lib/litegraph/src/Reroute'
import type { LinkConnector } from '@/lib/litegraph/src/canvas/LinkConnector'
import { SUBGRAPH_OUTPUT_ID } from '@/lib/litegraph/src/constants'
import type {
  DefaultConnectionColors,
  INodeInputSlot,
  INodeOutputSlot,
  ISlotType,
  Positionable
} from '@/lib/litegraph/src/interfaces'
import type { NodeLike } from '@/lib/litegraph/src/types/NodeLike'
import type { CanvasPointerEvent } from '@/lib/litegraph/src/types/events'
import type { SubgraphIO } from '@/lib/litegraph/src/types/serialisation'
import { findFreeSlotOfType } from '@/lib/litegraph/src/utils/collections'

import { EmptySubgraphOutput } from './EmptySubgraphOutput'
import { SubgraphIONodeBase } from './SubgraphIONodeBase'
import type { SubgraphInput } from './SubgraphInput'
import type { SubgraphOutput } from './SubgraphOutput'

export class SubgraphOutputNode
  extends SubgraphIONodeBase<SubgraphOutput>
  implements Positionable
{
  readonly id: NodeId = SUBGRAPH_OUTPUT_ID

  readonly emptySlot: EmptySubgraphOutput = new EmptySubgraphOutput(this)

  get slots() {
    return this.subgraph.outputs
  }

  override get allSlots(): SubgraphOutput[] {
    return [...this.slots, this.emptySlot]
  }

  get slotAnchorX() {
    const [x] = this.boundingRect
    return x + SubgraphIONodeBase.roundedRadius
  }

  override onPointerDown(
    e: CanvasPointerEvent,
    pointer: CanvasPointer,
    linkConnector: LinkConnector
  ): void {
    // Left-click handling for dragging connections
    if (e.button === 0) {
      for (const slot of this.allSlots) {
        // Check if click is within the full slot area (including label)
        if (slot.boundingRect.containsXy(e.canvasX, e.canvasY)) {
          pointer.onDragStart = () => {
            linkConnector.dragNewFromSubgraphOutput(this.subgraph, this, slot)
          }
          pointer.onDragEnd = (eUp) => {
            linkConnector.dropLinks(this.subgraph, eUp)
          }
          pointer.onDoubleClick = () => {
            this.handleSlotDoubleClick(slot, e)
          }
          pointer.finally = () => {
            linkConnector.reset(true)
          }
        }
      }
      // Check for right-click
    } else if (e.button === 2) {
      const slot = this.getSlotInPosition(e.canvasX, e.canvasY)
      if (slot) this.showSlotContextMenu(slot, e)
    }
  }

  /** @inheritdoc */
  override renameSlot(slot: SubgraphOutput, name: string): void {
    this.subgraph.renameOutput(slot, name)
  }

  /** @inheritdoc */
  override removeSlot(slot: SubgraphOutput): void {
    this.subgraph.removeOutput(slot)
  }

  canConnectTo(
    outputNode: NodeLike,
    fromSlot: SubgraphOutput,
    output: INodeOutputSlot | SubgraphIO
  ): boolean {
    return outputNode.canConnectTo(this, fromSlot, output)
  }

  connectByTypeOutput(
    slot: number,
    target_node: LGraphNode,
    target_slotType: ISlotType,
    optsIn?: { afterRerouteId?: RerouteId }
  ): LLink | undefined {
    const outputSlot = target_node.findOutputByType(target_slotType)
    if (!outputSlot) return

    return this.slots[slot].connect(
      outputSlot.slot,
      target_node,
      optsIn?.afterRerouteId
    )
  }

  findInputByType(type: ISlotType): SubgraphOutput | undefined {
    return findFreeSlotOfType(
      this.slots,
      type,
      (slot) => slot.linkIds.length > 0
    )?.slot
  }

  override drawProtected(
    ctx: CanvasRenderingContext2D,
    colorContext: DefaultConnectionColors,
    fromSlot?:
      | INodeInputSlot
      | INodeOutputSlot
      | SubgraphInput
      | SubgraphOutput,
    editorAlpha?: number
  ): void {
    const { roundedRadius } = SubgraphIONodeBase
    const transform = ctx.getTransform()

    const [x, y, , height] = this.boundingRect
    ctx.translate(x, y)

    // Draw bottom rounded part
    ctx.strokeStyle = this.sideStrokeStyle
    ctx.lineWidth = this.sideLineWidth
    ctx.beginPath()
    ctx.arc(roundedRadius, roundedRadius, roundedRadius, Math.PI, Math.PI * 1.5)

    // Straight line to bottom
    ctx.moveTo(0, roundedRadius)
    ctx.lineTo(0, height - roundedRadius)

    // Bottom rounded part
    ctx.arc(
      roundedRadius,
      height - roundedRadius,
      roundedRadius,
      Math.PI,
      Math.PI * 0.5,
      true
    )
    ctx.stroke()

    // Restore context
    ctx.setTransform(transform)

    this.drawSlots(ctx, colorContext, fromSlot, editorAlpha)
  }
}
