import type { CanvasPointer } from '@/lib/litegraph/src/CanvasPointer'
import type { LGraphNode, NodeId } from '@/lib/litegraph/src/LGraphNode'
import { LLink } from '@/lib/litegraph/src/LLink'
import type { RerouteId } from '@/lib/litegraph/src/Reroute'
import type { LinkConnector } from '@/lib/litegraph/src/canvas/LinkConnector'
import { SUBGRAPH_INPUT_ID } from '@/lib/litegraph/src/constants'
import type {
  DefaultConnectionColors,
  INodeInputSlot,
  INodeOutputSlot,
  ISlotType,
  Positionable
} from '@/lib/litegraph/src/interfaces'
import type { NodeLike } from '@/lib/litegraph/src/types/NodeLike'
import type { CanvasPointerEvent } from '@/lib/litegraph/src/types/events'
import { NodeSlotType } from '@/lib/litegraph/src/types/globalEnums'
import { findFreeSlotOfType } from '@/lib/litegraph/src/utils/collections'

import { EmptySubgraphInput } from './EmptySubgraphInput'
import { SubgraphIONodeBase } from './SubgraphIONodeBase'
import type { SubgraphInput } from './SubgraphInput'
import type { SubgraphOutput } from './SubgraphOutput'

export class SubgraphInputNode
  extends SubgraphIONodeBase<SubgraphInput>
  implements Positionable
{
  readonly id: NodeId = SUBGRAPH_INPUT_ID

  readonly emptySlot: EmptySubgraphInput = new EmptySubgraphInput(this)

  get slots() {
    return this.subgraph.inputs
  }

  override get allSlots(): SubgraphInput[] {
    return [...this.slots, this.emptySlot]
  }

  get slotAnchorX() {
    const [x, , width] = this.boundingRect
    return x + width - SubgraphIONodeBase.roundedRadius
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
            linkConnector.dragNewFromSubgraphInput(this.subgraph, this, slot)
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
  override renameSlot(slot: SubgraphInput, name: string): void {
    this.subgraph.renameInput(slot, name)
  }

  /** @inheritdoc */
  override removeSlot(slot: SubgraphInput): void {
    this.subgraph.removeInput(slot)
  }

  canConnectTo(
    inputNode: NodeLike,
    input: INodeInputSlot,
    fromSlot: SubgraphInput
  ): boolean {
    return inputNode.canConnectTo(this, input, fromSlot)
  }

  connectSlots(
    fromSlot: SubgraphInput,
    inputNode: LGraphNode,
    input: INodeInputSlot,
    afterRerouteId: RerouteId | undefined
  ): LLink {
    const { subgraph } = this

    const outputIndex = this.slots.indexOf(fromSlot)
    const inputIndex = inputNode.inputs.indexOf(input)

    if (outputIndex === -1 || inputIndex === -1)
      throw new Error('Invalid slot indices.')

    return new LLink(
      ++subgraph.state.lastLinkId,
      input.type || fromSlot.type,
      this.id,
      outputIndex,
      inputNode.id,
      inputIndex,
      afterRerouteId
    )
  }

  // #region Legacy LGraphNode compatibility

  connectByType(
    slot: number,
    target_node: LGraphNode,
    target_slotType: ISlotType,
    optsIn?: { afterRerouteId?: RerouteId }
  ): LLink | undefined {
    const inputSlot = target_node.findInputByType(target_slotType)
    if (!inputSlot) return

    if (slot === -1) {
      // This indicates a connection is being made from the "Empty" slot.
      // We need to create a new, concrete input on the subgraph that matches the target.
      const newSubgraphInput = this.subgraph.addInput(
        inputSlot.slot.name,
        String(inputSlot.slot.type ?? '')
      )
      const newSlotIndex = this.slots.indexOf(newSubgraphInput)
      if (newSlotIndex === -1) {
        console.error('Could not find newly created subgraph input slot.')
        return
      }
      slot = newSlotIndex
    }

    return this.slots[slot].connect(
      inputSlot.slot,
      target_node,
      optsIn?.afterRerouteId
    )
  }

  findOutputSlot(name: string): SubgraphInput | undefined {
    return this.slots.find((output) => output.name === name)
  }

  findOutputByType(type: ISlotType): SubgraphInput | undefined {
    return findFreeSlotOfType(
      this.slots,
      type,
      (slot) => slot.linkIds.length > 0
    )?.slot
  }

  // #endregion Legacy LGraphNode compatibility

  _disconnectNodeInput(
    node: LGraphNode,
    input: INodeInputSlot,
    link: LLink | undefined
  ): void {
    const { subgraph } = this

    // Break floating links
    if (input._floatingLinks?.size) {
      for (const link of input._floatingLinks) {
        subgraph.removeFloatingLink(link)
      }
    }

    input.link = null
    subgraph.setDirtyCanvas(false, true)

    if (!link) return

    const subgraphInputIndex = link.origin_slot
    link.disconnect(subgraph, 'output')
    subgraph._version++

    const subgraphInput = this.slots.at(subgraphInputIndex)
    if (!subgraphInput) {
      console.warn(
        'disconnectNodeInput: subgraphInput not found',
        this,
        subgraphInputIndex
      )
      return
    }

    // search in the inputs list for this link
    const index = subgraphInput.linkIds.indexOf(link.id)
    if (index !== -1) {
      subgraphInput.linkIds.splice(index, 1)
    } else {
      console.warn(
        'disconnectNodeInput: link ID not found in subgraphInput linkIds',
        link.id
      )
    }
    const slotIndex = node.inputs.findIndex((inp) => inp === input)
    if (slotIndex !== -1) {
      node.onConnectionsChange?.(
        NodeSlotType.INPUT,
        slotIndex,
        false,
        link,
        subgraphInput
      )
    }
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

    const [x, y, width, height] = this.boundingRect
    ctx.translate(x, y)

    // Draw top rounded part
    ctx.strokeStyle = this.sideStrokeStyle
    ctx.lineWidth = this.sideLineWidth
    ctx.beginPath()
    ctx.arc(
      width - roundedRadius,
      roundedRadius,
      roundedRadius,
      Math.PI * 1.5,
      0
    )

    // Straight line to bottom
    ctx.moveTo(width, roundedRadius)
    ctx.lineTo(width, height - roundedRadius)

    // Bottom rounded part
    ctx.arc(
      width - roundedRadius,
      height - roundedRadius,
      roundedRadius,
      0,
      Math.PI * 0.5
    )
    ctx.stroke()

    // Restore context
    ctx.setTransform(transform)

    this.drawSlots(ctx, colorContext, fromSlot, editorAlpha)
  }
}
