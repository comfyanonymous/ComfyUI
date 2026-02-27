import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { LLink } from '@/lib/litegraph/src/LLink'
import type { Reroute } from '@/lib/litegraph/src/Reroute'
import type { CustomEventTarget } from '@/lib/litegraph/src/infrastructure/CustomEventTarget'
import type { LinkConnectorEventMap } from '@/lib/litegraph/src/infrastructure/LinkConnectorEventMap'
import type {
  INodeInputSlot,
  LinkNetwork,
  Point
} from '@/lib/litegraph/src/interfaces'
import type { SubgraphInput } from '@/lib/litegraph/src/subgraph/SubgraphInput'
import type { SubgraphInputNode } from '@/lib/litegraph/src/subgraph/SubgraphInputNode'
import type { NodeLike } from '@/lib/litegraph/src/types/NodeLike'
import { LinkDirection } from '@/lib/litegraph/src/types/globalEnums'

import type { RenderLink } from './RenderLink'

/** Connecting TO an input slot. */

export class ToInputFromIoNodeLink implements RenderLink {
  readonly toType = 'input'
  readonly fromSlotIndex: number
  readonly fromPos: Point
  fromDirection: LinkDirection = LinkDirection.RIGHT
  readonly existingLink?: LLink

  constructor(
    readonly network: LinkNetwork,
    readonly node: SubgraphInputNode,
    readonly fromSlot: SubgraphInput,
    readonly fromReroute?: Reroute,
    public dragDirection: LinkDirection = LinkDirection.CENTER,
    existingLink?: LLink
  ) {
    const outputIndex = node.slots.indexOf(fromSlot)
    if (outputIndex === -1 && fromSlot !== node.emptySlot) {
      throw new Error(
        `Creating render link for node [${this.node.id}] failed: Slot index not found.`
      )
    }

    this.fromSlotIndex = outputIndex
    this.fromPos = fromReroute ? fromReroute.pos : fromSlot.pos
    this.existingLink = existingLink
  }

  canConnectToInput(inputNode: NodeLike, input: INodeInputSlot): boolean {
    return this.node.canConnectTo(inputNode, input, this.fromSlot)
  }

  canConnectToOutput(): false {
    return false
  }

  connectToInput(
    node: LGraphNode,
    input: INodeInputSlot,
    events: CustomEventTarget<LinkConnectorEventMap>
  ) {
    const { fromSlot, fromReroute, existingLink } = this
    if (
      existingLink &&
      node.id === existingLink.target_id &&
      node.inputs[existingLink.target_slot] === input
    )
      return

    const newLink = fromSlot.connect(input, node, fromReroute?.id)

    if (existingLink) {
      // Moving an existing link
      const { input, inputNode } = existingLink.resolve(this.network)
      if (inputNode && input)
        this.node._disconnectNodeInput(inputNode, input, existingLink)
      events.dispatch('input-moved', this)
    } else {
      // Creating a new link
      events.dispatch('link-created', newLink)
    }
  }

  connectToSubgraphOutput(): void {
    throw new Error('Not implemented')
  }

  connectToRerouteInput(
    reroute: Reroute,
    {
      node: inputNode,
      input,
      link
    }: { node: LGraphNode; input: INodeInputSlot; link: LLink },
    events: CustomEventTarget<LinkConnectorEventMap>,
    originalReroutes: Reroute[]
  ) {
    const { fromSlot, fromReroute } = this

    // Check before creating new link overwrites the value
    const floatingTerminus = fromReroute?.floating?.slotType === 'output'

    // Set the parentId of the reroute we dropped on, to the reroute we dragged from
    reroute.parentId = fromReroute?.id

    const newLink = fromSlot.connect(input, inputNode, link.parentId)

    // Connecting from the final reroute of a floating reroute chain
    if (floatingTerminus) fromReroute.removeAllFloatingLinks()

    // Clean up reroutes
    for (const reroute of originalReroutes) {
      if (reroute.id === fromReroute?.id) break

      reroute.removeLink(link)
      if (reroute.totalLinks === 0) {
        if (link.isFloating) {
          // Cannot float from both sides - remove
          reroute.remove()
        } else {
          // Convert to floating
          const cl = link.toFloating('output', reroute.id)
          this.network.addFloatingLink(cl)
          reroute.floating = { slotType: 'output' }
        }
      }
    }

    if (this.existingLink) {
      // Moving an existing link
      events.dispatch('input-moved', this)
    } else {
      // Creating a new link
      events.dispatch('link-created', newLink)
    }
  }

  connectToOutput() {
    throw new Error('ToInputRenderLink cannot connect to an output.')
  }

  connectToSubgraphInput(): void {
    throw new Error('ToInputRenderLink cannot connect to a subgraph input.')
  }

  connectToRerouteOutput() {
    throw new Error('ToInputRenderLink cannot connect to an output.')
  }
  disconnect(): boolean {
    if (!this.existingLink) return false
    const { input, inputNode } = this.existingLink.resolve(this.network)
    if (!inputNode || !input) return false
    this.node._disconnectNodeInput(inputNode, input, this.existingLink)
    return true
  }
}
