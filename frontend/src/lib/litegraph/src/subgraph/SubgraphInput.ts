import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import { LLink } from '@/lib/litegraph/src/LLink'
import type { RerouteId } from '@/lib/litegraph/src/Reroute'
import { CustomEventTarget } from '@/lib/litegraph/src/infrastructure/CustomEventTarget'
import type { SubgraphInputEventMap } from '@/lib/litegraph/src/infrastructure/SubgraphInputEventMap'
import type {
  INodeInputSlot,
  INodeOutputSlot,
  Point,
  ReadOnlyRect
} from '@/lib/litegraph/src/interfaces'
import { LiteGraph } from '@/lib/litegraph/src/litegraph'
import { NodeSlotType } from '@/lib/litegraph/src/types/globalEnums'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'

import type { SubgraphInputNode } from './SubgraphInputNode'
import type { SubgraphOutput } from './SubgraphOutput'
import { SubgraphSlot } from './SubgraphSlotBase'
import { isNodeSlot, isSubgraphOutput } from './subgraphUtils'

/**
 * An input "slot" from a parent graph into a subgraph.
 *
 * IMPORTANT: A subgraph "input" is both an input AND an output.  It creates an extra link connection point between
 * a parent graph and a subgraph, so is conceptually similar to a reroute.
 *
 * This can be a little confusing, but is easier to visualise when imagining editing a subgraph.
 * You have "Subgraph Inputs", because they are coming into the subgraph, which then connect to "node inputs".
 *
 * Functionally, however, when editing a subgraph, that "subgraph input" is the "origin" or "output side" of a link.
 */
export class SubgraphInput extends SubgraphSlot {
  declare parent: SubgraphInputNode

  events = new CustomEventTarget<SubgraphInputEventMap>()

  /** The linked widget that this slot is connected to. */
  private _widgetRef?: WeakRef<IBaseWidget>

  get _widget() {
    return this._widgetRef?.deref()
  }

  set _widget(widget) {
    this._widgetRef = widget ? new WeakRef(widget) : undefined
  }

  override connect(
    slot: INodeInputSlot,
    node: LGraphNode,
    afterRerouteId?: RerouteId
  ): LLink | undefined {
    const { subgraph } = this.parent

    // Allow nodes to block connection
    const inputIndex = node.inputs.indexOf(slot)
    if (
      node.onConnectInput?.(inputIndex, this.type, this, this.parent, -1) ===
      false
    )
      return

    // if (slot instanceof SubgraphOutput) {
    //   // Subgraph IO nodes have no special handling at present.
    //   return new LLink(
    //     ++subgraph.state.lastLinkId,
    //     this.type,
    //     this.parent.id,
    //     this.parent.slots.indexOf(this),
    //     node.id,
    //     inputIndex,
    //     afterRerouteId,
    //   )
    // }

    // Disconnect target input, if it is already connected.
    if (slot.link != null) {
      subgraph.beforeChange()
      const link = subgraph.getLink(slot.link)
      this.parent._disconnectNodeInput(node, slot, link)
    }

    const inputWidget = node.getWidgetFromSlot(slot)
    if (inputWidget) {
      if (!this.matchesWidget(inputWidget)) {
        console.warn('Target input has invalid widget.', slot, node)
        return
      }

      this._widget ??= inputWidget
      this.events.dispatch('input-connected', {
        input: slot,
        widget: inputWidget,
        node
      })
    }

    const link = new LLink(
      ++subgraph.state.lastLinkId,
      slot.type,
      this.parent.id,
      this.parent.slots.indexOf(this),
      node.id,
      inputIndex,
      afterRerouteId
    )

    // Add to graph links list
    subgraph._links.set(link.id, link)

    // Set link ID in each slot
    this.linkIds.push(link.id)
    slot.link = link.id

    // Reroutes
    const reroutes = LLink.getReroutes(subgraph, link)
    for (const reroute of reroutes) {
      reroute.linkIds.add(link.id)
      if (reroute.floating) delete reroute.floating
      reroute._dragging = undefined
    }

    // If this is the terminus of a floating link, remove it
    const lastReroute = reroutes.at(-1)
    if (lastReroute) {
      for (const linkId of lastReroute.floatingLinkIds) {
        const link = subgraph.floatingLinks.get(linkId)
        if (link?.parentId === lastReroute.id) {
          subgraph.removeFloatingLink(link)
        }
      }
    }
    subgraph._version++

    node.onConnectionsChange?.(NodeSlotType.INPUT, inputIndex, true, link, slot)

    subgraph.afterChange()

    return link
  }

  get labelPos(): Point {
    const [x, y, , height] = this.boundingRect
    return [x, y + height * 0.5]
  }

  getConnectedWidgets(): IBaseWidget[] {
    const { subgraph } = this.parent
    const widgets: IBaseWidget[] = []

    for (const linkId of this.linkIds) {
      const link = subgraph.getLink(linkId)
      if (!link) {
        console.error('Link not found', linkId)
        continue
      }

      const resolved = link.resolve(subgraph)
      if (resolved.input && resolved.inputNode?.widgets) {
        // Has no widget
        const widgetNamePojo = resolved.input.widget
        if (!widgetNamePojo) continue

        // Invalid widget name
        if (!widgetNamePojo.name) {
          console.warn('Invalid widget name', widgetNamePojo)
          continue
        }

        const widget = resolved.inputNode.widgets.find(
          (w) => w.name === widgetNamePojo.name
        )
        if (!widget) {
          console.warn('Widget not found', widgetNamePojo)
          continue
        }

        widgets.push(widget)
      }
    }
    return widgets
  }

  /**
   * Validates that the connection between the new slot and the existing widget is valid.
   * Used to prevent connections between widgets that are not of the same type.
   * @param otherWidget The widget to compare to.
   * @returns `true` if the connection is valid, otherwise `false`.
   */
  matchesWidget(otherWidget: IBaseWidget): boolean {
    const widget = this._widgetRef?.deref()
    if (!widget) return true

    if (
      otherWidget.type !== widget.type ||
      otherWidget.options.min !== widget.options.min ||
      otherWidget.options.max !== widget.options.max ||
      otherWidget.options.step !== widget.options.step ||
      otherWidget.options.step2 !== widget.options.step2 ||
      otherWidget.options.precision !== widget.options.precision
    ) {
      return false
    }

    return true
  }

  override disconnect(): void {
    super.disconnect()

    this.events.dispatch('input-disconnected', { input: this })
  }

  /** For inputs, x is the right edge of the input node. */
  override arrange(rect: ReadOnlyRect): void {
    const [right, top, width, height] = rect
    const { boundingRect: b, pos } = this

    b[0] = right - width
    b[1] = top
    b[2] = width
    b[3] = height

    pos[0] = right - height * 0.5
    pos[1] = top + height * 0.5
  }

  /**
   * Checks if this slot is a valid target for a connection from the given slot.
   * For SubgraphInput (which acts as an output inside the subgraph),
   * the fromSlot should be an input slot.
   */
  override isValidTarget(
    fromSlot: INodeInputSlot | INodeOutputSlot | SubgraphInput | SubgraphOutput
  ): boolean {
    if (isNodeSlot(fromSlot)) {
      return (
        'link' in fromSlot &&
        LiteGraph.isValidConnection(this.type, fromSlot.type)
      )
    }

    if (isSubgraphOutput(fromSlot)) {
      return LiteGraph.isValidConnection(this.type, fromSlot.type)
    }

    return false
  }
}
