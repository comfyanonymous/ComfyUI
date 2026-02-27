import { remove } from 'es-toolkit'

import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import { LLink } from '@/lib/litegraph/src/LLink'
import type { Reroute } from '@/lib/litegraph/src/Reroute'
import {
  SUBGRAPH_INPUT_ID,
  SUBGRAPH_OUTPUT_ID
} from '@/lib/litegraph/src/constants'
import { CustomEventTarget } from '@/lib/litegraph/src/infrastructure/CustomEventTarget'
import type { LinkConnectorEventMap } from '@/lib/litegraph/src/infrastructure/LinkConnectorEventMap'
import type {
  ConnectingLink,
  INodeInputSlot,
  INodeOutputSlot,
  ItemLocator,
  LinkNetwork,
  LinkSegment,
  Point
} from '@/lib/litegraph/src/interfaces'
import { EmptySubgraphInput } from '@/lib/litegraph/src/subgraph/EmptySubgraphInput'
import { EmptySubgraphOutput } from '@/lib/litegraph/src/subgraph/EmptySubgraphOutput'
import { Subgraph } from '@/lib/litegraph/src/subgraph/Subgraph'
import type { SubgraphInput } from '@/lib/litegraph/src/subgraph/SubgraphInput'
import { SubgraphInputNode } from '@/lib/litegraph/src/subgraph/SubgraphInputNode'
import type { SubgraphOutput } from '@/lib/litegraph/src/subgraph/SubgraphOutput'
import { SubgraphOutputNode } from '@/lib/litegraph/src/subgraph/SubgraphOutputNode'
import type { CanvasPointerEvent } from '@/lib/litegraph/src/types/events'
import { LinkDirection } from '@/lib/litegraph/src/types/globalEnums'
import type { IBaseWidget } from '@/lib/litegraph/src/types/widgets'

import { FloatingRenderLink } from './FloatingRenderLink'
import { MovingInputLink } from './MovingInputLink'
import { MovingLinkBase } from './MovingLinkBase'
import { MovingOutputLink } from './MovingOutputLink'
import type { RenderLink } from './RenderLink'
import { ToInputFromIoNodeLink } from './ToInputFromIoNodeLink'
import { ToInputRenderLink } from './ToInputRenderLink'
import { ToOutputFromIoNodeLink } from './ToOutputFromIoNodeLink'
import { ToOutputFromRerouteLink } from './ToOutputFromRerouteLink'
import { ToOutputRenderLink } from './ToOutputRenderLink'

/**
 * A Litegraph state object for the {@link LinkConnector}.
 * References are only held atomically within a function, never passed.
 * The concrete implementation may be replaced or proxied without side-effects.
 */
interface LinkConnectorState {
  /**
   * The type of slot that links are being connected **to**.
   * - When `undefined`, no operation is being performed.
   * - A change in this property indicates the start or end of dragging links.
   */
  connectingTo: 'input' | 'output' | undefined
  multi: boolean
  /** When `true`, existing links are being repositioned. Otherwise, new links are being created. */
  draggingExistingLinks: boolean
  /** When set, connecting links will all snap to this position. */
  snapLinksPos?: [number, number]
}

/** Discriminated union to simplify type narrowing. */
type RenderLinkUnion =
  | MovingInputLink
  | MovingOutputLink
  | FloatingRenderLink
  | ToInputRenderLink
  | ToOutputRenderLink
  | ToInputFromIoNodeLink
  | ToOutputFromIoNodeLink

interface LinkConnectorExport {
  renderLinks: RenderLink[]
  inputLinks: LLink[]
  outputLinks: LLink[]
  floatingLinks: LLink[]
  state: LinkConnectorState
  network: LinkNetwork
}

/**
 * Component of {@link LGraphCanvas} that handles connecting and moving links.
 * @see {@link LLink}
 */
export class LinkConnector {
  /**
   * Link connection state POJO. Source of truth for state of link drag operations.
   *
   * Can be replaced or proxied to allow notifications.
   * Is always dereferenced at the start of an operation.
   */
  state: LinkConnectorState = {
    connectingTo: undefined,
    multi: false,
    draggingExistingLinks: false,
    snapLinksPos: undefined
  }

  readonly events = new CustomEventTarget<LinkConnectorEventMap>()

  /** Contains information for rendering purposes only. */
  readonly renderLinks: RenderLinkUnion[] = []

  /** Existing links that are being moved **to** a new input slot. */
  readonly inputLinks: LLink[] = []
  /** Existing links that are being moved **to** a new output slot. */
  readonly outputLinks: LLink[] = []
  /** Existing floating links that are being moved to a new slot. */
  readonly floatingLinks: LLink[] = []

  readonly hiddenReroutes: Set<Reroute> = new Set()

  /** The widget beneath the pointer, if it is a valid connection target. */
  overWidget?: IBaseWidget
  /** The type (returned by downstream callback) for {@link overWidget} */
  overWidgetType?: string

  /** The reroute beneath the pointer, if it is a valid connection target. */
  overReroute?: Reroute

  private readonly _setConnectingLinks: (value: ConnectingLink[]) => void

  constructor(setConnectingLinks: (value: ConnectingLink[]) => void) {
    this._setConnectingLinks = setConnectingLinks
  }

  get isConnecting() {
    return this.state.connectingTo !== undefined
  }

  get draggingExistingLinks() {
    return this.state.draggingExistingLinks
  }

  /** Drag an existing link to a different input. */
  moveInputLink(
    network: LinkNetwork,
    input: INodeInputSlot,
    opts?: { startPoint?: Point }
  ): void {
    if (this.isConnecting) throw new Error('Already dragging links.')

    const { state, inputLinks, renderLinks } = this

    const linkId = input.link
    if (linkId == null) {
      // No link connected, check for a floating link
      const floatingLink = input._floatingLinks?.values().next().value
      if (floatingLink?.parentId == null) return

      try {
        const reroute = network.reroutes.get(floatingLink.parentId)
        if (!reroute)
          throw new Error(
            `Invalid reroute id: [${floatingLink.parentId}] for floating link id: [${floatingLink.id}].`
          )

        const renderLink = new FloatingRenderLink(
          network,
          floatingLink,
          'input',
          reroute
        )
        const mayContinue = this.events.dispatch(
          'before-move-input',
          renderLink
        )
        if (mayContinue === false) return

        renderLinks.push(renderLink)
      } catch (error) {
        console.warn(
          `Could not create render link for link id: [${floatingLink.id}].`,
          floatingLink,
          error
        )
      }

      floatingLink._dragging = true
      this.floatingLinks.push(floatingLink)
    } else {
      const link = network.links.get(linkId)
      if (!link) return

      // Special handling for links from subgraph input nodes
      if (link.origin_id === SUBGRAPH_INPUT_ID) {
        // For subgraph input links, we need to handle them differently
        // since they don't have a regular output node
        const subgraphInput = network.inputNode?.slots[link.origin_slot]
        if (!subgraphInput) {
          console.warn(
            `Could not find subgraph input for slot [${link.origin_slot}]`
          )
          return
        }

        try {
          const reroute = network.getReroute(link.parentId)
          const renderLink = new ToInputFromIoNodeLink(
            network,
            network.inputNode,
            subgraphInput,
            reroute,
            LinkDirection.CENTER,
            link
          )

          // Note: We don't dispatch the before-move-input event for subgraph input links
          // as the event type doesn't support ToInputFromIoNodeLink

          renderLinks.push(renderLink)

          this.listenUntilReset('input-moved', () => {
            link.disconnect(network, 'input')
          })
        } catch (error) {
          console.warn(
            `Could not create render link for subgraph input link id: [${link.id}].`,
            link,
            error
          )
          return
        }

        link._dragging = true
        inputLinks.push(link)
      } else {
        // Regular node links
        try {
          const reroute = network.getReroute(link.parentId)
          const renderLink = new MovingInputLink(
            network,
            link,
            reroute,
            undefined,
            opts?.startPoint
          )

          const mayContinue = this.events.dispatch(
            'before-move-input',
            renderLink
          )
          if (mayContinue === false) return

          renderLinks.push(renderLink)

          this.listenUntilReset('input-moved', (e) => {
            if ('link' in e.detail && e.detail.link) {
              e.detail.link.disconnect(network, 'output')
            }
          })
        } catch (error) {
          console.warn(
            `Could not create render link for link id: [${link.id}].`,
            link,
            error
          )
          return
        }

        link._dragging = true
        inputLinks.push(link)
      }
    }

    state.connectingTo = 'input'
    state.draggingExistingLinks = true

    this._setLegacyLinks(false)
  }

  /** Drag all links from an output to a new output. */
  moveOutputLink(network: LinkNetwork, output: INodeOutputSlot): void {
    if (this.isConnecting) throw new Error('Already dragging links.')

    const { state, renderLinks } = this

    // Floating links
    if (output._floatingLinks?.size) {
      for (const floatingLink of output._floatingLinks.values()) {
        try {
          const reroute = LLink.getFirstReroute(network, floatingLink)
          if (!reroute)
            throw new Error(
              `Invalid reroute id: [${floatingLink.parentId}] for floating link id: [${floatingLink.id}].`
            )

          const renderLink = new FloatingRenderLink(
            network,
            floatingLink,
            'output',
            reroute
          )
          const mayContinue = this.events.dispatch(
            'before-move-output',
            renderLink
          )
          if (mayContinue === false) continue

          renderLinks.push(renderLink)
          this.floatingLinks.push(floatingLink)
        } catch (error) {
          console.warn(
            `Could not create render link for link id: [${floatingLink.id}].`,
            floatingLink,
            error
          )
        }
      }
    }

    // Normal links
    if (output.links?.length) {
      for (const linkId of output.links) {
        const link = network.links.get(linkId)
        if (!link) continue

        const firstReroute = LLink.getFirstReroute(network, link)
        if (firstReroute) {
          firstReroute._dragging = true
          this.hiddenReroutes.add(firstReroute)
        } else {
          link._dragging = true
        }
        this.outputLinks.push(link)

        try {
          if (link.target_id === SUBGRAPH_OUTPUT_ID) {
            if (!(network instanceof Subgraph)) {
              console.warn(
                'Subgraph output link found in non-subgraph network.'
              )
              continue
            }

            const output = network.outputs.at(link.target_slot)
            if (!output) throw new Error('No subgraph output found for link.')

            const renderLink = new ToOutputFromIoNodeLink(
              network,
              network.outputNode,
              output
            )
            renderLink.fromDirection = LinkDirection.NONE
            renderLinks.push(renderLink)

            continue
          }
          const renderLink = new MovingOutputLink(
            network,
            link,
            firstReroute,
            LinkDirection.RIGHT
          )

          const mayContinue = this.events.dispatch(
            'before-move-output',
            renderLink
          )
          if (mayContinue === false) continue

          renderLinks.push(renderLink)
        } catch (error) {
          console.warn(
            `Could not create render link for link id: [${link.id}].`,
            link,
            error
          )
          continue
        }
      }
    }

    if (renderLinks.length === 0) return

    state.draggingExistingLinks = true
    state.multi = true
    state.connectingTo = 'output'

    this._setLegacyLinks(true)
  }

  /**
   * Drags a new link from an output slot to an input slot.
   * @param network The network that the link being connected belongs to
   * @param node The node the link is being dragged from
   * @param output The output slot that the link is being dragged from
   */
  dragNewFromOutput(
    network: LinkNetwork,
    node: LGraphNode,
    output: INodeOutputSlot,
    fromReroute?: Reroute
  ): void {
    if (this.isConnecting) throw new Error('Already dragging links.')

    const { state } = this
    const renderLink = new ToInputRenderLink(network, node, output, fromReroute)
    this.renderLinks.push(renderLink)

    state.connectingTo = 'input'

    this._setLegacyLinks(false)
  }

  /**
   * Drags a new link from an input slot to an output slot.
   * @param network The network that the link being connected belongs to
   * @param node The node the link is being dragged from
   * @param input The input slot that the link is being dragged from
   */
  dragNewFromInput(
    network: LinkNetwork,
    node: LGraphNode,
    input: INodeInputSlot,
    fromReroute?: Reroute
  ): void {
    if (this.isConnecting) throw new Error('Already dragging links.')

    const { state } = this
    const renderLink = new ToOutputRenderLink(network, node, input, fromReroute)
    this.renderLinks.push(renderLink)

    state.connectingTo = 'output'

    this._setLegacyLinks(true)
  }

  dragNewFromSubgraphInput(
    network: LinkNetwork,
    inputNode: SubgraphInputNode,
    input: SubgraphInput,
    fromReroute?: Reroute
  ): void {
    if (this.isConnecting) throw new Error('Already dragging links.')

    const renderLink = new ToInputFromIoNodeLink(
      network,
      inputNode,
      input,
      fromReroute
    )
    this.renderLinks.push(renderLink)

    this.state.connectingTo = 'input'

    this._setLegacyLinks(false)
  }

  dragNewFromSubgraphOutput(
    network: LinkNetwork,
    outputNode: SubgraphOutputNode,
    output: SubgraphOutput,
    fromReroute?: Reroute
  ): void {
    if (this.isConnecting) throw new Error('Already dragging links.')

    const renderLink = new ToOutputFromIoNodeLink(
      network,
      outputNode,
      output,
      fromReroute
    )
    this.renderLinks.push(renderLink)

    this.state.connectingTo = 'output'

    this._setLegacyLinks(true)
  }

  /**
   * Drags a new link from a reroute to an input slot.
   * @param network The network that the link being connected belongs to
   * @param reroute The reroute that the link is being dragged from
   */
  dragFromReroute(network: LinkNetwork, reroute: Reroute): void {
    if (this.isConnecting) throw new Error('Already dragging links.')

    const link = reroute.firstLink ?? reroute.firstFloatingLink
    if (!link) {
      console.warn('No link found for reroute.')
      return
    }

    if (link.origin_id === SUBGRAPH_INPUT_ID) {
      if (!(network instanceof Subgraph)) {
        console.warn('Subgraph input link found in non-subgraph network.')
        return
      }

      const input = network.inputs.at(link.origin_slot)
      if (!input) throw new Error('No subgraph input found for link.')

      const renderLink = new ToInputFromIoNodeLink(
        network,
        network.inputNode,
        input,
        reroute
      )
      renderLink.fromDirection = LinkDirection.NONE
      this.renderLinks.push(renderLink)

      this.state.connectingTo = 'input'

      this._setLegacyLinks(false)
      return
    }

    const outputNode = network.getNodeById(link.origin_id)
    if (!outputNode) {
      console.warn('No output node found for link.', link)
      return
    }

    const outputSlot = outputNode.outputs.at(link.origin_slot)
    if (!outputSlot) {
      console.warn('No output slot found for link.', link)
      return
    }

    const renderLink = new ToInputRenderLink(
      network,
      outputNode,
      outputSlot,
      reroute
    )
    renderLink.fromDirection = LinkDirection.NONE
    this.renderLinks.push(renderLink)

    this.state.connectingTo = 'input'

    this._setLegacyLinks(false)
  }

  /**
   * Drags a new link from a reroute to an output slot.
   * @param network The network that the link being connected belongs to
   * @param reroute The reroute that the link is being dragged from
   */
  dragFromRerouteToOutput(network: LinkNetwork, reroute: Reroute): void {
    if (this.isConnecting) throw new Error('Already dragging links.')

    const link = reroute.firstLink ?? reroute.firstFloatingLink
    if (!link) {
      console.warn('No link found for reroute.')
      return
    }

    if (link.target_id === SUBGRAPH_OUTPUT_ID) {
      if (!(network instanceof Subgraph)) {
        console.warn('Subgraph output link found in non-subgraph network.')
        return
      }

      const output = network.outputs.at(link.target_slot)
      if (!output) throw new Error('No subgraph output found for link.')

      const renderLink = new ToOutputFromIoNodeLink(
        network,
        network.outputNode,
        output,
        reroute
      )
      renderLink.fromDirection = LinkDirection.NONE
      this.renderLinks.push(renderLink)

      this.state.connectingTo = 'output'

      this._setLegacyLinks(false)
      return
    }

    const inputNode = network.getNodeById(link.target_id)
    if (!inputNode) {
      console.warn('No input node found for link.', link)
      return
    }

    const inputSlot = inputNode.inputs.at(link.target_slot)
    if (!inputSlot) {
      console.warn('No input slot found for link.', link)
      return
    }

    const renderLink = new ToOutputFromRerouteLink(
      network,
      inputNode,
      inputSlot,
      reroute,
      this
    )
    renderLink.fromDirection = LinkDirection.LEFT
    this.renderLinks.push(renderLink)

    this.state.connectingTo = 'output'

    this._setLegacyLinks(true)
  }

  dragFromLinkSegment(network: LinkNetwork, linkSegment: LinkSegment): void {
    if (this.isConnecting) throw new Error('Already dragging links.')

    const { state } = this
    if (linkSegment.origin_id == null || linkSegment.origin_slot == null) return

    const node = network.getNodeById(linkSegment.origin_id)
    if (!node) return

    const slot = node.outputs.at(linkSegment.origin_slot)
    if (!slot) return

    const reroute = network.getReroute(linkSegment.parentId)
    const renderLink = new ToInputRenderLink(network, node, slot, reroute)
    renderLink.fromDirection = LinkDirection.NONE
    this.renderLinks.push(renderLink)

    state.connectingTo = 'input'

    this._setLegacyLinks(false)
  }

  /**
   * Connects the links being dropped
   * @param event Contains the drop location, in canvas space
   */
  dropLinks(locator: ItemLocator, event: CanvasPointerEvent): void {
    if (!this.isConnecting) {
      const mayContinue = this.events.dispatch('before-drop-links', {
        renderLinks: this.renderLinks,
        event
      })
      if (mayContinue === false) return
    }

    try {
      const { canvasX, canvasY } = event

      const ioNode = locator.getIoNodeOnPos?.(canvasX, canvasY)
      if (ioNode) {
        this.dropOnIoNode(ioNode, event)
        return
      }

      const node = locator.getNodeOnPos(canvasX, canvasY) ?? undefined
      if (node) {
        this.dropOnNode(node, event)
      } else {
        // Get reroute if no node is found
        const reroute = locator.getRerouteOnPos(canvasX, canvasY)
        // Drop output->input link on reroute is not impl.
        if (reroute && this.isRerouteValidDrop(reroute)) {
          this.dropOnReroute(reroute, event)
        } else {
          this.dropOnNothing(event)
        }
      }
    } finally {
      this.events.dispatch('after-drop-links', {
        renderLinks: this.renderLinks,
        event
      })
    }
  }

  dropOnIoNode(
    ioNode: SubgraphInputNode | SubgraphOutputNode,
    event: CanvasPointerEvent
  ): void {
    const { renderLinks, state } = this
    const { connectingTo } = state
    const { canvasX, canvasY } = event

    if (connectingTo === 'input' && ioNode instanceof SubgraphOutputNode) {
      const output = ioNode.getSlotInPosition(canvasX, canvasY)
      if (!output) {
        this.dropOnNothing(event)
        return
      }

      // Track the actual slot to use for all connections
      let targetSlot = output

      for (const link of renderLinks) {
        link.connectToSubgraphOutput(targetSlot, this.events)

        // If we just connected to an EmptySubgraphOutput, check if we should reuse the slot
        if (output instanceof EmptySubgraphOutput && ioNode.slots.length > 0) {
          // Get the last created slot (newest one)
          const createdSlot = ioNode.slots[ioNode.slots.length - 1]

          // Only reuse the slot if the next link's type would be compatible
          // Otherwise, keep using EmptySubgraphOutput to create a new slot
          const nextLink = renderLinks[renderLinks.indexOf(link) + 1]
          if (nextLink && link.fromSlot.type === nextLink.fromSlot.type) {
            targetSlot = createdSlot
          } else {
            // Reset to EmptySubgraphOutput for different types
            targetSlot = output
          }
        }
      }
    } else if (
      connectingTo === 'output' &&
      ioNode instanceof SubgraphInputNode
    ) {
      const input = ioNode.getSlotInPosition(canvasX, canvasY)
      if (!input) {
        this.dropOnNothing(event)
        return
      }

      // Same logic for SubgraphInputNode if needed
      let targetSlot = input

      for (const link of renderLinks) {
        // Validate the connection type before proceeding
        if (
          'canConnectToSubgraphInput' in link &&
          !link.canConnectToSubgraphInput(targetSlot)
        ) {
          console.warn(
            'Invalid connection type',
            link.fromSlot.type,
            '->',
            targetSlot.type
          )
          continue
        }

        link.connectToSubgraphInput(targetSlot, this.events)

        // If we just connected to an EmptySubgraphInput, check if we should reuse the slot
        if (input instanceof EmptySubgraphInput && ioNode.slots.length > 0) {
          // Get the last created slot (newest one)
          const createdSlot = ioNode.slots[ioNode.slots.length - 1]

          // Only reuse the slot if the next link's type would be compatible
          // Otherwise, keep using EmptySubgraphInput to create a new slot
          const nextLink = renderLinks[renderLinks.indexOf(link) + 1]
          if (nextLink && link.fromSlot.type === nextLink.fromSlot.type) {
            targetSlot = createdSlot
          } else {
            // Reset to EmptySubgraphInput for different types
            targetSlot = input
          }
        }
      }
    } else {
      console.error(
        'Invalid connectingTo state &/ ioNode',
        connectingTo,
        ioNode
      )
    }
  }

  dropOnNode(node: LGraphNode, event: CanvasPointerEvent) {
    const { renderLinks, state } = this
    const { connectingTo } = state
    const { canvasX, canvasY } = event

    // Do nothing if every connection would loop back
    if (renderLinks.every((link) => link.node === node)) return

    // To output
    if (connectingTo === 'output') {
      const output = node.getOutputOnPos([canvasX, canvasY])

      if (output) {
        this._dropOnOutput(node, output)
      } else {
        this.connectToNode(node, event)
      }
      // To input
    } else if (connectingTo === 'input') {
      const input = node.getInputOnPos([canvasX, canvasY])
      const inputOrSocket = input ?? node.getSlotFromWidget(this.overWidget)

      // Input slot
      if (inputOrSocket) {
        this._dropOnInput(node, inputOrSocket)
      } else {
        // Node background / title
        this.connectToNode(node, event)
      }
    }
  }

  dropOnReroute(reroute: Reroute, event: CanvasPointerEvent): void {
    const mayContinue = this.events.dispatch('dropped-on-reroute', {
      reroute,
      event
    })
    if (mayContinue === false) return

    // Connecting to input
    if (this.state.connectingTo === 'input') {
      if (this.renderLinks.length !== 1)
        throw new Error(
          `Attempted to connect ${this.renderLinks.length} input links to a reroute.`
        )

      const renderLink = this.renderLinks[0]
      this._connectOutputToReroute(reroute, renderLink)

      return
    }

    // Connecting to output
    for (const link of this.renderLinks) {
      if (link.toType !== 'output') continue

      const result = reroute.findSourceOutput()
      if (!result) continue

      const { node, output } = result
      if (!link.canConnectToOutput(node, output)) continue

      link.connectToRerouteOutput(reroute, node, output, this.events)
    }
  }

  /** @internal Temporary workaround - requires refactor. */
  _connectOutputToReroute(reroute: Reroute, renderLink: RenderLinkUnion): void {
    const results = reroute.findTargetInputs()
    if (!results?.length) return

    const maybeReroutes = reroute.getReroutes()
    if (maybeReroutes === null) throw new Error('Reroute loop detected.')

    const originalReroutes = maybeReroutes.slice(0, -1).reverse()

    // From reroute to reroute
    if (renderLink instanceof ToInputRenderLink) {
      const { node, fromSlot, fromSlotIndex, fromReroute } = renderLink

      reroute.setFloatingLinkOrigin(node, fromSlot, fromSlotIndex)

      // Clean floating link IDs from reroutes about to be removed from the chain
      if (fromReroute != null) {
        for (const originalReroute of originalReroutes) {
          if (originalReroute.id === fromReroute.id) break

          for (const linkId of reroute.floatingLinkIds) {
            originalReroute.floatingLinkIds.delete(linkId)
          }
        }
      }
    }

    // Filter before any connections are re-created
    const filtered = results.filter(
      (result) =>
        renderLink.toType === 'input' &&
        canConnectInputLinkToReroute(
          renderLink,
          result.node,
          result.input,
          reroute
        )
    )

    for (const result of filtered) {
      renderLink.connectToRerouteInput(
        reroute,
        result,
        this.events,
        originalReroutes
      )
    }

    return
  }

  dropOnNothing(event: CanvasPointerEvent): void {
    remove(
      this.renderLinks,
      (link) => link instanceof MovingInputLink && link.disconnectOnDrop
    ).forEach((link) => (link as MovingLinkBase).disconnect())
    if (this.renderLinks.length === 0) return
    // For external event only.
    const mayContinue = this.events.dispatch('dropped-on-canvas', event)
    if (mayContinue === false) return

    this.disconnectLinks()
  }

  /**
   * Disconnects all moving links.
   * @remarks This is called when the links are dropped on the canvas.
   * May be called by consumers to e.g. drag links into a bin / void.
   */
  disconnectLinks(): void {
    for (const link of this.renderLinks) {
      if (
        link instanceof MovingLinkBase ||
        link instanceof ToInputFromIoNodeLink
      ) {
        link.disconnect()
      }
    }
  }

  /**
   * Connects the links being dropped onto a node to the first matching slot.
   * @param node The node that the links are being dropped on
   * @param event Contains the drop location, in canvas space
   */
  connectToNode(node: LGraphNode, event: CanvasPointerEvent): void {
    const {
      state: { connectingTo }
    } = this

    const mayContinue = this.events.dispatch('dropped-on-node', { node, event })
    if (mayContinue === false) return

    // Assume all links are the same type, disallow loopback
    const firstLink = this.renderLinks[0]
    if (!firstLink) return

    // Use a single type check before looping; ensures all dropped links go to the same slot
    if (connectingTo === 'output') {
      // Dropping new output link
      const output = node.findOutputByType(firstLink.fromSlot.type)?.slot
      if (output === undefined) {
        console.warn(
          `Could not find slot for link type: [${firstLink.fromSlot.type}].`
        )
        return
      }

      this._dropOnOutput(node, output)
    } else if (connectingTo === 'input') {
      // Dropping new input link
      const input = node.findInputByType(firstLink.fromSlot.type)?.slot
      if (input === undefined) {
        console.warn(
          `Could not find slot for link type: [${firstLink.fromSlot.type}].`
        )
        return
      }

      this._dropOnInput(node, input)
    }
  }

  private _dropOnInput(node: LGraphNode, input: INodeInputSlot): void {
    for (const link of this.renderLinks) {
      if (!link.canConnectToInput(node, input)) continue

      link.connectToInput(node, input, this.events)
    }
  }

  private _dropOnOutput(node: LGraphNode, output: INodeOutputSlot): void {
    for (const link of this.renderLinks) {
      if (!link.canConnectToOutput(node, output)) {
        if (
          link instanceof MovingOutputLink &&
          link.link.parentId !== undefined
        ) {
          // Reconnect link without reroutes
          link.outputNode.connectSlots(
            link.outputSlot,
            link.inputNode,
            link.inputSlot,
            undefined!
          )
        }
        continue
      }

      link.connectToOutput(node, output, this.events)
    }
  }

  isInputValidDrop(node: LGraphNode, input: INodeInputSlot): boolean {
    return this.renderLinks.some((link) => link.canConnectToInput(node, input))
  }

  isNodeValidDrop(node: LGraphNode): boolean {
    if (this.state.connectingTo === 'output') {
      return node.outputs.some((output) =>
        this.renderLinks.some((link) => link.canConnectToOutput(node, output))
      )
    }

    return node.inputs.some((input) =>
      this.renderLinks.some((link) => link.canConnectToInput(node, input))
    )
  }

  isSubgraphInputValidDrop(input: SubgraphInput): boolean {
    return this.renderLinks.some(
      (link) =>
        'canConnectToSubgraphInput' in link &&
        link.canConnectToSubgraphInput(input)
    )
  }

  /**
   * Checks if a reroute is a valid drop target for any of the links being connected.
   * @param reroute The reroute that would be dropped on.
   * @returns `true` if any of the current links being connected are valid for the given reroute.
   */
  isRerouteValidDrop(reroute: Reroute): boolean {
    if (this.state.connectingTo === 'input') {
      const results = reroute.findTargetInputs()
      if (!results?.length) return false

      for (const { node, input } of results) {
        for (const renderLink of this.renderLinks) {
          if (renderLink.toType !== 'input') continue
          if (canConnectInputLinkToReroute(renderLink, node, input, reroute))
            return true
        }
      }
    } else {
      const result = reroute.findSourceOutput()
      if (!result) return false

      const { node, output } = result

      for (const renderLink of this.renderLinks) {
        if (renderLink.toType !== 'output') continue
        if (!renderLink.canConnectToReroute(reroute)) continue
        if (renderLink.canConnectToOutput(node, output)) return true
      }
    }

    return false
  }

  /** Sets connecting_links, used by some extensions still. */
  private _setLegacyLinks(fromSlotIsInput: boolean): void {
    const links = this.renderLinks.map((link) => {
      const input = fromSlotIsInput ? (link.fromSlot as INodeInputSlot) : null
      const output = fromSlotIsInput ? null : (link.fromSlot as INodeOutputSlot)

      const afterRerouteId =
        link instanceof MovingLinkBase
          ? link.link?.parentId
          : link.fromReroute?.id

      return {
        node: link.node as LGraphNode,
        slot: link.fromSlotIndex,
        input,
        output,
        pos: link.fromPos,
        afterRerouteId
      } satisfies ConnectingLink
    })
    this._setConnectingLinks(links)
  }

  /**
   * Exports the current state of the link connector.
   * @param network The network that the links being connected belong to.
   * @returns A POJO with the state of the link connector, links being connected, and their network.
   * @remarks Other than {@link network}, all properties are shallow cloned.
   */
  export(network: LinkNetwork): LinkConnectorExport {
    return {
      renderLinks: [...this.renderLinks],
      inputLinks: [...this.inputLinks],
      outputLinks: [...this.outputLinks],
      floatingLinks: [...this.floatingLinks],
      state: { ...this.state },
      network
    }
  }

  /**
   * Adds an event listener that will be automatically removed when the reset event is fired.
   * @param eventName The event to listen for.
   * @param listener The listener to call when the event is fired.
   */
  listenUntilReset<K extends keyof LinkConnectorEventMap>(
    eventName: K,
    listener: Parameters<typeof this.events.addEventListener<K>>[1],
    options?: Parameters<typeof this.events.addEventListener<K>>[2]
  ) {
    this.events.addEventListener(eventName, listener, options)
    this.events.addEventListener(
      'reset',
      () => this.events.removeEventListener(eventName, listener),
      { once: true }
    )
  }

  /**
   * Resets everything to its initial state.
   *
   * Effectively cancels moving or connecting links.
   */
  reset(force = false): void {
    const mayContinue = this.events.dispatch('reset', force)
    if (mayContinue === false) return

    const {
      state,
      outputLinks,
      inputLinks,
      hiddenReroutes,
      renderLinks,
      floatingLinks
    } = this

    if (!force && state.connectingTo === undefined) return
    state.connectingTo = undefined

    for (const link of outputLinks) delete link._dragging
    for (const link of inputLinks) delete link._dragging
    for (const link of floatingLinks) delete link._dragging
    for (const reroute of hiddenReroutes) delete reroute._dragging

    renderLinks.length = 0
    inputLinks.length = 0
    outputLinks.length = 0
    floatingLinks.length = 0
    hiddenReroutes.clear()
    state.multi = false
    state.draggingExistingLinks = false
    state.snapLinksPos = undefined
  }
}

/** Validates that a single {@link RenderLink} can be dropped on the specified reroute. */
function canConnectInputLinkToReroute(
  link:
    | ToInputRenderLink
    | MovingInputLink
    | FloatingRenderLink
    | ToInputFromIoNodeLink,
  inputNode: LGraphNode,
  input: INodeInputSlot,
  reroute: Reroute
): boolean {
  const { fromReroute } = link

  if (
    !link.canConnectToInput(inputNode, input) ||
    // Would result in no change
    fromReroute?.id === reroute.id ||
    // Cannot connect from child to parent reroute
    fromReroute?.getReroutes()?.includes(reroute)
  ) {
    return false
  }

  // Would result in no change
  if (link instanceof ToInputRenderLink) {
    if (reroute.parentId == null) {
      // Link would make no change - output to reroute
      if (reroute.firstLink?.hasOrigin(link.node.id, link.fromSlotIndex))
        return false
    } else if (link.fromReroute?.id === reroute.parentId) {
      return false
    }
  }
  return true
}
