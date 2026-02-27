import {
  SUBGRAPH_INPUT_ID,
  SUBGRAPH_OUTPUT_ID
} from '@/lib/litegraph/src/constants'
import type { SubgraphInput } from '@/lib/litegraph/src/subgraph/SubgraphInput'
import type { SubgraphOutput } from '@/lib/litegraph/src/subgraph/SubgraphOutput'
import { useLayoutMutations } from '@/renderer/core/layout/operations/layoutMutations'
import { LayoutSource } from '@/renderer/core/layout/types'

import type { LGraphNode, NodeId } from './LGraphNode'
import type { Reroute, RerouteId } from './Reroute'
import type {
  CanvasColour,
  INodeInputSlot,
  INodeOutputSlot,
  ISlotType,
  LinkNetwork,
  LinkSegment,
  Point,
  ReadonlyLinkNetwork
} from './interfaces'
import type { Serialisable, SerialisableLLink } from './types/serialisation'

const layoutMutations = useLayoutMutations()

export type LinkId = number

export type SerialisedLLinkArray = [
  id: LinkId,
  origin_id: NodeId,
  origin_slot: number,
  target_id: NodeId,
  target_slot: number,
  type: ISlotType
]

// Resolved connection union; eliminates subgraph in/out as a possibility
export type ResolvedConnection = BaseResolvedConnection &
  (
    | (ResolvedSubgraphInput & ResolvedNormalOutput)
    | (ResolvedNormalInput & ResolvedSubgraphOutput)
    | (ResolvedNormalInput & ResolvedNormalOutput)
  )

interface BaseResolvedConnection {
  link: LLink
  /** The node on the input side of the link (owns {@link input}) */
  inputNode?: LGraphNode
  /** The input the link is connected to (mutually exclusive with {@link subgraphOutput}) */
  input?: INodeInputSlot
  /** The node on the output side of the link (owns {@link output}) */
  outputNode?: LGraphNode
  /** The output the link is connected to (mutually exclusive with {@link subgraphInput}) */
  output?: INodeOutputSlot
  /** The subgraph output the link is connected to (mutually exclusive with {@link input}) */
  subgraphOutput?: SubgraphOutput
  /** The subgraph input the link is connected to (mutually exclusive with {@link output}) */
  subgraphInput?: SubgraphInput
}

interface ResolvedNormalInput {
  inputNode: LGraphNode | undefined
  input: INodeInputSlot | undefined
  subgraphOutput?: undefined
}

interface ResolvedNormalOutput {
  outputNode: LGraphNode | undefined
  output: INodeOutputSlot | undefined
  subgraphInput?: undefined
}

interface ResolvedSubgraphInput {
  inputNode?: undefined
  /** The actual input slot the link is connected to (mutually exclusive with {@link subgraphOutput}) */
  input?: undefined
  subgraphOutput: SubgraphOutput
}

interface ResolvedSubgraphOutput {
  outputNode?: undefined
  output?: undefined
  subgraphInput: SubgraphInput
}

type BasicReadonlyNetwork = Pick<
  ReadonlyLinkNetwork,
  'getNodeById' | 'links' | 'getLink' | 'inputNode' | 'outputNode'
>

// this is the class in charge of storing link information
export class LLink implements LinkSegment, Serialisable<SerialisableLLink> {
  static _drawDebug = false

  /** Link ID */
  id: LinkId
  parentId?: RerouteId
  type: ISlotType
  /** Output node ID */
  origin_id: NodeId
  /** Output slot index */
  origin_slot: number
  /** Input node ID */
  target_id: NodeId
  /** Input slot index */
  target_slot: number

  data?: number | string | boolean | { toToolTip?(): string }
  _data?: unknown
  /** Centre point of the link, calculated during render only - can be inaccurate */
  _pos: Point
  /** @todo Clean up - never implemented in comfy. */
  _last_time?: number
  /** The last canvas 2D path that was used to render this link */
  path?: Path2D
  /** @inheritdoc */
  _centreAngle?: number

  /** @inheritdoc */
  _dragging?: boolean

  private _color?: CanvasColour | null
  /** Custom colour for this link only */
  public get color(): CanvasColour | null | undefined {
    return this._color
  }

  public set color(value: CanvasColour) {
    this._color = value === '' ? null : value
  }

  public get isFloatingOutput(): boolean {
    return this.origin_id === -1 && this.origin_slot === -1
  }

  public get isFloatingInput(): boolean {
    return this.target_id === -1 && this.target_slot === -1
  }

  public get isFloating(): boolean {
    return this.isFloatingOutput || this.isFloatingInput
  }

  /** `true` if this link is connected to a subgraph input node (the actual origin is in a different graph). */
  get originIsIoNode(): boolean {
    return this.origin_id === SUBGRAPH_INPUT_ID
  }

  /** `true` if this link is connected to a subgraph output node (the actual target is in a different graph). */
  get targetIsIoNode(): boolean {
    return this.target_id === SUBGRAPH_OUTPUT_ID
  }

  constructor(
    id: LinkId,
    type: ISlotType,
    origin_id: NodeId,
    origin_slot: number,
    target_id: NodeId,
    target_slot: number,
    parentId?: RerouteId
  ) {
    this.id = id
    this.type = type
    this.origin_id = origin_id
    this.origin_slot = origin_slot
    this.target_id = target_id
    this.target_slot = target_slot
    this.parentId = parentId

    this._data = null
    // center
    this._pos = [0, 0]
  }

  /** @deprecated Use {@link LLink.create} */
  static createFromArray(data: SerialisedLLinkArray): LLink {
    return new LLink(data[0], data[5], data[1], data[2], data[3], data[4])
  }

  /**
   * LLink static factory: creates a new LLink from the provided data.
   * @param data Serialised LLink data to create the link from
   * @returns A new LLink
   */
  static create(data: SerialisableLLink): LLink {
    return new LLink(
      data.id,
      data.type,
      data.origin_id,
      data.origin_slot,
      data.target_id,
      data.target_slot,
      data.parentId
    )
  }

  /**
   * Gets all reroutes from the output slot to this segment.  If this segment is a reroute, it will not be included.
   * @returns An ordered array of all reroutes from the node output to
   * this reroute or the reroute before it.  Otherwise, an empty array.
   */
  static getReroutes(
    network: Pick<ReadonlyLinkNetwork, 'reroutes'>,
    linkSegment: LinkSegment
  ): Reroute[] {
    if (linkSegment.parentId === undefined) return []
    return network.reroutes.get(linkSegment.parentId)?.getReroutes() ?? []
  }

  static getFirstReroute(
    network: Pick<ReadonlyLinkNetwork, 'reroutes'>,
    linkSegment: LinkSegment
  ): Reroute | undefined {
    return LLink.getReroutes(network, linkSegment).at(0)
  }

  /**
   * Finds the reroute in the chain after the provided reroute ID.
   * @param network The network this link belongs to
   * @param linkSegment The starting point of the search (input side).
   * Typically the LLink object itself, but can be any link segment.
   * @param rerouteId The matching reroute will have this set as its {@link parentId}.
   * @returns The reroute that was found, `undefined` if no reroute was found, or `null` if an infinite loop was detected.
   */
  static findNextReroute(
    network: Pick<ReadonlyLinkNetwork, 'reroutes'>,
    linkSegment: LinkSegment,
    rerouteId: RerouteId
  ): Reroute | null | undefined {
    if (linkSegment.parentId === undefined) return
    return network.reroutes
      .get(linkSegment.parentId)
      ?.findNextReroute(rerouteId)
  }

  /**
   * Gets the origin node of a link.
   * @param network The network to search
   * @param linkId The ID of the link to get the origin node of
   * @returns The origin node of the link, or `undefined` if the link is not found or the origin node is not found
   */
  static getOriginNode(
    network: BasicReadonlyNetwork,
    linkId: LinkId
  ): LGraphNode | undefined {
    const id = network.links.get(linkId)?.origin_id
    return network.getNodeById(id) ?? undefined
  }

  /**
   * Gets the target node of a link.
   * @param network The network to search
   * @param linkId The ID of the link to get the target node of
   * @returns The target node of the link, or `undefined` if the link is not found or the target node is not found
   */
  static getTargetNode(
    network: BasicReadonlyNetwork,
    linkId: LinkId
  ): LGraphNode | undefined {
    const id = network.links.get(linkId)?.target_id
    return network.getNodeById(id) ?? undefined
  }

  /**
   * Resolves a link ID to the link, node, and slot objects.
   * @param linkId The {@link id} of the link to resolve
   * @param network The link network to search
   * @returns An object containing the input and output nodes, as well as the input and output slots.
   * @remarks This method is heavier than others; it will always resolve all objects.
   * Whilst the performance difference should in most cases be negligible,
   * it is recommended to use simpler methods where appropriate.
   */
  static resolve(
    linkId: LinkId | null | undefined,
    network: BasicReadonlyNetwork
  ): ResolvedConnection | undefined {
    return network.getLink(linkId)?.resolve(network)
  }

  /**
   * Resolves a list of link IDs to the link, node, and slot objects.
   * Discards invalid link IDs.
   * @param linkIds An iterable of link {@link id}s to resolve
   * @param network The link network to search
   * @returns An array of resolved connections.  If a link is not found, it is not included in the array.
   * @see {@link LLink.resolve}
   */
  static resolveMany(
    linkIds: Iterable<LinkId>,
    network: BasicReadonlyNetwork
  ): ResolvedConnection[] {
    const resolved: ResolvedConnection[] = []
    for (const id of linkIds) {
      const r = network.getLink(id)?.resolve(network)
      if (r) resolved.push(r)
    }
    return resolved
  }

  /**
   * Resolves the primitive ID values stored in the link to the referenced objects.
   * @param network The link network to search
   * @returns An object containing the input and output nodes, as well as the input and output slots.
   * @remarks This method is heavier than others; it will always resolve all objects.
   * Whilst the performance difference should in most cases be negligible,
   * it is recommended to use simpler methods where appropriate.
   */
  resolve(network: BasicReadonlyNetwork): ResolvedConnection {
    const inputNode =
      this.target_id === -1
        ? undefined
        : (network.getNodeById(this.target_id) ?? undefined)
    const input = inputNode?.inputs[this.target_slot]
    const subgraphInput = this.originIsIoNode
      ? network.inputNode?.slots[this.origin_slot]
      : undefined
    if (subgraphInput) {
      return { inputNode, input, subgraphInput, link: this }
    }

    const outputNode =
      this.origin_id === -1
        ? undefined
        : (network.getNodeById(this.origin_id) ?? undefined)
    const output = outputNode?.outputs[this.origin_slot]
    const subgraphOutput = this.targetIsIoNode
      ? network.outputNode?.slots[this.target_slot]
      : undefined
    if (subgraphOutput) {
      return {
        outputNode,
        output,
        subgraphInput: undefined,
        subgraphOutput,
        link: this
      }
    }

    return {
      inputNode,
      outputNode,
      input,
      output,
      subgraphInput,
      subgraphOutput,
      link: this
    }
  }

  configure(o: LLink | SerialisedLLinkArray) {
    if (Array.isArray(o)) {
      this.id = o[0]
      this.origin_id = o[1]
      this.origin_slot = o[2]
      this.target_id = o[3]
      this.target_slot = o[4]
      this.type = o[5]
    } else {
      this.id = o.id
      this.type = o.type
      this.origin_id = o.origin_id
      this.origin_slot = o.origin_slot
      this.target_id = o.target_id
      this.target_slot = o.target_slot
      this.parentId = o.parentId
    }
  }

  /**
   * Checks if the specified node id and output index are this link's origin (output side).
   * @param nodeId ID of the node to check
   * @param outputIndex The array index of the node output
   * @returns `true` if the origin matches, otherwise `false`.
   */
  hasOrigin(nodeId: NodeId, outputIndex: number): boolean {
    return this.origin_id === nodeId && this.origin_slot === outputIndex
  }

  /**
   * Checks if the specified node id and input index are this link's target (input side).
   * @param nodeId ID of the node to check
   * @param inputIndex The array index of the node input
   * @returns `true` if the target matches, otherwise `false`.
   */
  hasTarget(nodeId: NodeId, inputIndex: number): boolean {
    return this.target_id === nodeId && this.target_slot === inputIndex
  }

  /**
   * Creates a floating link from this link.
   * @param slotType The side of the link that is still connected
   * @param parentId The parent reroute ID of the link
   * @returns A new LLink that is floating
   */
  toFloating(slotType: 'input' | 'output', parentId: RerouteId): LLink {
    const exported = this.asSerialisable()
    exported.id = -1
    exported.parentId = parentId

    if (slotType === 'input') {
      exported.origin_id = -1
      exported.origin_slot = -1
    } else {
      exported.target_id = -1
      exported.target_slot = -1
    }

    return LLink.create(exported)
  }

  /**
   * Disconnects a link and removes it from the graph, cleaning up any reroutes that are no longer used
   * @param network The container (LGraph) where reroutes should be updated
   * @param keepReroutes If `undefined`, reroutes will be automatically removed if no links remain.
   * If `input` or `output`, reroutes will not be automatically removed, and retain a connection to the input or output, respectively.
   */
  disconnect(network: LinkNetwork, keepReroutes?: 'input' | 'output'): void {
    const reroutes = LLink.getReroutes(network, this)

    const lastReroute = reroutes.at(-1)

    // When floating from output, 1-to-1 ratio of floating link to final reroute (tree-like)
    const outputFloating =
      keepReroutes === 'output' &&
      lastReroute?.linkIds.size === 1 &&
      lastReroute.floatingLinkIds.size === 0

    // When floating from inputs, the final (input side) reroute may have many floating links
    if (outputFloating || (keepReroutes === 'input' && lastReroute)) {
      const newLink = LLink.create(this)
      newLink.id = -1

      if (keepReroutes === 'input') {
        newLink.origin_id = -1
        newLink.origin_slot = -1

        lastReroute.floating = { slotType: 'input' }
      } else {
        newLink.target_id = -1
        newLink.target_slot = -1

        lastReroute.floating = { slotType: 'output' }
      }

      network.addFloatingLink(newLink)
    }

    for (const reroute of reroutes) {
      reroute.linkIds.delete(this.id)
      if (!keepReroutes && !reroute.totalLinks) {
        network.reroutes.delete(reroute.id)
        // Delete reroute from Layout Store
        layoutMutations.setSource(LayoutSource.Canvas)
        layoutMutations.deleteReroute(reroute.id)
      }
    }
    network.links.delete(this.id)
    // Delete link from Layout Store
    layoutMutations.setSource(LayoutSource.Canvas)
    layoutMutations.deleteLink(this.id)
  }

  /**
   * @deprecated Prefer {@link LLink.asSerialisable} (returns an object, not an array)
   * @returns An array representing this LLink
   */
  serialize(): SerialisedLLinkArray {
    return [
      this.id,
      this.origin_id,
      this.origin_slot,
      this.target_id,
      this.target_slot,
      this.type
    ]
  }

  asSerialisable(): SerialisableLLink {
    const copy: SerialisableLLink = {
      id: this.id,
      origin_id: this.origin_id,
      origin_slot: this.origin_slot,
      target_id: this.target_id,
      target_slot: this.target_slot,
      type: this.type
    }
    if (this.parentId !== undefined) copy.parentId = this.parentId
    return copy
  }
}
