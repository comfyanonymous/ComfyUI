import { useLayoutMutations } from '@/renderer/core/layout/operations/layoutMutations'
import { LayoutSource } from '@/renderer/core/layout/types'

import { LGraphBadge } from './LGraphBadge'
import type { LGraphNode, NodeId } from './LGraphNode'
import { LLink } from './LLink'
import type { LinkId } from './LLink'
import type {
  CanvasColour,
  INodeInputSlot,
  INodeOutputSlot,
  LinkNetwork,
  LinkSegment,
  Point,
  Positionable,
  ReadOnlyRect,
  ReadonlyLinkNetwork
} from './interfaces'
import { distance, isPointInRect } from './measure'
import type { Serialisable, SerialisableReroute } from './types/serialisation'

const layoutMutations = useLayoutMutations()

export type RerouteId = number

/** The input or output slot that an incomplete reroute link is connected to. */
export interface FloatingRerouteSlot {
  /** Floating connection to an input or output */
  slotType: 'input' | 'output'
}

/**
 * Represents an additional point on the graph that a link path will travel through.  Used for visual organisation only.
 *
 * Requires no disposal or clean up.
 * Stores only primitive values (IDs) to reference other items in its network,
 * and a `WeakRef` to a {@link LinkNetwork} to resolve them.
 */
export class Reroute
  implements Positionable, LinkSegment, Serialisable<SerialisableReroute>
{
  static radius: number = 10
  /** Maximum distance from reroutes to their bezier curve control points. */
  static maxSplineOffset: number = 80
  static drawIdBadge: boolean = false
  static slotRadius: number = 5
  /** Distance from reroute centre to slot centre. */
  static get slotOffset(): number {
    const gap = Reroute.slotRadius * 0.33
    return Reroute.radius + gap + Reroute.slotRadius
  }

  /** The network this reroute belongs to.  Contains all valid links and reroutes. */
  private readonly network: WeakRef<LinkNetwork>

  private parentIdInternal?: RerouteId
  public get parentId(): RerouteId | undefined {
    return this.parentIdInternal
  }

  /** Ignores attempts to create an infinite loop. @inheritdoc */
  public set parentId(value) {
    if (value === this.id) return
    if (this.getReroutes() === null) return
    this.parentIdInternal = value
  }

  public get parent(): Reroute | undefined {
    return this.network.deref()?.getReroute(this.parentIdInternal)
  }

  /** This property is only defined on the last reroute of a floating reroute chain (closest to input end). */
  floating?: FloatingRerouteSlot

  private readonly posInternal: Point = [0, 0]
  /** @inheritdoc */
  get pos(): Point {
    return this.posInternal
  }

  set pos(value: Point) {
    if (!(value?.length >= 2))
      throw new TypeError(
        'Reroute.pos is an x,y point, and expects an indexable with at least two values.'
      )
    this.posInternal[0] = value[0]
    this.posInternal[1] = value[1]
  }

  /** @inheritdoc */
  get boundingRect(): ReadOnlyRect {
    const { radius } = Reroute
    const [x, y] = this.posInternal
    return [x - radius, y - radius, 2 * radius, 2 * radius]
  }

  /**
   * Slightly over-sized rectangle, guaranteed to contain the entire surface area for hover detection.
   * Eliminates most hover positions using an extremely cheap check.
   */
  private get hoverArea(): ReadOnlyRect {
    const xOffset = 2 * Reroute.slotOffset
    const yOffset = 2 * Math.max(Reroute.radius, Reroute.slotRadius)

    const [x, y] = this.posInternal
    return [x - xOffset, y - yOffset, 2 * xOffset, 2 * yOffset]
  }

  /** The total number of links & floating links using this reroute */
  get totalLinks(): number {
    return this.linkIds.size + this.floatingLinkIds.size
  }

  /** @inheritdoc */
  selected?: boolean

  /** The ID ({@link LLink.id}) of every link using this reroute */
  linkIds: Set<LinkId>

  /** The ID ({@link LLink.id}) of every floating link using this reroute */
  floatingLinkIds: Set<LinkId>

  /** Cached cos */
  cos: number = 0
  sin: number = 0

  /** Bezier curve control point for the "target" (input) side of the link */
  controlPoint: Point = [0, 0]

  /** @inheritdoc */
  path?: Path2D
  /** @inheritdoc */
  _centreAngle?: number
  /** @inheritdoc */
  _pos: Point = [0, 0]

  /** @inheritdoc */
  _dragging?: boolean

  /** Colour of the first link that rendered this reroute */
  _colour?: CanvasColour

  /** Colour of the first link that rendered this reroute */
  get colour(): CanvasColour {
    return this._colour ?? '#18184d'
  }

  /**
   * Used to ensure reroute angles are only executed once per frame.
   * @todo Calculate on change instead.
   */
  private lastRenderTime: number = -Infinity

  private readonly inputSlot = new RerouteSlot(this, true)
  private readonly outputSlot = new RerouteSlot(this, false)

  get isSlotHovered(): boolean {
    return this.isInputHovered || this.isOutputHovered
  }

  get isInputHovered(): boolean {
    return this.inputSlot.hovering
  }

  get isOutputHovered(): boolean {
    return this.outputSlot.hovering
  }

  get firstLink(): LLink | undefined {
    const linkId = this.linkIds.values().next().value
    return linkId === undefined
      ? undefined
      : this.network.deref()?.links.get(linkId)
  }

  get firstFloatingLink(): LLink | undefined {
    const linkId = this.floatingLinkIds.values().next().value
    return linkId === undefined
      ? undefined
      : this.network.deref()?.floatingLinks.get(linkId)
  }

  /** @inheritdoc */
  get origin_id(): NodeId | undefined {
    return this.firstLink?.origin_id
  }

  /** @inheritdoc */
  get origin_slot(): number | undefined {
    return this.firstLink?.origin_slot
  }

  /**
   * Initialises a new link reroute object.
   * @param id Unique identifier for this reroute
   * @param network The network of links this reroute belongs to.  Internally converted to a WeakRef.
   * @param pos Position in graph coordinates
   * @param linkIds Link IDs ({@link LLink.id}) of all links that use this reroute
   */
  constructor(
    public readonly id: RerouteId,
    network: LinkNetwork,
    pos?: Point,
    parentId?: RerouteId,
    linkIds?: Iterable<LinkId>,
    floatingLinkIds?: Iterable<LinkId>
  ) {
    this.network = new WeakRef(network)
    this.parentId = parentId
    if (pos) this.pos = pos
    this.linkIds = new Set(linkIds)
    this.floatingLinkIds = new Set(floatingLinkIds)
  }

  /**
   * Applies a new parentId to the reroute, and optinoally a new position and linkId.
   * Primarily used for deserialisation.
   * @param parentId The ID of the reroute prior to this reroute, or
   * `undefined` if it is the first reroute connected to a nodes output
   * @param pos The position of this reroute
   * @param linkIds All link IDs that pass through this reroute
   */
  update(
    parentId: RerouteId | undefined,
    pos?: Point,
    linkIds?: Iterable<LinkId>,
    floating?: FloatingRerouteSlot
  ): void {
    this.parentId = parentId
    if (pos) this.pos = pos
    if (linkIds) this.linkIds = new Set(linkIds)
    this.floating = floating
  }

  /**
   * Validates the linkIds this reroute has.  Removes broken links.
   * @param links Collection of valid links
   * @returns true if any links remain after validation
   */
  validateLinks(
    links: ReadonlyMap<LinkId, LLink>,
    floatingLinks: ReadonlyMap<LinkId, LLink>
  ): boolean {
    const { linkIds, floatingLinkIds } = this
    for (const linkId of linkIds) {
      if (!links.has(linkId)) linkIds.delete(linkId)
    }
    for (const linkId of floatingLinkIds) {
      if (!floatingLinks.has(linkId)) floatingLinkIds.delete(linkId)
    }
    return linkIds.size > 0 || floatingLinkIds.size > 0
  }

  /**
   * Retrieves an ordered array of all reroutes from the node output.
   * @param visited Internal.  A set of reroutes that this function
   * has already visited whilst recursing up the chain.
   * @returns An ordered array of all reroutes from the node output to this reroute, inclusive.
   * `null` if an infinite loop is detected.
   * `undefined` if the reroute chain or {@link LinkNetwork} are invalid.
   */
  getReroutes(visited = new Set<Reroute>()): Reroute[] | null {
    // No parentId - last in the chain
    if (this.parentIdInternal === undefined) return [this]
    // Invalid chain - looped
    if (visited.has(this)) return null
    visited.add(this)

    const parent = this.network.deref()?.reroutes.get(this.parentIdInternal)
    // Invalid parent (or network) - drop silently to recover
    if (!parent) {
      this.parentIdInternal = undefined
      return [this]
    }

    const reroutes = parent.getReroutes(visited)
    reroutes?.push(this)
    return reroutes
  }

  /**
   * Internal.  Called by {@link LLink.findNextReroute}.  Not intended for use by itself.
   * @param withParentId The rerouteId to look for
   * @param visited A set of reroutes that have already been visited
   * @returns The reroute that was found, `undefined` if no reroute was found, or `null` if an infinite loop was detected.
   */
  findNextReroute(
    withParentId: RerouteId,
    visited = new Set<Reroute>()
  ): Reroute | null | undefined {
    if (this.parentIdInternal === withParentId) return this
    if (visited.has(this)) return null
    visited.add(this)
    if (this.parentIdInternal === undefined) return

    return this.network
      .deref()
      ?.reroutes.get(this.parentIdInternal)
      ?.findNextReroute(withParentId, visited)
  }

  /**
   * Finds the output node and output slot of the first link passing through this reroute.
   * @returns The output node and output slot of the first link passing through this reroute, or `undefined` if no link is found.
   */
  findSourceOutput():
    | { node: LGraphNode; output: INodeOutputSlot }
    | undefined {
    const link = this.firstLink ?? this.firstFloatingLink
    if (!link) return

    const node = this.network.deref()?.getNodeById(link.origin_id)
    if (!node) return

    return {
      node,
      output: node.outputs[link.origin_slot]
    }
  }

  /**
   * Finds the inputs and nodes of (floating) links passing through this reroute.
   * @returns An array of objects containing the node and input slot of each link passing through this reroute.
   */
  findTargetInputs():
    | { node: LGraphNode; input: INodeInputSlot; link: LLink }[]
    | undefined {
    const network = this.network.deref()
    if (!network) return

    const results: {
      node: LGraphNode
      input: INodeInputSlot
      link: LLink
    }[] = []

    addAllResults(network, this.linkIds, network.links)
    addAllResults(network, this.floatingLinkIds, network.floatingLinks)

    return results

    function addAllResults(
      network: ReadonlyLinkNetwork,
      linkIds: Iterable<LinkId>,
      links: ReadonlyMap<LinkId, LLink>
    ) {
      for (const linkId of linkIds) {
        const link = links.get(linkId)
        if (!link) continue

        const node = network.getNodeById(link.target_id)
        const input = node?.inputs[link.target_slot]
        if (!input) continue

        results.push({ node, input, link })
      }
    }
  }

  /**
   * Retrieves all floating links passing through this reroute.
   * @param from Filters the links by the currently connected link side.
   * @returns An array of floating links
   */
  getFloatingLinks(from: 'input' | 'output'): LLink[] | undefined {
    const floatingLinks = this.network.deref()?.floatingLinks
    if (!floatingLinks) return

    const idProp = from === 'input' ? 'origin_id' : 'target_id'
    const out: LLink[] = []

    for (const linkId of this.floatingLinkIds) {
      const link = floatingLinks.get(linkId)
      if (link?.[idProp] === -1) out.push(link)
    }
    return out
  }

  /**
   * Changes the origin node/output of all floating links that pass through this reroute.
   * @param node The new origin node
   * @param output The new origin output slot
   * @param index The slot index of {@link output}
   */
  setFloatingLinkOrigin(
    node: LGraphNode,
    output: INodeOutputSlot,
    index: number
  ) {
    const network = this.network.deref()
    const floatingOutLinks = this.getFloatingLinks('output')
    if (!floatingOutLinks)
      throw new Error('[setFloatingLinkOrigin]: Invalid network.')
    if (!floatingOutLinks.length) return

    output._floatingLinks ??= new Set()

    for (const link of floatingOutLinks) {
      // Update cached floating links
      output._floatingLinks.add(link)

      network
        ?.getNodeById(link.origin_id)
        ?.outputs[link.origin_slot]?._floatingLinks?.delete(link)

      // Update the floating link
      link.origin_id = node.id
      link.origin_slot = index
    }
  }

  /** @inheritdoc */
  move(deltaX: number, deltaY: number) {
    const previousPos = { x: this.posInternal[0], y: this.posInternal[1] }
    this.posInternal[0] += deltaX
    this.posInternal[1] += deltaY

    // Update Layout Store with new position
    layoutMutations.setSource(LayoutSource.Canvas)
    layoutMutations.moveReroute(
      this.id,
      { x: this.posInternal[0], y: this.posInternal[1] },
      previousPos
    )
  }

  /** @inheritdoc */
  snapToGrid(snapTo: number): boolean {
    if (!snapTo) return false

    const { pos } = this
    pos[0] = snapTo * Math.round(pos[0] / snapTo)
    pos[1] = snapTo * Math.round(pos[1] / snapTo)
    return true
  }

  removeAllFloatingLinks() {
    for (const linkId of this.floatingLinkIds) {
      this.removeFloatingLink(linkId)
    }
  }

  removeFloatingLink(linkId: LinkId) {
    const network = this.network.deref()
    if (!network) return

    const floatingLink = network.floatingLinks.get(linkId)
    if (!floatingLink) {
      console.warn(
        `[Reroute.removeFloatingLink] Floating link not found: ${linkId}, ignoring and discarding ID.`
      )
      this.floatingLinkIds.delete(linkId)
      return
    }

    network.removeFloatingLink(floatingLink)
  }

  /**
   * Removes a link or floating link from this reroute, by matching link object instance equality.
   * @param link The link to remove.
   * @remarks Does not remove the link from the network.
   */
  removeLink(link: LLink) {
    const network = this.network.deref()
    if (!network) return

    const floatingLink = network.floatingLinks.get(link.id)
    if (link === floatingLink) {
      this.floatingLinkIds.delete(link.id)
    } else {
      this.linkIds.delete(link.id)
    }
  }

  remove() {
    const network = this.network.deref()
    if (!network) return

    network.removeReroute(this.id)
  }

  calculateAngle(
    lastRenderTime: number,
    network: ReadonlyLinkNetwork,
    linkStart: Point
  ): void {
    // Ensure we run once per render
    if (!(lastRenderTime > this.lastRenderTime)) return
    this.lastRenderTime = lastRenderTime

    const { id, pos: thisPos } = this

    // Add all link angles
    const angles: number[] = []
    let sum = 0
    calculateAngles(this.linkIds, network.links)
    calculateAngles(this.floatingLinkIds, network.floatingLinks)

    // Invalid - reset
    if (!angles.length) {
      this.cos = 0
      this.sin = 0
      this.controlPoint[0] = 0
      this.controlPoint[1] = 0
      return
    }

    sum /= angles.length

    const originToReroute = Math.atan2(
      this.posInternal[1] - linkStart[1],
      this.posInternal[0] - linkStart[0]
    )
    let diff = (originToReroute - sum) * 0.5
    if (Math.abs(diff) > Math.PI * 0.5) diff += Math.PI
    const dist = Math.min(
      Reroute.maxSplineOffset,
      distance(linkStart, this.posInternal) * 0.25
    )

    // Store results
    const originDiff = originToReroute - diff
    const cos = Math.cos(originDiff)
    const sin = Math.sin(originDiff)

    this.cos = cos
    this.sin = sin
    this.controlPoint[0] = dist * -cos
    this.controlPoint[1] = dist * -sin

    /**
     * Calculates the direction of each link and adds it to the array.
     * @param linkIds The IDs of the links to calculate
     * @param links The link container from the link network.
     */
    function calculateAngles(
      linkIds: Iterable<LinkId>,
      links: ReadonlyMap<LinkId, LLink>
    ) {
      for (const linkId of linkIds) {
        const link = links.get(linkId)
        const pos = getNextPos(network, link, id)
        if (!pos) continue

        const angle = getDirection(thisPos, pos)
        angles.push(angle)
        sum += angle
      }
    }
  }

  /**
   * Renders the reroute on the canvas.
   * @param ctx Canvas context to draw on
   * @param backgroundPattern The canvas background pattern; used to make floating reroutes appear washed out.
   * @remarks Leaves {@link ctx}.fillStyle, strokeStyle, and lineWidth dirty (perf.).
   */
  draw(ctx: CanvasRenderingContext2D, backgroundPattern?: CanvasPattern): void {
    const { globalAlpha } = ctx
    const { pos } = this

    ctx.beginPath()
    ctx.arc(pos[0], pos[1], Reroute.radius, 0, 2 * Math.PI)

    if (this.linkIds.size === 0) {
      ctx.fillStyle = backgroundPattern ?? '#797979'
      ctx.fill()
      ctx.globalAlpha = globalAlpha * 0.33
    }

    ctx.fillStyle = this.colour
    ctx.lineWidth = Reroute.radius * 0.1
    ctx.strokeStyle = 'rgb(0,0,0,0.5)'
    ctx.fill()
    ctx.stroke()

    ctx.fillStyle = '#ffffff55'
    ctx.strokeStyle = 'rgb(0,0,0,0.3)'
    ctx.beginPath()
    ctx.arc(pos[0], pos[1], Reroute.radius * 0.8, 0, 2 * Math.PI)
    ctx.fill()
    ctx.stroke()

    if (this.selected) {
      ctx.strokeStyle = '#fff'
      ctx.beginPath()
      ctx.arc(pos[0], pos[1], Reroute.radius * 1.2, 0, 2 * Math.PI)
      ctx.stroke()
    }

    if (Reroute.drawIdBadge) {
      const idBadge = new LGraphBadge({ text: this.id.toString() })
      const x = pos[0] - idBadge.getWidth(ctx) * 0.5
      const y = pos[1] - idBadge.height - Reroute.radius - 2
      idBadge.draw(ctx, x, y)
    }

    ctx.globalAlpha = globalAlpha
  }

  /**
   * Draws the input and output slots on the canvas, if the slots are visible.
   * @param ctx The canvas context to draw on.
   */
  drawSlots(ctx: CanvasRenderingContext2D): void {
    this.inputSlot.draw(ctx)
    this.outputSlot.draw(ctx)
  }

  drawHighlight(ctx: CanvasRenderingContext2D, colour: CanvasColour): void {
    const { pos } = this

    const { strokeStyle, lineWidth } = ctx
    ctx.strokeStyle = colour
    ctx.lineWidth = 1

    ctx.beginPath()
    ctx.arc(pos[0], pos[1], Reroute.radius * 1.5, 0, 2 * Math.PI)
    ctx.stroke()

    ctx.strokeStyle = strokeStyle
    ctx.lineWidth = lineWidth
  }

  /**
   * Updates visibility of the input and output slots, based on the position of the pointer.
   * @param pos The position of the pointer.
   * @returns `true` if any changes require a redraw.
   */
  updateVisibility(pos: Point): boolean {
    const input = this.inputSlot
    const output = this.outputSlot
    input.dirty = false
    output.dirty = false

    const { firstFloatingLink } = this
    const hasLink = !!this.firstLink

    const showInput = hasLink || firstFloatingLink?.isFloatingOutput
    const showOutput = hasLink || firstFloatingLink?.isFloatingInput
    const showEither = showInput || showOutput

    // Check if even in the vicinity
    if (showEither && isPointInRect(pos, this.hoverArea)) {
      const outlineOnly = this.contains(pos)

      if (showInput) input.update(pos, outlineOnly)
      if (showOutput) output.update(pos, outlineOnly)
    } else {
      this.hideSlots()
    }

    return input.dirty || output.dirty
  }

  /** Prevents rendering of the input and output slots. */
  hideSlots() {
    this.inputSlot.hide()
    this.outputSlot.hide()
  }

  /**
   * Precisely determines if {@link pos} is inside this reroute.
   * @param pos The position to check (canvas space)
   * @returns `true` if {@link pos} is within the reroute's radius.
   */
  containsPoint(pos: Point): boolean {
    return isPointInRect(pos, this.hoverArea) && this.contains(pos)
  }

  private contains(pos: Point): boolean {
    return distance(this.pos, pos) <= Reroute.radius
  }

  /** @inheritdoc */
  asSerialisable(): SerialisableReroute {
    const { id, parentId, pos, linkIds } = this
    return {
      id,
      parentId,
      pos: [pos[0], pos[1]],
      linkIds: [...linkIds],
      floating: this.floating ? { slotType: this.floating.slotType } : undefined
    }
  }
}

/**
 * Represents a slot on a reroute.
 * @private Designed for internal use within this module.
 */
class RerouteSlot {
  /** The reroute that the slot belongs to. */
  private readonly reroute: Reroute

  private readonly offsetMultiplier: 1 | -1
  /** Centre point of this slot. */
  get pos(): Point {
    const [x, y] = this.reroute.pos
    return [x + Reroute.slotOffset * this.offsetMultiplier, y]
  }

  /** Whether any changes require a redraw. */
  dirty: boolean = false

  private hoveringInternal = false
  /** Whether the pointer is hovering over the slot itself. */
  get hovering() {
    return this.hoveringInternal
  }

  set hovering(value) {
    if (!Object.is(this.hoveringInternal, value)) {
      this.hoveringInternal = value
      this.dirty = true
    }
  }

  private showOutlineInternal = false
  /** Whether the slot outline / faint background is visible. */
  get showOutline() {
    return this.showOutlineInternal
  }

  set showOutline(value) {
    if (!Object.is(this.showOutlineInternal, value)) {
      this.showOutlineInternal = value
      this.dirty = true
    }
  }

  constructor(reroute: Reroute, isInput: boolean) {
    this.reroute = reroute
    this.offsetMultiplier = isInput ? -1 : 1
  }

  /**
   * Updates the slot's visibility based on the position of the pointer.
   * @param pos The position of the pointer.
   * @param outlineOnly If `true`, slot will display with the faded outline only ({@link showOutline}).
   */
  update(pos: Point, outlineOnly?: boolean) {
    if (outlineOnly) {
      this.hovering = false
      this.showOutline = true
    } else {
      const dist = distance(this.pos, pos)
      this.hovering = dist <= 2 * Reroute.slotRadius
      this.showOutline = dist <= 5 * Reroute.slotRadius
    }
  }

  /** Hides the slot. */
  hide() {
    this.hovering = false
    this.showOutline = false
  }

  /**
   * Draws the slot on the canvas.
   * @param ctx The canvas context to draw on.
   */
  draw(ctx: CanvasRenderingContext2D): void {
    const { fillStyle, strokeStyle, lineWidth } = ctx
    const {
      showOutline,
      hovering,
      pos: [x, y]
    } = this
    if (!showOutline) return

    try {
      ctx.fillStyle = hovering ? this.reroute.colour : 'rgba(127,127,127,0.3)'
      ctx.strokeStyle = 'rgb(0,0,0,0.5)'
      ctx.lineWidth = 1

      ctx.beginPath()
      ctx.arc(x, y, Reroute.slotRadius, 0, 2 * Math.PI)
      ctx.fill()
      ctx.stroke()
    } finally {
      ctx.fillStyle = fillStyle
      ctx.strokeStyle = strokeStyle
      ctx.lineWidth = lineWidth
    }
  }
}

/**
 * Retrieves the position of the next reroute in the chain, or the destination input slot on this link.
 * @param network The network of links
 * @param link The link representing the current reroute chain
 * @param id The ID of "this" reroute
 * @returns The position of the next reroute or the input slot target, otherwise `undefined`.
 */
function getNextPos(
  network: ReadonlyLinkNetwork,
  link: LLink | undefined,
  id: RerouteId
) {
  if (!link) return

  const linkPos = LLink.findNextReroute(network, link, id)?.pos
  if (linkPos) return linkPos

  // Floating link with no input to find
  if (link.target_id === -1 || link.target_slot === -1) return

  return network.getNodeById(link.target_id)?.getInputPos(link.target_slot)
}

/** Returns the direction from one point to another in radians. */
function getDirection(fromPos: Point, toPos: Point) {
  return Math.atan2(toPos[1] - fromPos[1], toPos[0] - fromPos[0])
}
