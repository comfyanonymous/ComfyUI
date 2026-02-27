import { LGraphCanvas } from '@/lib/litegraph/src/LGraphCanvas'
import type { LGraphNode } from '@/lib/litegraph/src/LGraphNode'
import type { LLink, LinkId } from '@/lib/litegraph/src/LLink'
import type { RerouteId } from '@/lib/litegraph/src/Reroute'
import { SlotShape } from '@/lib/litegraph/src/draw'
import { ConstrainedSize } from '@/lib/litegraph/src/infrastructure/ConstrainedSize'
import { Rectangle } from '@/lib/litegraph/src/infrastructure/Rectangle'
import type {
  DefaultConnectionColors,
  Hoverable,
  INodeInputSlot,
  INodeOutputSlot,
  Point,
  ReadOnlyRect,
  Size
} from '@/lib/litegraph/src/interfaces'
import { LiteGraph } from '@/lib/litegraph/src/litegraph'
import { SlotBase } from '@/lib/litegraph/src/node/SlotBase'
import type { CanvasPointerEvent } from '@/lib/litegraph/src/types/events'
import type {
  Serialisable,
  SubgraphIO
} from '@/lib/litegraph/src/types/serialisation'
import { createUuidv4 } from '@/lib/litegraph/src/utils/uuid'
import type { UUID } from '@/lib/litegraph/src/utils/uuid'

import type { SubgraphInput } from './SubgraphInput'
import type { SubgraphInputNode } from './SubgraphInputNode'
import type { SubgraphOutput } from './SubgraphOutput'
import type { SubgraphOutputNode } from './SubgraphOutputNode'

interface SubgraphSlotDrawOptions {
  ctx: CanvasRenderingContext2D
  colorContext: DefaultConnectionColors
  lowQuality?: boolean
  fromSlot?: INodeInputSlot | INodeOutputSlot | SubgraphInput | SubgraphOutput
  editorAlpha?: number
}

/** Shared base class for the slots used on Subgraph . */
export abstract class SubgraphSlot
  extends SlotBase
  implements SubgraphIO, Hoverable, Serialisable<SubgraphIO>
{
  static get defaultHeight() {
    return LiteGraph.NODE_SLOT_HEIGHT
  }

  private readonly _pos: Point = [0, 0]

  readonly measurement: ConstrainedSize = new ConstrainedSize(
    SubgraphSlot.defaultHeight,
    SubgraphSlot.defaultHeight
  )

  readonly id: UUID
  readonly parent: SubgraphInputNode | SubgraphOutputNode
  override type: string

  readonly linkIds: LinkId[] = []

  override readonly boundingRect: Rectangle = new Rectangle(
    0,
    0,
    0,
    SubgraphSlot.defaultHeight
  )

  override get pos() {
    return this._pos
  }

  override set pos(value) {
    if (!value || value.length < 2) return

    this._pos[0] = value[0]
    this._pos[1] = value[1]
  }

  /** Whether this slot is connected to another slot. */
  override get isConnected() {
    return this.linkIds.length > 0
  }

  /** The display name of this slot. */
  get displayName() {
    return this.label ?? this.localized_name ?? this.name
  }

  abstract get labelPos(): Point

  constructor(
    slot: SubgraphIO,
    parent: SubgraphInputNode | SubgraphOutputNode
  ) {
    super(slot.name, slot.type)

    Object.assign(this, slot)
    this.id = slot.id ?? createUuidv4()
    this.type = slot.type
    this.parent = parent
  }

  isPointerOver: boolean = false

  containsPoint(point: Point): boolean {
    return this.boundingRect.containsPoint(point)
  }

  onPointerMove(e: CanvasPointerEvent): void {
    this.isPointerOver = this.boundingRect.containsXy(e.canvasX, e.canvasY)
  }

  getLinks(): LLink[] {
    const links: LLink[] = []
    const { subgraph } = this.parent

    for (const id of this.linkIds) {
      const link = subgraph.getLink(id)
      if (link) links.push(link)
    }
    return links
  }

  decrementSlots(inputsOrOutputs: 'inputs' | 'outputs'): void {
    const { links } = this.parent.subgraph
    const linkProperty =
      inputsOrOutputs === 'inputs' ? 'origin_slot' : 'target_slot'

    for (const linkId of this.linkIds) {
      const link = links.get(linkId)
      if (link) link[linkProperty]--
      else console.warn('decrementSlots: link ID not found', linkId)
    }
  }

  measure(): Readonly<Size> {
    const width = LGraphCanvas._measureText?.(this.displayName) ?? 0

    const { defaultHeight } = SubgraphSlot
    this.measurement.setValues(width + defaultHeight, defaultHeight)
    return this.measurement.toSize()
  }

  abstract arrange(rect: ReadOnlyRect): void

  abstract connect(
    slot: INodeInputSlot | INodeOutputSlot,
    node: LGraphNode,
    afterRerouteId?: RerouteId
  ): LLink | undefined

  /**
   * Disconnects all links connected to this slot.
   */
  disconnect(): void {
    const { subgraph } = this.parent

    for (const linkId of this.linkIds) {
      subgraph.removeLink(linkId)
    }

    this.linkIds.length = 0
  }

  /**
   * Checks if this slot is a valid target for a connection from the given slot.
   * @param fromSlot The slot that is being dragged to connect to this slot.
   * @returns true if the connection is valid, false otherwise.
   */
  abstract isValidTarget(
    fromSlot: INodeInputSlot | INodeOutputSlot | SubgraphInput | SubgraphOutput
  ): boolean

  /** @remarks Leaves the context dirty. */
  draw({
    ctx,
    colorContext,
    lowQuality,
    fromSlot,
    editorAlpha = 1
  }: SubgraphSlotDrawOptions): void {
    // Assertion: SlotShape is a subset of RenderShape
    const shape = this.shape as unknown as SlotShape
    const {
      isPointerOver,
      pos: [x, y]
    } = this

    // Check if this slot is a valid target for the current dragging connection
    const isValidTarget = fromSlot ? this.isValidTarget(fromSlot) : true
    const isValid = !fromSlot || isValidTarget

    // Only highlight if the slot is valid AND mouse is over it
    const highlight = isValid && isPointerOver

    // Save current alpha
    const previousAlpha = ctx.globalAlpha

    // Set opacity based on validity when dragging a connection
    ctx.globalAlpha = isValid ? editorAlpha : 0.4 * editorAlpha

    ctx.beginPath()

    // Default rendering for circle, hollow circle.
    const color = this.renderingColor(colorContext)
    if (lowQuality) {
      ctx.fillStyle = color

      ctx.rect(x - 4, y - 4, 8, 8)
      ctx.fill()
    } else if (shape === SlotShape.HollowCircle) {
      ctx.lineWidth = 3
      ctx.strokeStyle = color

      const radius = highlight ? 4 : 3
      ctx.arc(x, y, radius, 0, Math.PI * 2)
      ctx.stroke()
    } else {
      // Normal circle
      ctx.fillStyle = color

      const radius = highlight ? 5 : 4
      ctx.arc(x, y, radius, 0, Math.PI * 2)
      ctx.fill()
    }

    // Draw label with current opacity
    if (this.displayName) {
      const [labelX, labelY] = this.labelPos
      // Also apply highlight logic to text color
      ctx.fillStyle = highlight ? 'white' : LiteGraph.NODE_TEXT_COLOR || '#AAA'
      ctx.fillText(this.displayName, labelX, labelY)
    }

    // Restore alpha
    ctx.globalAlpha = previousAlpha
  }

  asSerialisable(): SubgraphIO {
    const {
      id,
      name,
      type,
      linkIds,
      localized_name,
      label,
      dir,
      shape,
      color_off,
      color_on,
      pos
    } = this
    return {
      id,
      name,
      type,
      linkIds,
      localized_name,
      label,
      dir,
      shape,
      color_off,
      color_on,
      pos
    }
  }
}
