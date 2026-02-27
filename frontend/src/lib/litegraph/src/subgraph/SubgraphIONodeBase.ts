import type { NodeId } from '@/lib/litegraph/src/LGraphNode'
import type { LinkConnector } from '@/lib/litegraph/src/canvas/LinkConnector'
import { Rectangle } from '@/lib/litegraph/src/infrastructure/Rectangle'
import type {
  DefaultConnectionColors,
  Hoverable,
  INodeInputSlot,
  INodeOutputSlot,
  Point,
  Positionable
} from '@/lib/litegraph/src/interfaces'
import { LiteGraph } from '@/lib/litegraph/src/litegraph'
import type {
  CanvasColour,
  CanvasPointer,
  CanvasPointerEvent,
  IContextMenuValue
} from '@/lib/litegraph/src/litegraph'
import { snapPoint } from '@/lib/litegraph/src/measure'
import { CanvasItem } from '@/lib/litegraph/src/types/globalEnums'
import type {
  ExportedSubgraphIONode,
  Serialisable
} from '@/lib/litegraph/src/types/serialisation'

import type { EmptySubgraphInput } from './EmptySubgraphInput'
import type { EmptySubgraphOutput } from './EmptySubgraphOutput'
import type { Subgraph } from './Subgraph'
import type { SubgraphInput } from './SubgraphInput'
import type { SubgraphOutput } from './SubgraphOutput'

export abstract class SubgraphIONodeBase<
  TSlot extends SubgraphInput | SubgraphOutput
>
  implements Positionable, Hoverable, Serialisable<ExportedSubgraphIONode>
{
  static margin = 10
  static minWidth = 100
  static roundedRadius = 10

  private readonly _boundingRect: Rectangle = new Rectangle()

  abstract readonly id: NodeId

  get boundingRect(): Rectangle {
    return this._boundingRect
  }

  selected: boolean = false
  pinned: boolean = false
  readonly removable = false

  isPointerOver: boolean = false

  abstract readonly emptySlot: EmptySubgraphInput | EmptySubgraphOutput

  get pos() {
    return this.boundingRect.pos
  }

  set pos(value) {
    this.boundingRect.pos = value
  }

  get size() {
    return this.boundingRect.size
  }

  set size(value) {
    this.boundingRect.size = value
  }

  protected get sideLineWidth(): number {
    return this.isPointerOver ? 2.5 : 2
  }

  protected get sideStrokeStyle(): CanvasColour {
    return this.isPointerOver ? 'white' : '#efefef'
  }

  abstract readonly slots: TSlot[]
  abstract get allSlots(): TSlot[]

  constructor(
    /** The subgraph that this node belongs to. */
    readonly subgraph: Subgraph
  ) {}

  move(deltaX: number, deltaY: number): void {
    this.pos[0] += deltaX
    this.pos[1] += deltaY
  }

  /** @inheritdoc */
  snapToGrid(snapTo: number): boolean {
    return this.pinned ? false : snapPoint(this.pos, snapTo)
  }

  abstract onPointerDown(
    e: CanvasPointerEvent,
    pointer: CanvasPointer,
    linkConnector: LinkConnector
  ): void

  // #region Hoverable

  containsPoint(point: Point): boolean {
    return this.boundingRect.containsPoint(point)
  }

  abstract get slotAnchorX(): number

  onPointerMove(e: CanvasPointerEvent): CanvasItem {
    const containsPoint = this.boundingRect.containsXy(e.canvasX, e.canvasY)
    let underPointer = containsPoint
      ? CanvasItem.SubgraphIoNode
      : CanvasItem.Nothing

    if (containsPoint) {
      if (!this.isPointerOver) this.onPointerEnter()

      for (const slot of this.allSlots) {
        slot.onPointerMove(e)
        if (slot.isPointerOver) underPointer |= CanvasItem.SubgraphIoSlot
      }
    } else if (this.isPointerOver) {
      this.onPointerLeave()
    }
    return underPointer
  }

  onPointerEnter() {
    this.isPointerOver = true
  }

  onPointerLeave() {
    this.isPointerOver = false

    for (const slot of this.slots) {
      slot.isPointerOver = false
    }
  }

  // #endregion Hoverable

  /**
   * Renames an IO slot in the subgraph.
   * @param slot The slot to rename.
   * @param name The new name for the slot.
   */
  abstract renameSlot(slot: TSlot, name: string): void

  /**
   * Removes an IO slot from the subgraph.
   * @param slot The slot to remove.
   */
  abstract removeSlot(slot: TSlot): void

  /**
   * Gets the slot at a given position in canvas space.
   * @param x The x coordinate of the position.
   * @param y The y coordinate of the position.
   * @returns The slot at the given position, otherwise `undefined`.
   */
  getSlotInPosition(x: number, y: number): TSlot | undefined {
    for (const slot of this.allSlots) {
      if (slot.boundingRect.containsXy(x, y)) {
        return slot
      }
    }
  }

  /**
   * Handles double-click on an IO slot to rename it.
   * @param slot The slot that was double-clicked.
   * @param event The event that triggered the double-click.
   */
  protected handleSlotDoubleClick(
    slot: TSlot,
    event: CanvasPointerEvent
  ): void {
    // Only allow renaming non-empty slots
    if (slot !== this.emptySlot) {
      this._promptForSlotRename(slot, event)
    }
  }

  /**
   * Shows the context menu for an IO slot.
   * @param slot The slot to show the context menu for.
   * @param event The event that triggered the context menu.
   */
  protected showSlotContextMenu(slot: TSlot, event: CanvasPointerEvent): void {
    const options: (IContextMenuValue | null)[] = this._getSlotMenuOptions(slot)
    if (!(options.length > 0)) return

    new LiteGraph.ContextMenu(options, {
      event,
      title: slot.name || 'Subgraph Output',
      callback: (item: IContextMenuValue) => {
        this._onSlotMenuAction(item, slot, event)
      }
    })
  }

  /**
   * Gets the context menu options for an IO slot.
   * @param slot The slot to get the context menu options for.
   * @returns The context menu options.
   */
  private _getSlotMenuOptions(slot: TSlot): (IContextMenuValue | null)[] {
    const options: (IContextMenuValue | null)[] = []

    // Disconnect option if slot has connections
    if (slot !== this.emptySlot && slot.linkIds.length > 0) {
      options.push({ content: 'Disconnect Links', value: 'disconnect' })
    }

    // Rename slot option (except for the empty slot)
    if (slot !== this.emptySlot) {
      options.push({ content: 'Rename Slot', value: 'rename' })
    }

    if (slot !== this.emptySlot) {
      options.push(null) // separator
      options.push({
        content: 'Remove Slot',
        value: 'remove',
        className: 'danger'
      })
    }

    return options
  }

  /**
   * Handles the action for an IO slot context menu.
   * @param selectedItem The item that was selected from the context menu.
   * @param slot The slot
   * @param event The event that triggered the context menu.
   */
  private _onSlotMenuAction(
    selectedItem: IContextMenuValue,
    slot: TSlot,
    event: CanvasPointerEvent
  ): void {
    switch (selectedItem.value) {
      // Disconnect all links from this output
      case 'disconnect':
        slot.disconnect()
        break

      // Remove the slot
      case 'remove':
        if (slot !== this.emptySlot) {
          this.removeSlot(slot)
        }
        break

      // Rename the slot
      case 'rename':
        if (slot !== this.emptySlot) {
          this._promptForSlotRename(slot, event)
        }
        break
    }

    this.subgraph.setDirtyCanvas(true)
  }

  /**
   * Prompts the user to rename a slot.
   * @param slot The slot to rename.
   * @param event The event that triggered the rename.
   */
  private _promptForSlotRename(slot: TSlot, event: CanvasPointerEvent): void {
    this.subgraph.canvasAction((c) =>
      c.prompt(
        'Slot name',
        slot.displayName,
        (newName: string) => {
          if (newName) this.renameSlot(slot, newName)
        },
        event
      )
    )
  }

  /** Arrange the slots in this node. */
  arrange(): void {
    const { minWidth, roundedRadius } = SubgraphIONodeBase
    const [, y] = this.boundingRect
    const x = this.slotAnchorX
    const { size } = this

    let maxWidth = minWidth
    let currentY = y + roundedRadius

    for (const slot of this.allSlots) {
      const [slotWidth, slotHeight] = slot.measure()
      slot.arrange([x, currentY, slotWidth, slotHeight])

      currentY += slotHeight
      if (slotWidth > maxWidth) maxWidth = slotWidth
    }

    size[0] = maxWidth + 2 * roundedRadius
    size[1] = currentY - y + roundedRadius
  }

  draw(
    ctx: CanvasRenderingContext2D,
    colorContext: DefaultConnectionColors,
    fromSlot?:
      | INodeInputSlot
      | INodeOutputSlot
      | SubgraphInput
      | SubgraphOutput,
    editorAlpha?: number
  ): void {
    const { lineWidth, strokeStyle, fillStyle, font, textBaseline } = ctx
    this.drawProtected(ctx, colorContext, fromSlot, editorAlpha)
    Object.assign(ctx, {
      lineWidth,
      strokeStyle,
      fillStyle,
      font,
      textBaseline
    })
  }

  /** @internal Leaves {@link ctx} dirty. */
  protected abstract drawProtected(
    ctx: CanvasRenderingContext2D,
    colorContext: DefaultConnectionColors,
    fromSlot?:
      | INodeInputSlot
      | INodeOutputSlot
      | SubgraphInput
      | SubgraphOutput,
    editorAlpha?: number
  ): void

  /** @internal Leaves {@link ctx} dirty. */
  protected drawSlots(
    ctx: CanvasRenderingContext2D,
    colorContext: DefaultConnectionColors,
    fromSlot?:
      | INodeInputSlot
      | INodeOutputSlot
      | SubgraphInput
      | SubgraphOutput,
    editorAlpha?: number
  ): void {
    ctx.fillStyle = '#AAA'
    ctx.font = '12px Inter, sans-serif'
    ctx.textBaseline = 'middle'

    for (const slot of this.allSlots) {
      slot.draw({ ctx, colorContext, fromSlot, editorAlpha })
    }
  }

  configure(data: ExportedSubgraphIONode): void {
    this._boundingRect.set(data.bounding)
    this.pinned = data.pinned ?? false
  }

  asSerialisable(): ExportedSubgraphIONode {
    return {
      id: this.id,
      bounding: this.boundingRect.export(),
      pinned: this.pinned ? true : undefined
    }
  }
}
