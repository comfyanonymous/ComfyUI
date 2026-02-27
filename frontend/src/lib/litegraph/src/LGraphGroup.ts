import { NullGraphError } from '@/lib/litegraph/src/infrastructure/NullGraphError'

import type { LGraph } from './LGraph'
import { LGraphCanvas } from './LGraphCanvas'
import { LGraphNode } from './LGraphNode'
import { strokeShape } from './draw'
import type {
  ColorOption,
  IColorable,
  IContextMenuValue,
  IPinnable,
  Point,
  Positionable,
  Size
} from './interfaces'
import { LiteGraph, Rectangle } from './litegraph'
import {
  containsCentre,
  containsRect,
  createBounds,
  isInRect,
  isInRectangle,
  isPointInRect,
  snapPoint
} from './measure'
import type { ISerialisedGroup } from './types/serialisation'

export interface IGraphGroupFlags extends Record<string, unknown> {
  pinned?: true
}

export class LGraphGroup implements Positionable, IPinnable, IColorable {
  static minWidth = 140
  static minHeight = 80
  static resizeLength = 10
  static padding = 4
  static defaultColour = '#335'

  id: number
  color?: string
  title: string
  font?: string
  font_size: number = LiteGraph.DEFAULT_GROUP_FONT || 24
  _bounding = new Rectangle(10, 10, LGraphGroup.minWidth, LGraphGroup.minHeight)

  _pos: Point = this._bounding.pos
  _size: Size = this._bounding.size
  /** @deprecated See {@link _children} */
  _nodes: LGraphNode[] = []
  _children: Set<Positionable> = new Set()
  graph?: LGraph
  flags: IGraphGroupFlags = {}
  selected?: boolean

  constructor(title?: string, id?: number) {
    // TODO: Object instantiation pattern requires too much boilerplate and null checking.  ID should be passed in via constructor.
    this.id = id ?? -1
    this.title = title || 'Group'

    const { pale_blue } = LGraphCanvas.node_colors
    this.color = pale_blue ? pale_blue.groupcolor : '#AAA'
  }

  /** @inheritdoc {@link IColorable.setColorOption} */
  setColorOption(colorOption: ColorOption | null): void {
    if (colorOption == null) {
      delete this.color
    } else {
      this.color = colorOption.groupcolor
    }
  }

  /** @inheritdoc {@link IColorable.getColorOption} */
  getColorOption(): ColorOption | null {
    return (
      Object.values(LGraphCanvas.node_colors).find(
        (colorOption) => colorOption.groupcolor === this.color
      ) ?? null
    )
  }

  /** Position of the group, as x,y co-ordinates in graph space */
  get pos() {
    return this._pos
  }

  set pos(v) {
    if (!v || v.length < 2) return

    this._pos[0] = v[0]
    this._pos[1] = v[1]
  }

  /** Size of the group, as width,height in graph units */
  get size() {
    return this._size
  }

  set size(v) {
    if (!v || v.length < 2) return

    this._size[0] = Math.max(LGraphGroup.minWidth, v[0])
    this._size[1] = Math.max(LGraphGroup.minHeight, v[1])
  }

  get boundingRect() {
    return this._bounding
  }

  getBounding() {
    return this._bounding
  }

  get nodes() {
    return this._nodes
  }

  get titleHeight() {
    return this.font_size * 1.4
  }

  get children(): ReadonlySet<Positionable> {
    return this._children
  }

  get pinned() {
    return !!this.flags.pinned
  }

  /**
   * Prevents the group being accidentally moved or resized by mouse interaction.
   * Toggles pinned state if no value is provided.
   */
  pin(value?: boolean): void {
    const newState = value === undefined ? !this.pinned : value

    if (newState) this.flags.pinned = true
    else delete this.flags.pinned
  }

  unpin(): void {
    this.pin(false)
  }

  configure(o: ISerialisedGroup): void {
    this.id = o.id
    this.title = o.title
    this._bounding.set(o.bounding)
    this.color = o.color
    this.flags = o.flags || this.flags
    if (o.font_size) this.font_size = o.font_size
  }

  serialize(): ISerialisedGroup {
    const b = this._bounding
    return {
      id: this.id,
      title: this.title,
      bounding: [...b],
      color: this.color,
      font_size: this.font_size,
      flags: this.flags
    }
  }

  /**
   * Draws the group on the canvas
   * @param graphCanvas
   * @param ctx
   */
  draw(graphCanvas: LGraphCanvas, ctx: CanvasRenderingContext2D): void {
    const { padding, resizeLength, defaultColour } = LGraphGroup
    const font_size = this.font_size || LiteGraph.DEFAULT_GROUP_FONT_SIZE

    const [x, y] = this._pos
    const [width, height] = this._size
    const color = this.color || defaultColour

    // Titlebar
    ctx.globalAlpha = 0.25 * graphCanvas.editor_alpha
    ctx.fillStyle = color
    ctx.strokeStyle = color
    ctx.beginPath()
    ctx.rect(x + 0.5, y + 0.5, width, font_size * 1.4)
    ctx.fill()

    // Group background, border
    ctx.fillStyle = color
    ctx.strokeStyle = color
    ctx.beginPath()
    ctx.rect(x + 0.5, y + 0.5, width, height)
    ctx.fill()
    ctx.globalAlpha = graphCanvas.editor_alpha
    ctx.stroke()

    // Resize marker
    ctx.beginPath()
    ctx.moveTo(x + width, y + height)
    ctx.lineTo(x + width - resizeLength, y + height)
    ctx.lineTo(x + width, y + height - resizeLength)
    ctx.fill()

    // Title
    ctx.font = `${font_size}px ${LiteGraph.GROUP_FONT}`
    ctx.textAlign = 'left'
    ctx.fillText(
      this.title + (this.pinned ? '📌' : ''),
      x + padding,
      y + font_size
    )

    if (LiteGraph.highlight_selected_group && this.selected) {
      strokeShape(ctx, this._bounding, {
        title_height: this.titleHeight,
        padding
      })
    }
  }

  resize(width: number, height: number): boolean {
    if (this.pinned) return false

    this._size[0] = Math.max(LGraphGroup.minWidth, width)
    this._size[1] = Math.max(LGraphGroup.minHeight, height)
    return true
  }

  move(deltaX: number, deltaY: number, skipChildren: boolean = false): void {
    if (this.pinned) return

    this._pos[0] += deltaX
    this._pos[1] += deltaY
    if (skipChildren === true) return

    for (const item of this._children) {
      item.move(deltaX, deltaY)
    }
  }

  /** @inheritdoc */
  snapToGrid(snapTo: number): boolean {
    return this.pinned ? false : snapPoint(this.pos, snapTo)
  }

  /**
   * Recomputes which items (nodes, reroutes, nested groups) are inside this group.
   * Recursively processes nested groups to ensure their children are also computed.
   * @param maxDepth Maximum recursion depth for nested groups. Use 1 to skip nested group computation.
   * @param visited Set of already visited group IDs to prevent redundant computation.
   */
  recomputeInsideNodes(
    maxDepth: number = 100,
    visited: Set<number> = new Set()
  ): void {
    if (!this.graph) throw new NullGraphError()
    if (maxDepth <= 0 || visited.has(this.id)) return

    visited.add(this.id)

    const { nodes, reroutes, groups } = this.graph
    const children = this._children
    this._nodes.length = 0
    children.clear()

    // Move nodes we overlap the centre point of
    for (const node of nodes) {
      if (containsCentre(this._bounding, node.boundingRect)) {
        this._nodes.push(node)
        children.add(node)
      }
    }

    // Move reroutes we overlap the centre point of
    for (const reroute of reroutes.values()) {
      if (isPointInRect(reroute.pos, this._bounding)) children.add(reroute)
    }

    // Move groups we wholly contain and recursively compute their children
    const containedGroups: LGraphGroup[] = []
    for (const group of groups) {
      if (group !== this && containsRect(this._bounding, group._bounding)) {
        children.add(group)
        containedGroups.push(group)
      }
    }
    for (const group of containedGroups)
      group.recomputeInsideNodes(maxDepth - 1, visited)

    groups.sort((a, b) => {
      if (a === this) {
        return children.has(b) ? -1 : 0
      } else if (b === this) {
        return children.has(a) ? 1 : 0
      } else {
        return 0
      }
    })
  }

  /**
   * Resizes and moves the group to neatly fit all given {@link objects}.
   * @param objects All objects that should be inside the group
   * @param padding Value in graph units to add to all sides of the group.  Default: 10
   */
  resizeTo(objects: Iterable<Positionable>, padding: number = 10): void {
    const boundingBox = createBounds(objects, padding)
    if (boundingBox === null) return

    this.pos[0] = boundingBox[0]
    this.pos[1] = boundingBox[1] - this.titleHeight
    this.size[0] = boundingBox[2]
    this.size[1] = boundingBox[3] + this.titleHeight
  }

  /**
   * Add nodes to the group and adjust the group's position and size accordingly
   * @param nodes The nodes to add to the group
   * @param padding The padding around the group
   */
  addNodes(nodes: LGraphNode[], padding: number = 10): void {
    if (!this._nodes && nodes.length === 0) return
    this.resizeTo([...this.children, ...this._nodes, ...nodes], padding)
  }

  getMenuOptions(): (
    | IContextMenuValue<string>
    | IContextMenuValue<string | null>
    | null
  )[] {
    return [
      {
        content: this.pinned ? 'Unpin' : 'Pin',
        callback: () => {
          if (this.pinned) this.unpin()
          else this.pin()
          this.setDirtyCanvas(false, true)
        }
      },
      null,
      { content: 'Title', callback: LGraphCanvas.onShowPropertyEditor },
      {
        content: 'Color',
        has_submenu: true,
        callback: LGraphCanvas.onMenuNodeColors
      },
      {
        content: 'Font size',
        property: 'font_size',
        type: 'Number',
        callback: LGraphCanvas.onShowPropertyEditor
      },
      null,
      { content: 'Remove', callback: LGraphCanvas.onMenuNodeRemove }
    ]
  }

  isPointInTitlebar(x: number, y: number): boolean {
    const b = this.boundingRect
    return isInRectangle(x, y, b[0], b[1], b[2], this.titleHeight)
  }

  isInResize(x: number, y: number): boolean {
    const b = this.boundingRect
    const right = b[0] + b[2]
    const bottom = b[1] + b[3]

    return (
      x < right &&
      y < bottom &&
      x - right + (y - bottom) > -LGraphGroup.resizeLength
    )
  }

  isPointInside(x: number, y: number): boolean {
    return isInRect(x, y, this.boundingRect)
  }
  setDirtyCanvas = LGraphNode.prototype.setDirtyCanvas
}
