import type {
  CompassCorners,
  Point,
  ReadOnlyRect,
  ReadOnlyTypedArray,
  Size
} from '@/lib/litegraph/src/interfaces'
import { isInRectangle } from '@/lib/litegraph/src/measure'

/**
 * A rectangle, represented as a float64 array of 4 numbers: [x, y, width, height].
 *
 * This class is a subclass of Float64Array, and so has all the methods of that class.  Notably,
 * {@link Rectangle.from} can be used to convert a {@link ReadOnlyRect}. Typing of this however,
 * is broken due to the base TS lib returning Float64Array rather than `this`.
 *
 * Sub-array properties ({@link Float64Array.subarray}):
 * - {@link pos}: The position of the top-left corner of the rectangle.
 * - {@link size}: The size of the rectangle.
 */
export class Rectangle extends Float64Array {
  private _pos: Float64Array<ArrayBuffer> | undefined
  private _size: Float64Array<ArrayBuffer> | undefined

  constructor(
    x: number = 0,
    y: number = 0,
    width: number = 0,
    height: number = 0
  ) {
    super(4)

    this[0] = x
    this[1] = y
    this[2] = width
    this[3] = height
  }

  static override from([x, y, width, height]: ReadOnlyRect): Rectangle {
    return new Rectangle(x, y, width, height)
  }

  /**
   * Creates a new rectangle positioned at the given centre, with the given width/height.
   * @param centre The centre of the rectangle, as an `[x, y]` point
   * @param width The width of the rectangle
   * @param height The height of the rectangle.  Default: {@link width}
   * @returns A new rectangle whose centre is at {@link x}
   */
  static fromCentre(
    [x, y]: Readonly<Point>,
    width: number,
    height = width
  ): Rectangle {
    const left = x - width * 0.5
    const top = y - height * 0.5
    return new Rectangle(left, top, width, height)
  }

  static ensureRect(rect: ReadOnlyRect): Rectangle {
    return rect instanceof Rectangle
      ? rect
      : new Rectangle(rect[0], rect[1], rect[2], rect[3])
  }

  override subarray(
    begin: number = 0,
    end?: number
  ): Float64Array<ArrayBuffer> {
    const byteOffset = begin << 3
    const length = end === undefined ? end : end - begin
    return new Float64Array(this.buffer, byteOffset, length)
  }

  /**
   * A reference to the position of the top-left corner of this rectangle.
   *
   * Updating the values of the returned object will update this rectangle.
   */
  get pos(): Point {
    this._pos ??= this.subarray(0, 2)
    return this._pos! as unknown as Point
  }

  set pos(value: Readonly<Point>) {
    this[0] = value[0]
    this[1] = value[1]
  }

  /**
   * A reference to the size of this rectangle.
   *
   * Updating the values of the returned object will update this rectangle.
   */
  get size(): Size {
    this._size ??= this.subarray(2, 4)
    return this._size! as unknown as Size
  }

  set size(value: Readonly<Size>) {
    this[2] = value[0]
    this[3] = value[1]
  }

  // #region Property accessors
  /** The x co-ordinate of the top-left corner of this rectangle. */
  get x() {
    return this[0]
  }

  set x(value: number) {
    this[0] = value
  }

  /** The y co-ordinate of the top-left corner of this rectangle. */
  get y() {
    return this[1]
  }

  set y(value: number) {
    this[1] = value
  }

  /** The width of this rectangle. */
  get width() {
    return this[2]
  }

  set width(value: number) {
    this[2] = value
  }

  /** The height of this rectangle. */
  get height() {
    return this[3]
  }

  set height(value: number) {
    this[3] = value
  }

  /** The x co-ordinate of the left edge of this rectangle. */
  get left() {
    return this[0]
  }

  set left(value: number) {
    this[0] = value
  }

  /** The y co-ordinate of the top edge of this rectangle. */
  get top() {
    return this[1]
  }

  set top(value: number) {
    this[1] = value
  }

  /** The x co-ordinate of the right edge of this rectangle. */
  get right() {
    return this[0] + this[2]
  }

  set right(value: number) {
    this[0] = value - this[2]
  }

  /** The y co-ordinate of the bottom edge of this rectangle. */
  get bottom() {
    return this[1] + this[3]
  }

  set bottom(value: number) {
    this[1] = value - this[3]
  }

  /** The x co-ordinate of the centre of this rectangle. */
  get centreX() {
    return this[0] + this[2] * 0.5
  }

  /** The y co-ordinate of the centre of this rectangle. */
  get centreY() {
    return this[1] + this[3] * 0.5
  }
  // #endregion Property accessors

  /**
   * Updates the rectangle to the values of {@link rect}.
   * @param rect The rectangle to update to.
   */
  updateTo(rect: ReadOnlyRect) {
    this[0] = rect[0]
    this[1] = rect[1]
    this[2] = rect[2]
    this[3] = rect[3]
  }

  /**
   * Checks if the point [{@link x}, {@link y}] is inside this rectangle.
   * @param x The x-coordinate to check
   * @param y The y-coordinate to check
   * @returns `true` if the point is inside this rectangle, otherwise `false`.
   */
  containsXy(x: number, y: number): boolean {
    const [left, top, width, height] = this
    return x >= left && x < left + width && y >= top && y < top + height
  }

  /**
   * Checks if {@link point} is inside this rectangle.
   * @param point The point to check
   * @returns `true` if {@link point} is inside this rectangle, otherwise `false`.
   */
  containsPoint([x, y]: Readonly<Point>): boolean {
    const [left, top, width, height] = this
    return x >= left && x < left + width && y >= top && y < top + height
  }

  /**
   * Checks if {@link other} is a smaller rectangle inside this rectangle.
   * One **must** be larger than the other; identical rectangles are not considered to contain each other.
   * @param other The rectangle to check
   * @returns `true` if {@link other} is inside this rectangle, otherwise `false`.
   */
  containsRect(other: ReadOnlyRect): boolean {
    const { right, bottom } = this
    const otherRight = other[0] + other[2]
    const otherBottom = other[1] + other[3]

    const identical =
      this.x === other[0] &&
      this.y === other[1] &&
      right === otherRight &&
      bottom === otherBottom

    return (
      !identical &&
      this.x <= other[0] &&
      this.y <= other[1] &&
      right >= otherRight &&
      bottom >= otherBottom
    )
  }

  /**
   * Checks if {@link rect} overlaps with this rectangle.
   * @param rect The rectangle to check
   * @returns `true` if {@link rect} overlaps with this rectangle, otherwise `false`.
   */
  overlaps(rect: ReadOnlyRect): boolean {
    return (
      this.x < rect[0] + rect[2] &&
      this.y < rect[1] + rect[3] &&
      this.x + this.width > rect[0] &&
      this.y + this.height > rect[1]
    )
  }

  /**
   * Finds the corner (if any) of this rectangle that contains the point [{@link x}, {@link y}].
   * @param x The x-coordinate to check
   * @param y The y-coordinate to check
   * @param cornerSize Each corner is treated as an inset square with this width and height.
   * @returns The compass direction of the corner that contains the point, or `undefined` if the point is not in any corner.
   */
  findContainingCorner(
    x: number,
    y: number,
    cornerSize: number
  ): CompassCorners | undefined {
    if (this.isInTopLeftCorner(x, y, cornerSize)) return 'NW'
    if (this.isInTopRightCorner(x, y, cornerSize)) return 'NE'
    if (this.isInBottomLeftCorner(x, y, cornerSize)) return 'SW'
    if (this.isInBottomRightCorner(x, y, cornerSize)) return 'SE'
  }

  /** @returns `true` if the point [{@link x}, {@link y}] is in the top-left corner of this rectangle, otherwise `false`. */
  isInTopLeftCorner(x: number, y: number, cornerSize: number): boolean {
    return isInRectangle(x, y, this.x, this.y, cornerSize, cornerSize)
  }

  /** @returns `true` if the point [{@link x}, {@link y}] is in the top-right corner of this rectangle, otherwise `false`. */
  isInTopRightCorner(x: number, y: number, cornerSize: number): boolean {
    return isInRectangle(
      x,
      y,
      this.right - cornerSize,
      this.y,
      cornerSize,
      cornerSize
    )
  }

  /** @returns `true` if the point [{@link x}, {@link y}] is in the bottom-left corner of this rectangle, otherwise `false`. */
  isInBottomLeftCorner(x: number, y: number, cornerSize: number): boolean {
    return isInRectangle(
      x,
      y,
      this.x,
      this.bottom - cornerSize,
      cornerSize,
      cornerSize
    )
  }

  /** @returns `true` if the point [{@link x}, {@link y}] is in the bottom-right corner of this rectangle, otherwise `false`. */
  isInBottomRightCorner(x: number, y: number, cornerSize: number): boolean {
    return isInRectangle(
      x,
      y,
      this.right - cornerSize,
      this.bottom - cornerSize,
      cornerSize,
      cornerSize
    )
  }

  /** @returns `true` if the point [{@link x}, {@link y}] is in the top edge of this rectangle, otherwise `false`. */
  isInTopEdge(x: number, y: number, edgeSize: number): boolean {
    return isInRectangle(x, y, this.x, this.y, this.width, edgeSize)
  }

  /** @returns `true` if the point [{@link x}, {@link y}] is in the bottom edge of this rectangle, otherwise `false`. */
  isInBottomEdge(x: number, y: number, edgeSize: number): boolean {
    return isInRectangle(
      x,
      y,
      this.x,
      this.bottom - edgeSize,
      this.width,
      edgeSize
    )
  }

  /** @returns `true` if the point [{@link x}, {@link y}] is in the left edge of this rectangle, otherwise `false`. */
  isInLeftEdge(x: number, y: number, edgeSize: number): boolean {
    return isInRectangle(x, y, this.x, this.y, edgeSize, this.height)
  }

  /** @returns `true` if the point [{@link x}, {@link y}] is in the right edge of this rectangle, otherwise `false`. */
  isInRightEdge(x: number, y: number, edgeSize: number): boolean {
    return isInRectangle(
      x,
      y,
      this.right - edgeSize,
      this.y,
      edgeSize,
      this.height
    )
  }

  /** @returns The centre point of this rectangle, as a new {@link Point}. */
  getCentre(): Point {
    return [this.centreX, this.centreY]
  }

  /** @returns The area of this rectangle. */
  getArea(): number {
    return this.width * this.height
  }

  /** @returns The perimeter of this rectangle. */
  getPerimeter(): number {
    return 2 * (this.width + this.height)
  }

  /** @returns The top-left corner of this rectangle, as a new {@link Point}. */
  getTopLeft(): Point {
    return [this[0], this[1]]
  }

  /** @returns The bottom-right corner of this rectangle, as a new {@link Point}. */
  getBottomRight(): Point {
    return [this.right, this.bottom]
  }

  /** @returns The width and height of this rectangle, as a new {@link Size}. */
  getSize(): Size {
    return [this[2], this[3]]
  }

  /** @returns The offset from the top-left of this rectangle to the point [{@link x}, {@link y}], as a new {@link Point}. */
  getOffsetTo([x, y]: Readonly<Point>): Point {
    return [x - this[0], y - this[1]]
  }

  /** @returns The offset from the point [{@link x}, {@link y}] to the top-left of this rectangle, as a new {@link Point}. */
  getOffsetFrom([x, y]: Readonly<Point>): Point {
    return [this[0] - x, this[1] - y]
  }

  /** Resizes the rectangle without moving it, setting its top-left corner to [{@link x}, {@link y}]. */
  resizeTopLeft(x1: number, y1: number) {
    this[2] += this[0] - x1
    this[3] += this[1] - y1

    this[0] = x1
    this[1] = y1
  }

  /** Resizes the rectangle without moving it, setting its bottom-left corner to [{@link x}, {@link y}]. */
  resizeBottomLeft(x1: number, y2: number) {
    this[2] += this[0] - x1
    this[3] = y2 - this[1]

    this[0] = x1
  }

  /** Resizes the rectangle without moving it, setting its top-right corner to [{@link x}, {@link y}]. */
  resizeTopRight(x2: number, y1: number) {
    this[2] = x2 - this[0]
    this[3] += this[1] - y1

    this[1] = y1
  }

  /** Resizes the rectangle without moving it, setting its bottom-right corner to [{@link x}, {@link y}]. */
  resizeBottomRight(x2: number, y2: number) {
    this[2] = x2 - this[0]
    this[3] = y2 - this[1]
  }

  /** Sets the width without moving the right edge (changes position) */
  setWidthRightAnchored(width: number) {
    const currentWidth = this[2]
    this[2] = width
    this[0] += currentWidth - width
  }

  /** Sets the height without moving the bottom edge (changes position) */
  setHeightBottomAnchored(height: number) {
    const currentHeight = this[3]
    this[3] = height
    this[1] += currentHeight - height
  }

  clone(): Rectangle {
    return new Rectangle(this[0], this[1], this[2], this[3])
  }

  /** Alias of {@link export}. */
  toArray() {
    return this.export()
  }

  /** @returns A new, untyped array (serializable) containing the values of this rectangle. */
  export(): [number, number, number, number] {
    return [this[0], this[1], this[2], this[3]]
  }

  /**
   * Draws a debug outline of this rectangle.
   * @internal Convenience debug/development interface; not for production use.
   */
  _drawDebug(ctx: CanvasRenderingContext2D, colour = 'red') {
    const { strokeStyle, lineWidth } = ctx
    try {
      ctx.strokeStyle = colour
      ctx.lineWidth = 0.5
      ctx.beginPath()
      ctx.strokeRect(this[0], this[1], this[2], this[3])
    } finally {
      ctx.strokeStyle = strokeStyle
      ctx.lineWidth = lineWidth
    }
  }
}

export type ReadOnlyRectangle = Omit<
  ReadOnlyTypedArray<Rectangle>,
  | 'setHeightBottomAnchored'
  | 'setWidthRightAnchored'
  | 'resizeTopLeft'
  | 'resizeBottomLeft'
  | 'resizeTopRight'
  | 'resizeBottomRight'
  | 'resizeBottomRight'
  | 'updateTo'
>
