import { clamp } from 'es-toolkit/compat'

import type { ReadOnlyRect, Size } from '@/lib/litegraph/src/interfaces'

/**
 * Basic width and height, with min/max constraints.
 *
 * - The {@link width} and {@link height} properties are readonly
 * - Size is set via {@link desiredWidth} and {@link desiredHeight} properties
 * - Width and height are then updated, clamped to min/max values
 */
export class ConstrainedSize {
  private _width: number = 0
  private _height: number = 0
  private _desiredWidth: number = 0
  private _desiredHeight: number = 0

  minWidth: number = 0
  minHeight: number = 0
  maxWidth: number = Infinity
  maxHeight: number = Infinity

  get width() {
    return this._width
  }

  get height() {
    return this._height
  }

  get desiredWidth() {
    return this._desiredWidth
  }

  set desiredWidth(value: number) {
    this._desiredWidth = value
    this._width = clamp(value, this.minWidth, this.maxWidth)
  }

  get desiredHeight() {
    return this._desiredHeight
  }

  set desiredHeight(value: number) {
    this._desiredHeight = value
    this._height = clamp(value, this.minHeight, this.maxHeight)
  }

  constructor(width: number, height: number) {
    this.desiredWidth = width
    this.desiredHeight = height
  }

  static fromSize(size: Readonly<Size>): ConstrainedSize {
    return new ConstrainedSize(size[0], size[1])
  }

  static fromRect(rect: ReadOnlyRect): ConstrainedSize {
    return new ConstrainedSize(rect[2], rect[3])
  }

  setSize(size: Readonly<Size>): void {
    this.desiredWidth = size[0]
    this.desiredHeight = size[1]
  }

  setValues(width: number, height: number): void {
    this.desiredWidth = width
    this.desiredHeight = height
  }

  toSize(): Size {
    return [this._width, this._height]
  }
}
