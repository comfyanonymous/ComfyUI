import type { LGraphCanvas } from '@/lib/litegraph/src/LGraphCanvas'

/**
 * A class that can be added to the render cycle to show pointer / keyboard status symbols.
 *
 * Used to create videos of feature changes.
 *
 * Example usage with ComfyUI_frontend, via console / devtools:
 *
 * ```ts
 * const inputIndicators = new InputIndicators(canvas)
 * // Dispose:
 * inputIndicators.dispose()
 * ```
 */
export class InputIndicators implements Disposable {
  // #region config
  radius = 8
  startAngle = 0
  endAngle = Math.PI * 2

  inactiveColour = '#ffffff10'
  colour1 = '#ff5f00'
  colour2 = '#00ff7c'
  colour3 = '#dea7ff'
  fontString = 'bold 12px Inter, sans-serif'
  // #endregion

  // #region state
  enabled: boolean = true

  shiftDown: boolean = false
  undoDown: boolean = false
  redoDown: boolean = false
  ctrlDown: boolean = false
  altDown: boolean = false
  mouse0Down: boolean = false
  mouse1Down: boolean = false
  mouse2Down: boolean = false

  x: number = 0
  y: number = 0
  // #endregion

  controller?: AbortController

  constructor(public canvas: LGraphCanvas) {
    this.controller = new AbortController()
    const { signal } = this.controller

    const element = canvas.canvas
    const options = { capture: true, signal } satisfies AddEventListenerOptions

    element.addEventListener('pointerdown', this._onPointerDownOrMove, options)
    element.addEventListener('pointermove', this._onPointerDownOrMove, options)
    element.addEventListener('pointerup', this._onPointerUp, options)
    element.addEventListener('keydown', this._onKeyDownOrUp, options)
    document.addEventListener('keyup', this._onKeyDownOrUp, options)

    const origDrawFrontCanvas = canvas.drawFrontCanvas.bind(canvas)
    signal.addEventListener('abort', () => {
      canvas.drawFrontCanvas = origDrawFrontCanvas
    })

    canvas.drawFrontCanvas = () => {
      origDrawFrontCanvas()
      this.draw()
    }
  }

  private _onPointerDownOrMove = this.onPointerDownOrMove.bind(this)
  onPointerDownOrMove(e: MouseEvent): void {
    this.mouse0Down = (e.buttons & 1) === 1
    this.mouse1Down = (e.buttons & 4) === 4
    this.mouse2Down = (e.buttons & 2) === 2

    this.x = e.clientX
    this.y = e.clientY

    this.canvas.setDirty(true)
  }

  private _onPointerUp = this.onPointerUp.bind(this)
  onPointerUp(): void {
    this.mouse0Down = false
    this.mouse1Down = false
    this.mouse2Down = false
  }

  private _onKeyDownOrUp = this.onKeyDownOrUp.bind(this)
  onKeyDownOrUp(e: KeyboardEvent): void {
    this.ctrlDown = e.ctrlKey
    this.altDown = e.altKey
    this.shiftDown = e.shiftKey
    this.undoDown = e.ctrlKey && e.code === 'KeyZ' && e.type === 'keydown'
    this.redoDown = e.ctrlKey && e.code === 'KeyY' && e.type === 'keydown'
  }

  draw() {
    const {
      canvas: { ctx },
      radius,
      startAngle,
      endAngle,
      x,
      y,
      inactiveColour,
      colour1,
      colour2,
      colour3,
      fontString
    } = this

    const { fillStyle, font } = ctx

    const mouseDotX = x
    const mouseDotY = y - 80

    const textX = mouseDotX
    const textY = mouseDotY - 15
    ctx.font = fontString

    textMarker(
      textX + 0,
      textY,
      'Shift',
      this.shiftDown ? colour1 : inactiveColour
    )
    textMarker(
      textX + 45,
      textY + 20,
      'Alt',
      this.altDown ? colour2 : inactiveColour
    )
    textMarker(
      textX + 30,
      textY,
      'Control',
      this.ctrlDown ? colour3 : inactiveColour
    )
    textMarker(textX - 30, textY, '↩️', this.undoDown ? '#000' : 'transparent')
    textMarker(textX + 45, textY, '↪️', this.redoDown ? '#000' : 'transparent')

    ctx.beginPath()
    drawDot(mouseDotX, mouseDotY)
    drawDot(mouseDotX + 15, mouseDotY)
    drawDot(mouseDotX + 30, mouseDotY)
    ctx.fillStyle = inactiveColour
    ctx.fill()

    const leftButtonColour = this.mouse0Down ? colour1 : inactiveColour
    const middleButtonColour = this.mouse1Down ? colour2 : inactiveColour
    const rightButtonColour = this.mouse2Down ? colour3 : inactiveColour
    if (this.mouse0Down) mouseMarker(mouseDotX, mouseDotY, leftButtonColour)
    if (this.mouse1Down)
      mouseMarker(mouseDotX + 15, mouseDotY, middleButtonColour)
    if (this.mouse2Down)
      mouseMarker(mouseDotX + 30, mouseDotY, rightButtonColour)

    ctx.fillStyle = fillStyle
    ctx.font = font

    function textMarker(x: number, y: number, text: string, colour: string) {
      ctx.fillStyle = colour
      ctx.fillText(text, x, y)
    }

    function mouseMarker(x: number, y: number, colour: string) {
      ctx.beginPath()
      ctx.fillStyle = colour
      drawDot(x, y)
      ctx.fill()
    }

    function drawDot(x: number, y: number) {
      ctx.arc(x, y, radius, startAngle, endAngle)
    }
  }

  dispose() {
    this.controller?.abort()
    this.controller = undefined
  }

  [Symbol.dispose](): void {
    this.dispose()
  }
}
