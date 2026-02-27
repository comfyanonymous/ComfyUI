import type { Mouse } from '@playwright/test'

import type { ComfyPage } from './ComfyPage'
import type { Position } from './types'

/**
 * Used for drag and drop ops
 * @see
 * - {@link Mouse.down}
 * - {@link Mouse.move}
 * - {@link Mouse.up}
 */
interface DragOptions {
  button?: 'left' | 'right' | 'middle'
  clickCount?: number
  steps?: number
}

/**
 * Wraps mouse drag and drop to work with a canvas-based app.
 *
 * Requires the next frame animated before and after all steps, giving the
 * canvas time to render the changes before screenshots are taken.
 */
export class ComfyMouse implements Omit<Mouse, 'move'> {
  static defaultSteps = 5
  static defaultOptions: DragOptions = { steps: ComfyMouse.defaultSteps }

  constructor(readonly comfyPage: ComfyPage) {}

  /** The normal Playwright {@link Mouse} property from {@link ComfyPage.page}. */
  get mouse() {
    return this.comfyPage.page.mouse
  }

  async nextFrame() {
    await this.comfyPage.nextFrame()
  }

  /** Drags from current location to a new location and hovers there (no pointerup event) */
  async drag(to: Position, options = ComfyMouse.defaultOptions) {
    const { steps, ...downOptions } = options

    await this.mouse.down(downOptions)
    await this.nextFrame()

    await this.move(to, { steps })
    await this.nextFrame()
  }

  async drop(options = ComfyMouse.defaultOptions) {
    await this.mouse.up(options)
    await this.nextFrame()
  }

  async dragAndDrop(
    from: Position,
    to: Position,
    options = ComfyMouse.defaultOptions
  ) {
    const { steps } = options

    await this.nextFrame()

    await this.move(from, { steps })
    await this.drag(to, options)
    await this.drop(options)
  }

  /** @see {@link Mouse.move} */
  async move(to: Position, options = ComfyMouse.defaultOptions) {
    await this.mouse.move(to.x, to.y, options)
    await this.nextFrame()
  }

  //#region Pass-through
  async click(...args: Parameters<Mouse['click']>) {
    return await this.mouse.click(...args)
  }

  async dblclick(...args: Parameters<Mouse['dblclick']>) {
    return await this.mouse.dblclick(...args)
  }

  async down(...args: Parameters<Mouse['down']>) {
    return await this.mouse.down(...args)
  }

  async up(...args: Parameters<Mouse['up']>) {
    return await this.mouse.up(...args)
  }

  async wheel(...args: Parameters<Mouse['wheel']>) {
    return await this.mouse.wheel(...args)
  }
  //#endregion Pass-through
}
