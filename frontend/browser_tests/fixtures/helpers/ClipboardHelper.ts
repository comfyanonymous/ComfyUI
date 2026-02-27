import type { Locator } from '@playwright/test'

import type { KeyboardHelper } from './KeyboardHelper'

export class ClipboardHelper {
  constructor(private readonly keyboard: KeyboardHelper) {}

  async copy(locator?: Locator | null): Promise<void> {
    await this.keyboard.ctrlSend('KeyC', locator ?? null)
  }

  async paste(locator?: Locator | null): Promise<void> {
    await this.keyboard.ctrlSend('KeyV', locator ?? null)
  }
}
