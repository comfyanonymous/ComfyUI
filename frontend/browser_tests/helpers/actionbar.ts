import type { Locator, Page } from '@playwright/test'

import type { AutoQueueMode } from '../../src/stores/queueStore'
import { TestIds } from '../fixtures/selectors'
import type { WorkspaceStore } from '../types/globals'

export class ComfyActionbar {
  public readonly root: Locator
  public readonly queueButton: ComfyQueueButton
  public readonly propertiesButton: Locator

  constructor(public readonly page: Page) {
    this.root = page.locator('.actionbar-container')
    this.queueButton = new ComfyQueueButton(this)
    this.propertiesButton = this.root.getByLabel('Toggle properties panel')
  }

  async isDocked() {
    const className = await this.root
      .locator('.actionbar')
      .getAttribute('class')
    return className?.includes('static') ?? false
  }
}

class ComfyQueueButton {
  public readonly root: Locator
  public readonly primaryButton: Locator
  public readonly dropdownButton: Locator
  constructor(public readonly actionbar: ComfyActionbar) {
    this.root = actionbar.root.getByTestId(TestIds.topbar.queueButton)
    this.primaryButton = this.root.locator('.p-splitbutton-button')
    this.dropdownButton = this.root.locator('.p-splitbutton-dropdown')
  }

  public async toggleOptions() {
    await this.dropdownButton.click()
    return new ComfyQueueButtonOptions(this.actionbar.page)
  }
}

class ComfyQueueButtonOptions {
  constructor(public readonly page: Page) {}

  public async setMode(mode: AutoQueueMode) {
    await this.page.evaluate((mode) => {
      ;(window.app!.extensionManager as WorkspaceStore).queueSettings.mode =
        mode
    }, mode)
  }

  public async getMode() {
    return await this.page.evaluate(() => {
      return (window.app!.extensionManager as WorkspaceStore).queueSettings.mode
    })
  }
}
