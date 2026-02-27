import type { Page } from '@playwright/test'

import type { ComfyPage } from '../ComfyPage'
import { TestIds } from '../selectors'
import { BaseDialog } from './BaseDialog'

export class SettingDialog extends BaseDialog {
  constructor(
    page: Page,
    public readonly comfyPage: ComfyPage
  ) {
    super(page, TestIds.dialogs.settings)
  }

  async open() {
    await this.comfyPage.command.executeCommand('Comfy.ShowSettingsDialog')
    await this.waitForVisible()
  }

  /**
   * Set the value of a text setting
   * @param id - The id of the setting
   * @param value - The value to set
   */
  async setStringSetting(id: string, value: string) {
    const settingInputDiv = this.root.locator(`div[id="${id}"]`)
    await settingInputDiv.locator('input').fill(value)
  }

  /**
   * Toggle the value of a boolean setting
   * @param id - The id of the setting
   */
  async toggleBooleanSetting(id: string) {
    const settingInputDiv = this.root.locator(`div[id="${id}"]`)
    await settingInputDiv.locator('input').click()
  }

  get searchBox() {
    return this.root.getByPlaceholder(/Search/)
  }

  get categories() {
    return this.root.locator('nav').getByRole('button')
  }

  category(name: string) {
    return this.root.locator('nav').getByRole('button', { name })
  }

  get contentArea() {
    return this.root.getByRole('main')
  }

  async goToAboutPanel() {
    const aboutButton = this.root.locator('nav').getByRole('button', {
      name: 'About'
    })
    await aboutButton.click()
    await this.page.waitForSelector('.about-container')
  }
}
