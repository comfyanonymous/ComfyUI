import type { Locator, Page } from '@playwright/test'

export class BaseDialog {
  readonly root: Locator
  readonly closeButton: Locator

  constructor(
    public readonly page: Page,
    testId?: string
  ) {
    this.root = testId ? page.getByTestId(testId) : page.locator('.p-dialog')
    this.closeButton = this.root.getByRole('button', { name: 'Close' })
  }

  async isVisible(): Promise<boolean> {
    return this.root.isVisible()
  }

  async waitForVisible(): Promise<void> {
    await this.root.waitFor({ state: 'visible' })
  }

  async waitForHidden(): Promise<void> {
    await this.root.waitFor({ state: 'hidden' })
  }

  async close(): Promise<void> {
    await this.closeButton.click({ force: true })
    await this.waitForHidden()
  }
}
