import { expect } from '@playwright/test'
import type { Locator, Page } from '@playwright/test'

export class ToastHelper {
  constructor(private readonly page: Page) {}

  get visibleToasts(): Locator {
    return this.page.locator('.p-toast-message:visible')
  }

  async getToastErrorCount(): Promise<number> {
    return await this.page
      .locator('.p-toast-message.p-toast-message-error')
      .count()
  }

  async getVisibleToastCount(): Promise<number> {
    return await this.visibleToasts.count()
  }

  async closeToasts(requireCount = 0): Promise<void> {
    if (requireCount) {
      await this.visibleToasts
        .nth(requireCount - 1)
        .waitFor({ state: 'visible' })
    }

    // Clear all toasts
    const toastCloseButtons = await this.page
      .locator('.p-toast-close-button')
      .all()
    for (const button of toastCloseButtons) {
      await button.click()
    }

    // Assert all toasts are closed
    await expect(this.visibleToasts).toHaveCount(0, { timeout: 1000 })
  }
}
