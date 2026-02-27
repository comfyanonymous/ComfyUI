import type { Locator, Page } from '@playwright/test'

export class ContextMenu {
  constructor(public readonly page: Page) {}

  get primeVueMenu() {
    return this.page.locator('.p-contextmenu, .p-menu')
  }

  get litegraphMenu() {
    return this.page.locator('.litemenu')
  }

  get menuItems() {
    return this.page.locator('.p-menuitem, .litemenu-entry')
  }

  async clickMenuItem(name: string): Promise<void> {
    await this.page.getByRole('menuitem', { name }).click()
  }

  async clickLitegraphMenuItem(name: string): Promise<void> {
    await this.page.locator(`.litemenu-entry:has-text("${name}")`).click()
  }

  async isVisible(): Promise<boolean> {
    const primeVueVisible = await this.primeVueMenu
      .isVisible()
      .catch(() => false)
    const litegraphVisible = await this.litegraphMenu
      .isVisible()
      .catch(() => false)
    return primeVueVisible || litegraphVisible
  }

  async waitForHidden(): Promise<void> {
    const waitIfExists = async (locator: Locator, menuName: string) => {
      const count = await locator.count()
      if (count > 0) {
        await locator.waitFor({ state: 'hidden' }).catch((error: Error) => {
          console.warn(
            `[waitForHidden] ${menuName} waitFor failed:`,
            error.message
          )
        })
      }
    }

    await Promise.all([
      waitIfExists(this.primeVueMenu, 'primeVueMenu'),
      waitIfExists(this.litegraphMenu, 'litegraphMenu')
    ])
  }
}
