import type { Locator, Page } from '@playwright/test'

export class ManageGroupNode {
  footer: Locator
  header: Locator

  constructor(
    readonly page: Page,
    readonly root: Locator
  ) {
    this.footer = root.locator('footer')
    this.header = root.locator('header')
  }

  async setLabel(name: string, label: string) {
    const active = this.root.locator('.comfy-group-manage-node-page.active')
    const input = active.getByPlaceholder(name)
    await input.fill(label)
  }

  async save() {
    await this.footer.getByText('Save').click()
  }

  async close() {
    await this.footer.getByText('Close').click()
  }

  async getSelectedNodeType() {
    const select = this.header.locator('select').first()
    return await select.inputValue()
  }

  async selectNode(name: string) {
    const list = this.root.locator('.comfy-group-manage-list-items')
    const item = list.getByText(name)
    await item.click()
  }

  async changeTab(name: 'Inputs' | 'Widgets' | 'Outputs') {
    const header = this.root.locator('.comfy-group-manage-node header')
    const tab = header.getByText(name)
    await tab.click()
  }
}
