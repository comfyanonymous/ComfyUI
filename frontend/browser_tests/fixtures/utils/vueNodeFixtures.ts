import type { Locator } from '@playwright/test'

/** DOM-centric helper for a single Vue-rendered node on the canvas. */
export class VueNodeFixture {
  constructor(private readonly locator: Locator) {}

  get header(): Locator {
    return this.locator.locator('[data-testid^="node-header-"]')
  }

  get title(): Locator {
    return this.locator.locator('[data-testid="node-title"]')
  }

  get titleInput(): Locator {
    return this.locator.locator('[data-testid="node-title-input"]')
  }

  get body(): Locator {
    return this.locator.locator('[data-testid^="node-body-"]')
  }

  get collapseButton(): Locator {
    return this.locator.locator('[data-testid="node-collapse-button"]')
  }

  get collapseIcon(): Locator {
    return this.collapseButton.locator('i')
  }

  get root(): Locator {
    return this.locator
  }

  async getTitle(): Promise<string> {
    return (await this.title.textContent()) ?? ''
  }

  async setTitle(value: string): Promise<void> {
    await this.header.dblclick()
    const input = this.titleInput
    await input.waitFor({ state: 'visible' })
    await input.fill(value)
    await input.press('Enter')
  }

  async cancelTitleEdit(): Promise<void> {
    await this.header.dblclick()
    const input = this.titleInput
    await input.waitFor({ state: 'visible' })
    await input.press('Escape')
  }

  async toggleCollapse(): Promise<void> {
    await this.collapseButton.click()
  }

  async getCollapseIconClass(): Promise<string> {
    return (await this.collapseIcon.getAttribute('class')) ?? ''
  }

  boundingBox(): ReturnType<Locator['boundingBox']> {
    return this.locator.boundingBox()
  }
}
