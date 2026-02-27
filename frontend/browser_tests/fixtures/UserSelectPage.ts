import type { Page } from '@playwright/test'
import { test as base } from '@playwright/test'

export class UserSelectPage {
  constructor(
    public readonly url: string,
    public readonly page: Page
  ) {}

  get selectionUrl() {
    return this.url + '/user-select'
  }

  get container() {
    return this.page.locator('#comfy-user-selection')
  }

  get newUserInput() {
    return this.container.locator('#new-user-input')
  }

  get existingUserSelect() {
    return this.container.locator('#existing-user-select')
  }

  get nextButton() {
    return this.container.getByText('Next')
  }
}

export const userSelectPageFixture = base.extend<{
  userSelectPage: UserSelectPage
}>({
  userSelectPage: async ({ page }, use) => {
    const userSelectPage = new UserSelectPage(
      process.env.PLAYWRIGHT_TEST_URL || 'http://localhost:8188',
      page
    )
    await use(userSelectPage)
  }
})
