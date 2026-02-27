import { expect } from '@playwright/test'

import { userSelectPageFixture as test } from '../fixtures/UserSelectPage'

/**
 * Expects ComfyUI backend to be launched with `--multi-user` flag.
 */
test.describe('User Select View', { tag: '@settings' }, () => {
  test.beforeEach(async ({ userSelectPage, page }) => {
    await page.goto(userSelectPage.url)
    await page.evaluate(() => {
      localStorage.clear()
      sessionStorage.clear()
    })
  })

  test('Redirects to user select view if no user is logged in', async ({
    userSelectPage,
    page
  }) => {
    await page.goto(userSelectPage.url)
    await expect(userSelectPage.container).toBeVisible()
    expect(page.url()).toBe(userSelectPage.selectionUrl)
  })

  test('Can create new user', async ({ userSelectPage, page }) => {
    const randomUser = `test-user-${Math.random().toString(36).substring(2, 7)}`
    await page.goto(userSelectPage.url)
    await expect(page).toHaveURL(userSelectPage.selectionUrl)
    await userSelectPage.newUserInput.fill(randomUser)
    await userSelectPage.nextButton.click()
    await expect(page).toHaveURL(userSelectPage.url)
  })

  test('Can choose existing user', async ({ userSelectPage, page }) => {
    await page.goto(userSelectPage.url)
    await expect(page).toHaveURL(userSelectPage.selectionUrl)
    await userSelectPage.existingUserSelect.click()
    await page.locator('.p-select-list .p-select-option').first().click()
    await userSelectPage.nextButton.click()
    await expect(page).toHaveURL(userSelectPage.url)
  })
})
