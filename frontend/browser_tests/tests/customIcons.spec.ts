import { expect } from '@playwright/test'
import type { Locator } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'

async function verifyCustomIconSvg(iconElement: Locator) {
  const svgVariable = await iconElement.evaluate((element) => {
    const styles = getComputedStyle(element)
    return styles.getPropertyValue('--svg')
  })

  expect(svgVariable).toBeTruthy()
  const dataUrlMatch = svgVariable.match(
    /url\("data:image\/svg\+xml,([^"]+)"\)/
  )
  expect(dataUrlMatch).toBeTruthy()

  const encodedSvg = dataUrlMatch![1]
  const decodedSvg = decodeURIComponent(encodedSvg)

  // Check for SVG header to confirm it's a valid SVG
  expect(decodedSvg).toContain("<svg xmlns='http://www.w3.org/2000/svg'")
}

test.describe('Custom Icons', { tag: '@settings' }, () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')
  })

  test('sidebar tab icons use custom SVGs', async ({ comfyPage }) => {
    // Find the icon in the sidebar
    const icon = comfyPage.page.locator(
      '.icon-\\[comfy--ai-model\\].side-bar-button-icon'
    )
    await expect(icon).toBeVisible()

    // Verify the custom SVG content
    await verifyCustomIconSvg(icon)
  })

  test('Browse Templates menu item uses custom template icon', async ({
    comfyPage
  }) => {
    // Open the topbar menu
    await comfyPage.menu.topbar.openTopbarMenu()
    const menuItem = comfyPage.menu.topbar.getMenuItem('Browse Templates')

    // Find the icon as a previous sibling of the menu item label
    const templateIcon = menuItem
      .locator('..')
      .locator('.icon-\\[comfy--template\\]')
    await expect(templateIcon).toBeVisible()

    // Verify the custom SVG content
    await verifyCustomIconSvg(templateIcon)
  })
})
