import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'
import { TestIds } from '../fixtures/selectors'

test.beforeEach(async ({ comfyPage }) => {
  await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
})

test.describe('Graph Canvas Menu', { tag: ['@screenshot', '@canvas'] }, () => {
  test.beforeEach(async ({ comfyPage }) => {
    // Set link render mode to spline to make sure it's not affected by other tests'
    // side effects.
    await comfyPage.settings.setSetting('Comfy.LinkRenderMode', 2)
    // Enable canvas menu for all tests
    await comfyPage.settings.setSetting('Comfy.Graph.CanvasMenu', true)
  })

  test(
    'Can toggle link visibility',
    { tag: '@screenshot' },
    async ({ comfyPage }) => {
      const button = comfyPage.page.getByTestId(
        TestIds.canvas.toggleLinkVisibilityButton
      )
      await button.click()
      await comfyPage.nextFrame()
      await expect(comfyPage.canvas).toHaveScreenshot(
        'canvas-with-hidden-links.png'
      )
      const hiddenLinkRenderMode = await comfyPage.page.evaluate(() => {
        return window.LiteGraph!.HIDDEN_LINK
      })
      expect(await comfyPage.settings.getSetting('Comfy.LinkRenderMode')).toBe(
        hiddenLinkRenderMode
      )

      await button.click()
      await comfyPage.nextFrame()
      await expect(comfyPage.canvas).toHaveScreenshot(
        'canvas-with-visible-links.png'
      )
      expect(
        await comfyPage.settings.getSetting('Comfy.LinkRenderMode')
      ).not.toBe(hiddenLinkRenderMode)
    }
  )

  test('Toggle minimap button is clickable and has correct test id', async ({
    comfyPage
  }) => {
    const minimapButton = comfyPage.page.getByTestId(
      TestIds.canvas.toggleMinimapButton
    )
    await expect(minimapButton).toBeVisible()
    await expect(minimapButton).toBeEnabled()

    // Test that the button can be clicked without error
    await minimapButton.click()
    await comfyPage.nextFrame()
  })

  test('Zoom controls popup opens and closes', async ({ comfyPage }) => {
    // Find the zoom button by its percentage text content
    const zoomButton = comfyPage.page.locator('button').filter({
      hasText: '%'
    })
    await expect(zoomButton).toBeVisible()

    // Click to open zoom controls
    await zoomButton.click()
    await comfyPage.nextFrame()

    // Zoom controls modal should be visible
    const zoomModal = comfyPage.page
      .locator('div')
      .filter({
        hasText: 'Zoom To Fit'
      })
      .first()
    await expect(zoomModal).toBeVisible()

    // Click backdrop to close
    const backdrop = comfyPage.page.locator('.fixed.inset-0').first()
    await backdrop.click()
    await comfyPage.nextFrame()

    // Modal should be hidden
    await expect(zoomModal).not.toBeVisible()
  })
})
