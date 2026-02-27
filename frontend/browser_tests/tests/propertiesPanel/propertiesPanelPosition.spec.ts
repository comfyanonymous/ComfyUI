import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../../fixtures/ComfyPage'
import { TestIds } from '../../fixtures/selectors'

test.describe('Properties panel position', () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.NodeLibrary.NewDesign', false)
    // Open a sidebar tab to ensure sidebar is visible
    await comfyPage.menu.nodeLibraryTab.open()
    await comfyPage.actionbar.propertiesButton.click()
  })

  test('positions on the right when sidebar is on the left', async ({
    comfyPage
  }) => {
    await comfyPage.settings.setSetting('Comfy.Sidebar.Location', 'left')
    await comfyPage.nextFrame()

    const propertiesPanel = comfyPage.page.getByTestId(
      TestIds.propertiesPanel.root
    )
    const sidebar = comfyPage.page.locator('.side-bar-panel').first()

    await expect(propertiesPanel).toBeVisible()
    await expect(sidebar).toBeVisible()

    const propsBoundingBox = await propertiesPanel.boundingBox()
    const sidebarBoundingBox = await sidebar.boundingBox()

    expect(propsBoundingBox).not.toBeNull()
    expect(sidebarBoundingBox).not.toBeNull()

    // Properties panel should be to the right of the sidebar
    expect(propsBoundingBox!.x).toBeGreaterThan(
      sidebarBoundingBox!.x + sidebarBoundingBox!.width
    )
  })

  test('positions on the left when sidebar is on the right', async ({
    comfyPage
  }) => {
    await comfyPage.settings.setSetting('Comfy.Sidebar.Location', 'right')
    await comfyPage.nextFrame()

    const propertiesPanel = comfyPage.page.getByTestId(
      TestIds.propertiesPanel.root
    )
    const sidebar = comfyPage.page.locator('.side-bar-panel').first()

    await expect(propertiesPanel).toBeVisible()
    await expect(sidebar).toBeVisible()

    const propsBoundingBox = await propertiesPanel.boundingBox()
    const sidebarBoundingBox = await sidebar.boundingBox()

    expect(propsBoundingBox).not.toBeNull()
    expect(sidebarBoundingBox).not.toBeNull()

    // Properties panel should be to the left of the sidebar
    expect(propsBoundingBox!.x + propsBoundingBox!.width).toBeLessThan(
      sidebarBoundingBox!.x
    )
  })

  test('close button icon updates based on sidebar location', async ({
    comfyPage
  }) => {
    const propertiesPanel = comfyPage.page.getByTestId(
      TestIds.propertiesPanel.root
    )

    // When sidebar is on the left, panel is on the right
    await comfyPage.settings.setSetting('Comfy.Sidebar.Location', 'left')
    await comfyPage.nextFrame()

    await expect(propertiesPanel).toBeVisible()
    const closeButtonLeft = propertiesPanel
      .locator('button[aria-pressed]')
      .locator('i')
    await expect(closeButtonLeft).toBeVisible()
    await expect(closeButtonLeft).toHaveClass(/lucide--panel-right/)

    // When sidebar is on the right, panel is on the left
    await comfyPage.settings.setSetting('Comfy.Sidebar.Location', 'right')
    await comfyPage.nextFrame()

    const closeButtonRight = propertiesPanel
      .locator('button[aria-pressed]')
      .locator('i')
    await expect(closeButtonRight).toBeVisible()
    await expect(closeButtonRight).toHaveClass(/lucide--panel-left/)
  })
})
