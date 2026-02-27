import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'
import type { ComfyPage } from '../fixtures/ComfyPage'

test.describe('Workflow Tab Thumbnails', { tag: '@workflow' }, () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')
    await comfyPage.settings.setSetting(
      'Comfy.Workflow.WorkflowTabsPosition',
      'Topbar'
    )
    await comfyPage.setup()
  })

  async function getTab(comfyPage: ComfyPage, index: number) {
    const tab = comfyPage.page
      .locator(`.workflow-tabs .p-togglebutton`)
      .nth(index)
    return tab
  }

  async function getTabPopover(
    comfyPage: ComfyPage,
    index: number,
    name?: string
  ) {
    const tab = await getTab(comfyPage, index)
    await tab.hover()

    const popover = comfyPage.page.locator('.workflow-popover-fade')
    await expect(popover).toHaveCount(1)
    await expect(popover).toBeVisible({ timeout: 500 })
    if (name) {
      await expect(popover).toContainText(name)
    }
    return popover
  }

  async function getTabThumbnailImage(
    comfyPage: ComfyPage,
    index: number,
    name?: string
  ) {
    const popover = await getTabPopover(comfyPage, index, name)
    const thumbnailImg = popover.locator('.workflow-preview-thumbnail img')
    return thumbnailImg
  }

  async function getNodeThumbnailBase64(comfyPage: ComfyPage, index: number) {
    const thumbnailImg = await getTabThumbnailImage(comfyPage, index)
    const src = (await thumbnailImg.getAttribute('src'))!

    // Convert blob to base64, need to execute a script to get the base64
    const base64 = await comfyPage.page.evaluate(async (src: string) => {
      const blob = await fetch(src).then((res) => res.blob())
      return new Promise((resolve, reject) => {
        const reader = new FileReader()
        reader.onloadend = () => resolve(reader.result)
        reader.onerror = reject
        reader.readAsDataURL(blob)
      })
    }, src)
    return base64
  }

  test('Should show thumbnail when hovering over a non-active tab', async ({
    comfyPage
  }) => {
    await comfyPage.menu.topbar.triggerTopbarCommand(['New'])
    const thumbnailImg = await getTabThumbnailImage(
      comfyPage,
      0,
      'Unsaved Workflow'
    )
    await expect(thumbnailImg).toBeVisible()
  })

  test('Should not show thumbnail for active tab', async ({ comfyPage }) => {
    await comfyPage.menu.topbar.triggerTopbarCommand(['New'])
    const thumbnailImg = await getTabThumbnailImage(
      comfyPage,
      1,
      'Unsaved Workflow (2)'
    )
    await expect(thumbnailImg).not.toBeVisible()
  })

  async function addNode(comfyPage: ComfyPage, category: string, node: string) {
    const canvasArea = await comfyPage.canvas.boundingBox()

    await comfyPage.page.mouse.move(
      canvasArea!.x + canvasArea!.width - 100,
      100
    )
    await expect(comfyPage.page.locator('.workflow-popover-fade')).toBeHidden()

    await comfyPage.canvasOps.rightClick(200, 200)
    await comfyPage.page.getByText('Add Node').click()
    await comfyPage.nextFrame()
    await comfyPage.page.getByText(category).click()
    await comfyPage.nextFrame()
    await comfyPage.page.getByText(node).click()
    await comfyPage.nextFrame()
  }

  test('Thumbnail should update when switching tabs', async ({ comfyPage }) => {
    // Wait for initial workflow to load
    await comfyPage.nextFrame()

    // Create a new workflow (tab 1) which will be empty
    await comfyPage.menu.topbar.triggerTopbarCommand(['New'])
    await comfyPage.nextFrame()

    // Now we have two tabs: tab 0 (default workflow with nodes) and tab 1 (empty)
    // Tab 1 is currently active, so we can only get thumbnail for tab 0

    // Step 1: Different tabs should show different previews
    const tab0ThumbnailWithNodes = await getNodeThumbnailBase64(comfyPage, 0)

    // Add a node to tab 1 (current active tab)
    await addNode(comfyPage, 'loaders', 'Load Checkpoint')
    await comfyPage.nextFrame()

    // Switch to tab 0 so we can get tab 1's thumbnail
    await (await getTab(comfyPage, 0)).click()
    await comfyPage.nextFrame()

    const tab1ThumbnailWithNode = await getNodeThumbnailBase64(comfyPage, 1)

    // The thumbnails should be different
    expect(tab0ThumbnailWithNodes).not.toBe(tab1ThumbnailWithNode)

    // Step 2: Switching without changes shouldn't update thumbnail
    const tab1ThumbnailBefore = await getNodeThumbnailBase64(comfyPage, 1)

    // Switch to tab 1 and back to tab 0 without making changes
    await (await getTab(comfyPage, 1)).click()
    await comfyPage.nextFrame()
    await (await getTab(comfyPage, 0)).click()
    await comfyPage.nextFrame()

    const tab1ThumbnailAfter = await getNodeThumbnailBase64(comfyPage, 1)
    expect(tab1ThumbnailBefore).toBe(tab1ThumbnailAfter)

    // Step 3: Adding another node should cause thumbnail to change
    // We're on tab 0, add a node
    await addNode(comfyPage, 'loaders', 'Load VAE')
    await comfyPage.nextFrame()

    // Switch to tab 1 and back to update tab 0's thumbnail
    await (await getTab(comfyPage, 1)).click()

    const tab0ThumbnailAfterNewNode = await getNodeThumbnailBase64(comfyPage, 0)

    // The thumbnail should have changed after adding a node
    expect(tab0ThumbnailWithNodes).not.toBe(tab0ThumbnailAfterNewNode)
  })
})
