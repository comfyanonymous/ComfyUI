import { expect } from '@playwright/test'

import type { ComfyPage } from '../../../../fixtures/ComfyPage'
import { comfyPageFixture as test } from '../../../../fixtures/ComfyPage'
import {
  getPromotedWidgetNames,
  getPromotedWidgetCountByName
} from '../../../../helpers/promotedWidgets'

test.describe('Vue Nodes Image Preview', () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.VueNodes.Enabled', true)
  })

  async function loadImageOnNode(comfyPage: ComfyPage) {
    await comfyPage.workflow.loadWorkflow('widgets/load_image_widget')
    await comfyPage.vueNodes.waitForNodes()

    const loadImageNode = (
      await comfyPage.nodeOps.getNodeRefsByType('LoadImage')
    )[0]
    const { x, y } = await loadImageNode.getPosition()

    await comfyPage.dragDrop.dragAndDropFile('image64x64.webp', {
      dropPosition: { x, y }
    })

    const imagePreview = comfyPage.page.locator('.image-preview')
    await expect(imagePreview).toBeVisible()
    await expect(imagePreview.locator('img')).toBeVisible()
    await expect(imagePreview).toContainText('x')

    return {
      imagePreview,
      nodeId: String(loadImageNode.id)
    }
  }

  // TODO(#8143): Re-enable after image preview sync is working in CI
  test.fixme('opens mask editor from image preview button', async ({
    comfyPage
  }) => {
    const { imagePreview } = await loadImageOnNode(comfyPage)

    await imagePreview.locator('[role="img"]').focus()
    await comfyPage.page.getByLabel('Edit or mask image').click()

    await expect(comfyPage.page.locator('.mask-editor-dialog')).toBeVisible()
  })

  // TODO(#8143): Re-enable after image preview sync is working in CI
  test.fixme('shows image context menu options', async ({ comfyPage }) => {
    const { nodeId } = await loadImageOnNode(comfyPage)

    const nodeHeader = comfyPage.vueNodes
      .getNodeLocator(nodeId)
      .locator('.lg-node-header')
    await nodeHeader.click()
    await nodeHeader.click({ button: 'right' })

    const contextMenu = comfyPage.page.locator('.p-contextmenu')
    await expect(contextMenu).toBeVisible()
    await expect(contextMenu.getByText('Open Image')).toBeVisible()
    await expect(contextMenu.getByText('Copy Image')).toBeVisible()
    await expect(contextMenu.getByText('Save Image')).toBeVisible()
    await expect(contextMenu.getByText('Open in Mask Editor')).toBeVisible()
  })

  test(
    'renders promoted image previews for each subgraph node',
    { tag: '@screenshot' },
    async ({ comfyPage }) => {
      await comfyPage.workflow.loadWorkflow(
        'subgraphs/subgraph-with-multiple-promoted-previews'
      )
      await comfyPage.vueNodes.waitForNodes()

      const firstSubgraphNode = comfyPage.vueNodes.getNodeLocator('7')
      const secondSubgraphNode = comfyPage.vueNodes.getNodeLocator('8')

      await expect(firstSubgraphNode).toBeVisible()
      await expect(secondSubgraphNode).toBeVisible()

      const firstPromotedWidgets = await getPromotedWidgetNames(comfyPage, '7')
      const secondPromotedWidgets = await getPromotedWidgetNames(comfyPage, '8')
      expect(firstPromotedWidgets).toEqual([
        '$$canvas-image-preview',
        '$$canvas-image-preview'
      ])
      expect(secondPromotedWidgets).toEqual(['$$canvas-image-preview'])

      expect(
        await getPromotedWidgetCountByName(
          comfyPage,
          '7',
          '$$canvas-image-preview'
        )
      ).toBe(2)
      expect(
        await getPromotedWidgetCountByName(
          comfyPage,
          '8',
          '$$canvas-image-preview'
        )
      ).toBe(1)

      await expect(
        firstSubgraphNode.locator('.lg-node-widgets')
      ).not.toContainText('$$canvas-image-preview')
      await expect(
        secondSubgraphNode.locator('.lg-node-widgets')
      ).not.toContainText('$$canvas-image-preview')

      await comfyPage.command.executeCommand('Comfy.Canvas.FitView')
      await comfyPage.command.executeCommand('Comfy.QueuePrompt')

      const firstPreviewImages = firstSubgraphNode.locator('.image-preview img')
      const secondPreviewImages =
        secondSubgraphNode.locator('.image-preview img')

      await expect(firstPreviewImages).toHaveCount(2, { timeout: 30_000 })
      await expect(secondPreviewImages).toHaveCount(1, { timeout: 30_000 })

      await expect(firstPreviewImages.first()).toBeVisible({ timeout: 30_000 })
      await expect(firstPreviewImages.nth(1)).toBeVisible({ timeout: 30_000 })
      await expect(secondPreviewImages.first()).toBeVisible({ timeout: 30_000 })

      await expect(comfyPage.canvas).toHaveScreenshot(
        'vue-node-multiple-promoted-previews.png'
      )
    }
  )
})
