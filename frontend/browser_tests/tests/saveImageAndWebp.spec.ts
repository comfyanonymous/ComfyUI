import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'

test.describe(
  'Save Image and WEBM preview',
  { tag: ['@screenshot', '@widget'] },
  () => {
    test.beforeEach(async ({ comfyPage }) => {
      await comfyPage.settings.setSetting('Comfy.VueNodes.Enabled', true)
      await comfyPage.vueNodes.waitForNodes()
    })

    test('Can preview both SaveImage and SaveWEBM outputs', async ({
      comfyPage
    }) => {
      await comfyPage.workflow.loadWorkflow(
        'widgets/save_image_and_animated_webp'
      )
      await comfyPage.vueNodes.waitForNodes()

      await comfyPage.runButton.click()

      const saveImageNode = comfyPage.vueNodes.getNodeByTitle('Save Image')
      const saveWebmNode = comfyPage.vueNodes.getNodeByTitle('SaveWEBM')

      // Wait for SaveImage to render an img inside .image-preview
      await expect(saveImageNode.locator('.image-preview img')).toBeVisible({
        timeout: 30000
      })

      // Wait for SaveWEBM to render a video inside .video-preview
      await expect(saveWebmNode.locator('.video-preview video')).toBeVisible({
        timeout: 30000
      })

      await expect(comfyPage.page).toHaveScreenshot(
        'save-image-and-webm-preview.png'
      )
    })
  }
)
