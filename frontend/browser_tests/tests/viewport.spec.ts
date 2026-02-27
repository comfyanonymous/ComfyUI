import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'

test.describe('Viewport', { tag: ['@screenshot', '@smoke', '@canvas'] }, () => {
  test('Fits view to nodes when saved viewport position is offscreen', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow(
      'viewport/default-viewport-saved-offscreen'
    )

    // Wait a few frames for rendering to stabilize
    for (let i = 0; i < 5; i++) {
      await comfyPage.nextFrame()
    }

    await expect(comfyPage.canvas).toHaveScreenshot(
      'viewport-fits-when-saved-offscreen.png'
    )
  })
})
