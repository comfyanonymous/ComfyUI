import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'

test.describe('Confirm dialog text wrapping', { tag: ['@mobile'] }, () => {
  test('@mobile confirm dialog buttons are visible with long unbreakable text', async ({
    comfyPage
  }) => {
    const longFilename = 'workflow_checkpoint_' + 'a'.repeat(200) + '.json'

    await comfyPage.page.evaluate((msg) => {
      window
        .app!.extensionManager.dialog.confirm({
          title: 'Confirm',
          type: 'default',
          message: msg
        })
        .catch(() => {})
    }, longFilename)

    const dialog = comfyPage.page.getByRole('dialog')
    await expect(dialog).toBeVisible()

    const confirmButton = dialog.getByRole('button', { name: 'Confirm' })
    await expect(confirmButton).toBeVisible()
    await expect(confirmButton).toBeInViewport()

    const cancelButton = dialog.getByRole('button', { name: 'Cancel' })
    await expect(cancelButton).toBeVisible()
    await expect(cancelButton).toBeInViewport()
  })
})
