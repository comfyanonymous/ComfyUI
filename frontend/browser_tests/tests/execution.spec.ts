import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'

test.beforeEach(async ({ comfyPage }) => {
  await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
})

test.describe('Execution', { tag: ['@smoke', '@workflow'] }, () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')
    await comfyPage.setup()
  })

  test(
    'Report error on unconnected slot',
    { tag: '@screenshot' },
    async ({ comfyPage }) => {
      await comfyPage.canvasOps.disconnectEdge()
      await comfyPage.page.keyboard.press('Escape')

      await comfyPage.command.executeCommand('Comfy.QueuePrompt')
      await expect(
        comfyPage.page.locator('[data-testid="error-overlay"]')
      ).toBeVisible()
      await comfyPage.page
        .locator('[data-testid="error-overlay"]')
        .getByRole('button', { name: 'Dismiss' })
        .click()
      await comfyPage.page
        .locator('[data-testid="error-overlay"]')
        .waitFor({ state: 'hidden' })
      await expect(comfyPage.canvas).toHaveScreenshot(
        'execution-error-unconnected-slot.png'
      )
    }
  )
})

test.describe(
  'Execute to selected output nodes',
  { tag: ['@smoke', '@workflow'] },
  () => {
    test('Execute to selected output nodes', async ({ comfyPage }) => {
      await comfyPage.workflow.loadWorkflow('execution/partial_execution')
      const input = await comfyPage.nodeOps.getNodeRefById(3)
      const output1 = await comfyPage.nodeOps.getNodeRefById(1)
      const output2 = await comfyPage.nodeOps.getNodeRefById(4)
      expect(await (await input.getWidget(0)).getValue()).toBe('foo')
      expect(await (await output1.getWidget(0)).getValue()).toBe('')
      expect(await (await output2.getWidget(0)).getValue()).toBe('')

      await output1.click('title')

      await comfyPage.command.executeCommand('Comfy.QueueSelectedOutputNodes')
      await expect(async () => {
        expect(await (await input.getWidget(0)).getValue()).toBe('foo')
        expect(await (await output1.getWidget(0)).getValue()).toBe('foo')
        expect(await (await output2.getWidget(0)).getValue()).toBe('')
      }).toPass({ timeout: 2_000 })
    })
  }
)
