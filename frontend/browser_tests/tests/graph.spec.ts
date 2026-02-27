import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'

test.beforeEach(async ({ comfyPage }) => {
  await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
})

test.describe('Graph', { tag: ['@smoke', '@canvas'] }, () => {
  // Should be able to fix link input slot index after swap the input order
  // Ref: https://github.com/Comfy-Org/ComfyUI_frontend/issues/3348
  test('Fix link input slots', async ({ comfyPage }) => {
    await comfyPage.workflow.loadWorkflow('inputs/input_order_swap')
    expect(
      await comfyPage.page.evaluate(() => {
        return window.app!.graph!.links.get(1)?.target_slot
      })
    ).toBe(1)
  })

  test('Validate workflow links', async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.Validation.Workflows', true)
    await comfyPage.workflow.loadWorkflow('links/bad_link')
    await expect.poll(() => comfyPage.toast.getVisibleToastCount()).toBe(2)
  })
})
