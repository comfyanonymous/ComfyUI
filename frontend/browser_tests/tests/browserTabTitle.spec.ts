import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'
import type { WorkspaceStore } from '../types/globals'

test.describe('Browser tab title', { tag: '@smoke' }, () => {
  test.describe('Beta Menu', () => {
    test.beforeEach(async ({ comfyPage }) => {
      await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')
    })

    test('Can display workflow name', async ({ comfyPage }) => {
      const workflowName = await comfyPage.page.evaluate(async () => {
        return (window.app!.extensionManager as WorkspaceStore).workflow
          .activeWorkflow?.filename
      })
      await expect
        .poll(() => comfyPage.page.title())
        .toBe(`*${workflowName} - ComfyUI`)
    })

    // Failing on CI
    // Cannot reproduce locally
    test.skip('Can display workflow name with unsaved changes', async ({
      comfyPage
    }) => {
      const workflowName = await comfyPage.page.evaluate(async () => {
        return (window.app!.extensionManager as WorkspaceStore).workflow
          .activeWorkflow?.filename
      })
      expect(await comfyPage.page.title()).toBe(`${workflowName} - ComfyUI`)

      await comfyPage.menu.topbar.saveWorkflow('test')
      expect(await comfyPage.page.title()).toBe('test - ComfyUI')

      const textBox = comfyPage.widgetTextBox
      await textBox.fill('Hello World')
      await comfyPage.canvasOps.clickEmptySpace()
      expect(await comfyPage.page.title()).toBe(`*test - ComfyUI`)

      // Delete the saved workflow for cleanup.
      await comfyPage.page.evaluate(async () => {
        return (
          window.app!.extensionManager as WorkspaceStore
        ).workflow.activeWorkflow?.delete()
      })
    })
  })

  test.describe('Legacy Menu', () => {
    test.beforeEach(async ({ comfyPage }) => {
      await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
    })

    test('Can display default title', async ({ comfyPage }) => {
      await expect.poll(() => comfyPage.page.title()).toBe('ComfyUI')
    })
  })
})
