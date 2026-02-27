import {
  comfyExpect as expect,
  comfyPageFixture as test
} from '../../../../fixtures/ComfyPage'

test.describe('Vue Upload Widgets', () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.VueNodes.Enabled', true)
    await comfyPage.vueNodes.waitForNodes()
  })

  test('should hide canvas-only upload buttons', async ({ comfyPage }) => {
    await comfyPage.setup()
    await comfyPage.workflow.loadWorkflow('widgets/all_load_widgets')
    await comfyPage.vueNodes.waitForNodes()

    await expect(
      comfyPage.page.getByText('choose file to upload', { exact: true })
    ).not.toBeVisible()

    await expect
      .poll(() => comfyPage.page.getByText('Error loading image').count())
      .toBeGreaterThan(0)
    await expect
      .poll(() => comfyPage.page.getByText('Error loading video').count())
      .toBeGreaterThan(0)
  })
})
