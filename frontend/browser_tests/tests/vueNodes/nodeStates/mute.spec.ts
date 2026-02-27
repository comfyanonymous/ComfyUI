import {
  comfyExpect as expect,
  comfyPageFixture as test
} from '../../../fixtures/ComfyPage'

const MUTE_HOTKEY = 'Control+m'
const MUTE_OPACITY = '0.5'

test.describe('Vue Node Mute', () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.VueNodes.Enabled', true)
    await comfyPage.vueNodes.waitForNodes()
  })

  test(
    'should allow toggling mute on a selected node with hotkey',
    { tag: '@screenshot' },
    async ({ comfyPage }) => {
      await comfyPage.page.getByText('Load Checkpoint').click()
      await comfyPage.page.keyboard.press(MUTE_HOTKEY)

      const checkpointNode =
        comfyPage.vueNodes.getNodeByTitle('Load Checkpoint')
      await expect(checkpointNode).toHaveCSS('opacity', MUTE_OPACITY)
      await expect(comfyPage.canvas).toHaveScreenshot(
        'vue-node-muted-state.png'
      )

      await comfyPage.page.keyboard.press(MUTE_HOTKEY)
      await expect(checkpointNode).not.toHaveCSS('opacity', MUTE_OPACITY)
    }
  )

  test('should allow toggling mute on multiple selected nodes with hotkey', async ({
    comfyPage
  }) => {
    await comfyPage.page.getByText('Load Checkpoint').click()
    await comfyPage.page.getByText('KSampler').click({ modifiers: ['Control'] })

    const checkpointNode = comfyPage.vueNodes.getNodeByTitle('Load Checkpoint')
    const ksamplerNode = comfyPage.vueNodes.getNodeByTitle('KSampler')

    await comfyPage.page.keyboard.press(MUTE_HOTKEY)
    await expect(checkpointNode).toHaveCSS('opacity', MUTE_OPACITY)
    await expect(ksamplerNode).toHaveCSS('opacity', MUTE_OPACITY)

    await comfyPage.page.keyboard.press(MUTE_HOTKEY)
    await expect(checkpointNode).not.toHaveCSS('opacity', MUTE_OPACITY)
    await expect(ksamplerNode).not.toHaveCSS('opacity', MUTE_OPACITY)
  })
})
