import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'

test.beforeEach(async ({ comfyPage }) => {
  await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
  await comfyPage.settings.setSetting('Comfy.NodeSearchBoxImpl', 'v1 (legacy)')
})

test.describe('Record Audio Node', { tag: '@screenshot' }, () => {
  test('should add a record audio node and take a screenshot', async ({
    comfyPage
  }) => {
    // Open the search box by double clicking on the canvas
    await comfyPage.canvasOps.doubleClick()
    await expect(comfyPage.searchBox.input).toHaveCount(1)

    // Search for and add the RecordAudio node
    await comfyPage.searchBox.fillAndSelectFirstNode('RecordAudio')
    await comfyPage.nextFrame()

    // Verify the RecordAudio node was added
    const recordAudioNodes =
      await comfyPage.nodeOps.getNodeRefsByType('RecordAudio')
    expect(recordAudioNodes.length).toBe(1)

    // Take a screenshot of the canvas with the RecordAudio node
    await expect(comfyPage.canvas).toHaveScreenshot('record_audio_node.png')
  })
})
