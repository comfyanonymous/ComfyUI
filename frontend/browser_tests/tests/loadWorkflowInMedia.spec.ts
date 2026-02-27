import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'

test.beforeEach(async ({ comfyPage }) => {
  await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
})

test.describe(
  'Load Workflow in Media',
  { tag: ['@screenshot', '@workflow'] },
  () => {
    const fileNames = [
      'workflow.webp',
      'edited_workflow.webp',
      'no_workflow.webp',
      'large_workflow.webp',
      'workflow_prompt_parameters.png',
      'workflow_itxt.png',
      'workflow.webm',
      // Skipped due to 3d widget unstable visual result.
      // 3d widget shows grid after fully loaded.
      // 'workflow.glb',
      'workflow.mp4',
      'workflow.mov',
      'workflow.m4v',
      'workflow.svg'
      // TODO: Re-enable after fixing test asset to use core nodes only
      // Currently opens missing nodes dialog which is outside scope of AVIF loading functionality
      // 'workflow.avif'
    ]
    fileNames.forEach(async (fileName) => {
      test(`Load workflow in ${fileName} (drop from filesystem)`, async ({
        comfyPage
      }) => {
        await comfyPage.dragDrop.dragAndDropFile(`workflowInMedia/${fileName}`)
        await expect(comfyPage.canvas).toHaveScreenshot(`${fileName}.png`)
      })
    })

    const urls = [
      'https://comfyanonymous.github.io/ComfyUI_examples/hidream/hidream_dev_example.png'
    ]
    urls.forEach(async (url) => {
      test(`Load workflow from URL ${url} (drop from different browser tabs)`, async ({
        comfyPage
      }) => {
        await comfyPage.dragDrop.dragAndDropURL(url)
        const readableName = url.split('/').pop()
        await expect(comfyPage.canvas).toHaveScreenshot(
          `dropped_workflow_url_${readableName}.png`
        )
      })
    })
  }
)
