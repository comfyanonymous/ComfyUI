import {
  comfyExpect as expect,
  comfyPageFixture as test
} from '../../../../fixtures/ComfyPage'
import type { ComfyPage } from '../../../../fixtures/ComfyPage'
import { fitToViewInstant } from '../../../../helpers/fitToView'

test.describe('Vue Node Bring to Front', { tag: '@screenshot' }, () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
    await comfyPage.settings.setSetting('Comfy.VueNodes.Enabled', true)
    await comfyPage.workflow.loadWorkflow('vueNodes/simple-triple')
    await comfyPage.vueNodes.waitForNodes()
    await fitToViewInstant(comfyPage)
  })

  /**
   * Helper to get the z-index of a node by its title
   */
  async function getNodeZIndex(
    comfyPage: ComfyPage,
    title: string
  ): Promise<number> {
    const node = comfyPage.vueNodes.getNodeByTitle(title)
    const style = await node.getAttribute('style')
    if (!style) {
      throw new Error(
        `Node "${title}" has no style attribute (observed: ${style})`
      )
    }
    const match = style.match(/z-index:\s*(\d+)/)
    if (!match) {
      throw new Error(
        `Node "${title}" has no z-index in style (observed: "${style}")`
      )
    }
    return parseInt(match[1], 10)
  }

  /**
   * Helper to get the bounding box center of a node
   */
  async function getNodeCenter(
    comfyPage: ComfyPage,
    title: string
  ): Promise<{ x: number; y: number }> {
    const node = comfyPage.vueNodes.getNodeByTitle(title)
    const box = await node.boundingBox()
    if (!box) throw new Error(`Node "${title}" not found`)
    return { x: box.x + box.width / 2, y: box.y + box.height / 2 }
  }

  test('should bring overlapped node to front when clicking on it', async ({
    comfyPage
  }) => {
    // Get initial positions
    const clipCenter = await getNodeCenter(comfyPage, 'CLIP Text Encode')
    const ksamplerHeader = await comfyPage.page
      .getByText('KSampler')
      .boundingBox()
    if (!ksamplerHeader) throw new Error('KSampler header not found')

    // Drag KSampler on top of CLIP Text Encode
    await comfyPage.canvasOps.dragAndDrop(
      { x: ksamplerHeader.x + 50, y: ksamplerHeader.y + 10 },
      clipCenter
    )
    await comfyPage.nextFrame()

    // Screenshot showing KSampler on top of CLIP
    await expect(comfyPage.canvas).toHaveScreenshot(
      'bring-to-front-overlapped-before.png'
    )

    // KSampler should be on top (higher z-index) after being dragged
    const ksamplerZIndexBefore = await getNodeZIndex(comfyPage, 'KSampler')
    const clipZIndexBefore = await getNodeZIndex(comfyPage, 'CLIP Text Encode')
    expect(ksamplerZIndexBefore).toBeGreaterThan(clipZIndexBefore)

    // Click on CLIP Text Encode (underneath) - need to click on a visible part
    // Since KSampler is on top, we click on the edge of CLIP that should still be visible
    const clipNode = comfyPage.vueNodes.getNodeByTitle('CLIP Text Encode')
    const clipBox = await clipNode.boundingBox()
    if (!clipBox) throw new Error('CLIP node not found')

    // Click on a visible edge of CLIP
    await comfyPage.page.mouse.click(clipBox.x + 30, clipBox.y + 10)
    await comfyPage.nextFrame()

    // CLIP should now be on top - compare post-action z-indices
    const clipZIndexAfter = await getNodeZIndex(comfyPage, 'CLIP Text Encode')
    const ksamplerZIndexAfter = await getNodeZIndex(comfyPage, 'KSampler')
    expect(clipZIndexAfter).toBeGreaterThan(ksamplerZIndexAfter)

    // Screenshot showing CLIP now on top
    await expect(comfyPage.canvas).toHaveScreenshot(
      'bring-to-front-overlapped-after.png'
    )
  })

  test('should bring overlapped node to front when clicking on its widget', async ({
    comfyPage
  }) => {
    // Get CLIP Text Encode position (it has a text widget)
    const clipCenter = await getNodeCenter(comfyPage, 'CLIP Text Encode')

    // Get VAE Decode position and drag it on top of CLIP
    const vaeHeader = await comfyPage.page.getByText('VAE Decode').boundingBox()
    if (!vaeHeader) throw new Error('VAE Decode header not found')

    await comfyPage.canvasOps.dragAndDrop(
      { x: vaeHeader.x + 50, y: vaeHeader.y + 10 },
      { x: clipCenter.x - 50, y: clipCenter.y }
    )
    await comfyPage.nextFrame()

    // VAE should be on top after drag
    const vaeZIndexBefore = await getNodeZIndex(comfyPage, 'VAE Decode')
    const clipZIndexBefore = await getNodeZIndex(comfyPage, 'CLIP Text Encode')
    expect(vaeZIndexBefore).toBeGreaterThan(clipZIndexBefore)

    // Screenshot showing VAE on top
    await expect(comfyPage.canvas).toHaveScreenshot(
      'bring-to-front-widget-overlapped-before.png'
    )

    // Click on the text widget of CLIP Text Encode
    const clipNode = comfyPage.vueNodes.getNodeByTitle('CLIP Text Encode')
    const clipBox = await clipNode.boundingBox()
    if (!clipBox) throw new Error('CLIP node not found')
    await comfyPage.page.mouse.click(clipBox.x + 170, clipBox.y + 80)
    await comfyPage.nextFrame()

    // CLIP should now be on top - compare post-action z-indices
    const clipZIndexAfter = await getNodeZIndex(comfyPage, 'CLIP Text Encode')
    const vaeZIndexAfter = await getNodeZIndex(comfyPage, 'VAE Decode')
    expect(clipZIndexAfter).toBeGreaterThan(vaeZIndexAfter)

    // Screenshot showing CLIP now on top after widget click
    await expect(comfyPage.canvas).toHaveScreenshot(
      'bring-to-front-widget-overlapped-after.png'
    )
  })
})
