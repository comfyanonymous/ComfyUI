import {
  comfyExpect as expect,
  comfyPageFixture as test
} from '../../../../fixtures/ComfyPage'

test.describe('Vue Node Resizing', () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.VueNodes.Enabled', true)
    await comfyPage.vueNodes.waitForNodes()
  })

  test('should resize node without position drift after selecting', async ({
    comfyPage
  }) => {
    // Get a Vue node fixture
    const node = await comfyPage.vueNodes.getFixtureByTitle('Load Checkpoint')
    const initialBox = await node.boundingBox()
    if (!initialBox) throw new Error('Node bounding box not found')

    // Select the node first (this was causing the bug)
    await node.header.click()

    // Get position after selection
    const selectedBox = await node.boundingBox()
    if (!selectedBox)
      throw new Error('Node bounding box not found after select')

    // Verify position unchanged after selection
    expect(selectedBox.x).toBeCloseTo(initialBox.x, 1)
    expect(selectedBox.y).toBeCloseTo(initialBox.y, 1)

    // Now resize from bottom-right corner
    const resizeStartX = selectedBox.x + selectedBox.width - 5
    const resizeStartY = selectedBox.y + selectedBox.height - 5

    await comfyPage.page.mouse.move(resizeStartX, resizeStartY)
    await comfyPage.page.mouse.down()
    await comfyPage.page.mouse.move(resizeStartX + 50, resizeStartY + 30)
    await comfyPage.page.mouse.up()

    // Get final position and size
    const finalBox = await node.boundingBox()
    if (!finalBox) throw new Error('Node bounding box not found after resize')

    // Position should NOT have changed (the bug was position drift)
    expect(finalBox.x).toBeCloseTo(initialBox.x, 1)
    expect(finalBox.y).toBeCloseTo(initialBox.y, 1)

    // Size should have increased
    expect(finalBox.width).toBeGreaterThan(initialBox.width)
    expect(finalBox.height).toBeGreaterThan(initialBox.height)
  })
})
