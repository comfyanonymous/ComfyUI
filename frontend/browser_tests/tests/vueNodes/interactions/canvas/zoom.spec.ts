import {
  comfyExpect as expect,
  comfyPageFixture as test
} from '../../../../fixtures/ComfyPage'

test.describe('Vue Nodes Zoom', () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.VueNodes.Enabled', true)
    await comfyPage.settings.setSetting('LiteGraph.Canvas.MinFontSizeForLOD', 8)
    await comfyPage.vueNodes.waitForNodes()
  })

  test(
    'should not capture drag while zooming with ctrl+shift+drag',
    { tag: '@screenshot' },
    async ({ comfyPage }) => {
      const checkpointNode =
        comfyPage.vueNodes.getNodeByTitle('Load Checkpoint')
      const nodeBoundingBox = await checkpointNode.boundingBox()
      if (!nodeBoundingBox) throw new Error('Node bounding box not available')

      const nodeMidpointX = nodeBoundingBox.x + nodeBoundingBox.width / 2
      const nodeMidpointY = nodeBoundingBox.y + nodeBoundingBox.height / 2

      // Start the Ctrl+Shift drag-to-zoom on the canvas and continue dragging over
      // the node. The node should not capture the drag while drag-zooming.
      await comfyPage.page.keyboard.down('Control')
      await comfyPage.page.keyboard.down('Shift')
      await comfyPage.canvasOps.dragAndDrop(
        { x: 200, y: 300 },
        { x: nodeMidpointX, y: nodeMidpointY }
      )

      await expect(comfyPage.canvas).toHaveScreenshot(
        'zoomed-in-ctrl-shift.png'
      )
    }
  )
})
