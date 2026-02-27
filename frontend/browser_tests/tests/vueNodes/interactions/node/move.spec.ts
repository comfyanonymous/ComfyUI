import {
  comfyExpect as expect,
  comfyPageFixture as test
} from '../../../../fixtures/ComfyPage'
import type { ComfyPage } from '../../../../fixtures/ComfyPage'
import type { Position } from '../../../../fixtures/types'

test.describe('Vue Node Moving', () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.VueNodes.Enabled', true)
    await comfyPage.vueNodes.waitForNodes()
  })

  const getLoadCheckpointHeaderPos = async (comfyPage: ComfyPage) => {
    const loadCheckpointHeaderPos = await comfyPage.page
      .getByText('Load Checkpoint')
      .boundingBox()

    if (!loadCheckpointHeaderPos)
      throw new Error('Load Checkpoint header not found')

    return loadCheckpointHeaderPos
  }

  const expectPosChanged = async (pos1: Position, pos2: Position) => {
    const diffX = Math.abs(pos2.x - pos1.x)
    const diffY = Math.abs(pos2.y - pos1.y)
    expect(diffX).toBeGreaterThan(0)
    expect(diffY).toBeGreaterThan(0)
  }

  test(
    'should allow moving nodes by dragging',
    { tag: '@screenshot' },
    async ({ comfyPage }) => {
      const loadCheckpointHeaderPos =
        await getLoadCheckpointHeaderPos(comfyPage)
      await comfyPage.canvasOps.dragAndDrop(loadCheckpointHeaderPos, {
        x: 256,
        y: 256
      })

      const newHeaderPos = await getLoadCheckpointHeaderPos(comfyPage)
      await expectPosChanged(loadCheckpointHeaderPos, newHeaderPos)

      await expect(comfyPage.canvas).toHaveScreenshot('vue-node-moved-node.png')
    }
  )

  test(
    '@mobile should allow moving nodes by dragging on touch devices',
    { tag: '@screenshot' },
    async ({ comfyPage }) => {
      // Disable minimap (gets in way of the node on small screens)
      await comfyPage.settings.setSetting('Comfy.Minimap.Visible', false)

      const loadCheckpointHeaderPos =
        await getLoadCheckpointHeaderPos(comfyPage)
      await comfyPage.canvasOps.panWithTouch(
        {
          x: 64,
          y: 64
        },
        loadCheckpointHeaderPos
      )

      const newHeaderPos = await getLoadCheckpointHeaderPos(comfyPage)
      await expectPosChanged(loadCheckpointHeaderPos, newHeaderPos)

      await expect(comfyPage.canvas).toHaveScreenshot(
        'vue-node-moved-node-touch.png'
      )
    }
  )
})
