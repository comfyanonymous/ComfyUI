import {
  comfyExpect as expect,
  comfyPageFixture as test
} from '../../../../fixtures/ComfyPage'
import { TestIds } from '../../../../fixtures/selectors'

test.describe('Vue Nodes Renaming', () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.Graph.CanvasMenu', false)
    await comfyPage.settings.setSetting('Comfy.VueNodes.Enabled', true)
    await comfyPage.setup()
    await comfyPage.vueNodes.waitForNodes()
  })

  test('should display node title', async ({ comfyPage }) => {
    const vueNode = await comfyPage.vueNodes.getFixtureByTitle('KSampler')
    await expect(vueNode.header).toContainText('KSampler')
  })

  test('should allow title renaming by double clicking on the node header', async ({
    comfyPage
  }) => {
    const vueNode = await comfyPage.vueNodes.getFixtureByTitle('KSampler')
    // Test renaming with Enter
    await vueNode.setTitle('My Custom Sampler')
    await expect(await vueNode.getTitle()).toBe('My Custom Sampler')
    await expect(vueNode.header).toContainText('My Custom Sampler')

    // Test cancel with Escape
    await vueNode.title.dblclick()
    await comfyPage.nextFrame()
    await vueNode.titleInput.fill('This Should Be Cancelled')
    await vueNode.titleInput.press('Escape')
    await comfyPage.nextFrame()

    // Title should remain as the previously saved value
    await expect(await vueNode.getTitle()).toBe('My Custom Sampler')
  })

  test('Double click node body does not trigger edit', async ({
    comfyPage
  }) => {
    const loadCheckpointNode = comfyPage.vueNodes
      .getNodeByTitle('Load Checkpoint')
      .first()
    const nodeBbox = await loadCheckpointNode.boundingBox()
    if (!nodeBbox) throw new Error('Node not found')
    await loadCheckpointNode.dblclick()

    const editingTitleInput = comfyPage.page.getByTestId(
      TestIds.node.titleInput
    )
    await expect(editingTitleInput).not.toBeVisible()
  })
})
