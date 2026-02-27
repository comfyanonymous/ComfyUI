import {
  comfyExpect as expect,
  comfyPageFixture as test
} from '../../../fixtures/ComfyPage'

const CREATE_GROUP_HOTKEY = 'Control+g'

test.describe('Vue Node Groups', { tag: '@screenshot' }, () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.VueNodes.Enabled', true)
    await comfyPage.settings.setSetting('Comfy.Minimap.ShowGroups', true)
    await comfyPage.vueNodes.waitForNodes()
  })

  test('should allow creating groups with hotkey', async ({ comfyPage }) => {
    await comfyPage.page.getByText('Load Checkpoint').click()
    await comfyPage.page.getByText('KSampler').click({ modifiers: ['Control'] })
    await comfyPage.page.keyboard.press(CREATE_GROUP_HOTKEY)
    await comfyPage.nextFrame()
    await expect(comfyPage.canvas).toHaveScreenshot(
      'vue-groups-create-group.png'
    )
  })

  test('should allow fitting group to contents', async ({ comfyPage }) => {
    await comfyPage.setup()
    await comfyPage.workflow.loadWorkflow('groups/oversized_group')
    await comfyPage.keyboard.selectAll()
    await comfyPage.command.executeCommand('Comfy.Graph.FitGroupToContents')
    await comfyPage.nextFrame()
    await expect(comfyPage.canvas).toHaveScreenshot(
      'vue-groups-fit-to-contents.png'
    )
  })

  test('should move nested groups together when dragging outer group', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow('groups/nested-groups-1-inner-node')

    // Get initial positions with null guards
    const outerInitial =
      await comfyPage.canvasOps.getGroupPosition('Outer Group')
    const innerInitial =
      await comfyPage.canvasOps.getGroupPosition('Inner Group')

    const initialOffsetX = innerInitial.x - outerInitial.x
    const initialOffsetY = innerInitial.y - outerInitial.y

    // Drag the outer group
    const dragDelta = { x: 100, y: 80 }
    await comfyPage.canvasOps.dragGroup({
      name: 'Outer Group',
      deltaX: dragDelta.x,
      deltaY: dragDelta.y
    })

    // Use retrying assertion to wait for positions to update
    await expect(async () => {
      const outerFinal =
        await comfyPage.canvasOps.getGroupPosition('Outer Group')
      const innerFinal =
        await comfyPage.canvasOps.getGroupPosition('Inner Group')

      const finalOffsetX = innerFinal.x - outerFinal.x
      const finalOffsetY = innerFinal.y - outerFinal.y

      // Both groups should have moved
      expect(outerFinal.x).not.toBe(outerInitial.x)
      expect(innerFinal.x).not.toBe(innerInitial.x)

      // The relative offset should be maintained (inner group moved with outer)
      expect(finalOffsetX).toBeCloseTo(initialOffsetX, 0)
      expect(finalOffsetY).toBeCloseTo(initialOffsetY, 0)
    }).toPass({ timeout: 5000 })
  })
})
