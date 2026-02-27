import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'
import { getMiddlePoint } from '../fixtures/utils/litegraphUtils'

test.describe('Reroute Node', { tag: ['@screenshot', '@node'] }, () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')
  })

  test.afterEach(async ({ comfyPage }) => {
    await comfyPage.workflow.setupWorkflowsDirectory({})
  })

  test('loads from inserted workflow', async ({ comfyPage }) => {
    const workflowName = 'single_connected_reroute_node.json'
    await comfyPage.workflow.setupWorkflowsDirectory({
      [workflowName]: 'links/single_connected_reroute_node.json'
    })
    await comfyPage.setup()
    await comfyPage.menu.topbar.triggerTopbarCommand(['New'])

    // Insert the workflow
    const workflowsTab = comfyPage.menu.workflowsTab
    await workflowsTab.open()
    await workflowsTab.getPersistedItem(workflowName).click({ button: 'right' })
    const insertButton = comfyPage.page.locator('.p-contextmenu-item-link', {
      hasText: 'Insert'
    })
    await insertButton.click()

    // Close the sidebar tab
    await workflowsTab.tabButton.click()
    await workflowsTab.root.waitFor({ state: 'hidden' })
    await comfyPage.setFocusMode(true)

    await expect(comfyPage.canvas).toHaveScreenshot('reroute_inserted.png')
  })
})

test.describe(
  'LiteGraph Native Reroute Node',
  { tag: ['@screenshot', '@node'] },
  () => {
    test.beforeEach(async ({ comfyPage }) => {
      await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
      await comfyPage.settings.setSetting('LiteGraph.Reroute.SplineOffset', 80)
    })

    test('loads from workflow', async ({ comfyPage }) => {
      await comfyPage.workflow.loadWorkflow('reroute/native_reroute')
      await expect(comfyPage.canvas).toHaveScreenshot('native_reroute.png')
    })

    test('@2x @0.5x Can add reroute by alt clicking on link', async ({
      comfyPage
    }) => {
      const loadCheckpointNode = (
        await comfyPage.nodeOps.getNodeRefsByTitle('Load Checkpoint')
      )[0]
      const clipEncodeNode = (
        await comfyPage.nodeOps.getNodeRefsByTitle('CLIP Text Encode (Prompt)')
      )[0]

      const slot1 = await loadCheckpointNode.getOutput(1)
      const slot2 = await clipEncodeNode.getInput(0)
      const middlePoint = getMiddlePoint(
        await slot1.getPosition(),
        await slot2.getPosition()
      )

      await comfyPage.page.keyboard.down('Alt')
      await comfyPage.page.mouse.click(middlePoint.x, middlePoint.y)
      await comfyPage.page.keyboard.up('Alt')

      await expect(comfyPage.canvas).toHaveScreenshot(
        'native_reroute_alt_click.png'
      )
    })

    test('Can add reroute by clicking middle of link context menu', async ({
      comfyPage
    }) => {
      const loadCheckpointNode = (
        await comfyPage.nodeOps.getNodeRefsByTitle('Load Checkpoint')
      )[0]
      const clipEncodeNode = (
        await comfyPage.nodeOps.getNodeRefsByTitle('CLIP Text Encode (Prompt)')
      )[0]

      const slot1 = await loadCheckpointNode.getOutput(1)
      const slot2 = await clipEncodeNode.getInput(0)
      const middlePoint = getMiddlePoint(
        await slot1.getPosition(),
        await slot2.getPosition()
      )

      await comfyPage.page.mouse.click(middlePoint.x, middlePoint.y)
      await comfyPage.page
        .locator('.litecontextmenu .litemenu-entry', { hasText: 'Add Reroute' })
        .click()

      await expect(comfyPage.canvas).toHaveScreenshot(
        'native_reroute_context_menu.png'
      )
    })

    test('Can delete link that is connected to two reroutes', async ({
      comfyPage
    }) => {
      // https://github.com/Comfy-Org/ComfyUI_frontend/issues/4695
      await comfyPage.workflow.loadWorkflow(
        'reroute/single-native-reroute-default-workflow'
      )

      // To find the clickable midpoint button, we use the hardcoded value from the browser logs
      // since the link is a bezier curve and not a straight line.
      const middlePoint = { x: 359.4188232421875, y: 468.7716979980469 }

      // Click the middle point of the link to open the context menu.
      await comfyPage.page.mouse.click(middlePoint.x, middlePoint.y)

      // Click the "Delete" context menu option.
      await comfyPage.page
        .locator('.litecontextmenu .litemenu-entry', { hasText: 'Delete' })
        .click()

      await expect(comfyPage.canvas).toHaveScreenshot(
        'native_reroute_delete_from_midpoint_context_menu.png'
      )
    })
  }
)
