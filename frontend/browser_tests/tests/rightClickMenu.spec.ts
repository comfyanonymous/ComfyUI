import { expect } from '@playwright/test'

import { NodeBadgeMode } from '../../src/types/nodeSource'
import { comfyPageFixture as test } from '../fixtures/ComfyPage'
import { DefaultGraphPositions } from '../fixtures/constants/defaultGraphPositions'

test.beforeEach(async ({ comfyPage }) => {
  await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
})

test.describe(
  'Canvas Right Click Menu',
  { tag: ['@screenshot', '@ui'] },
  () => {
    test('Can add node', async ({ comfyPage }) => {
      await comfyPage.canvasOps.rightClick()
      await expect(comfyPage.canvas).toHaveScreenshot('right-click-menu.png')
      await comfyPage.page.getByText('Add Node').click()
      await comfyPage.nextFrame()
      await comfyPage.page.getByText('loaders').click()
      await comfyPage.nextFrame()
      await comfyPage.page.getByText('Load VAE').click()
      await comfyPage.nextFrame()
      await expect(comfyPage.canvas).toHaveScreenshot('add-node-node-added.png')
    })

    test('Can add group', async ({ comfyPage }) => {
      await comfyPage.canvasOps.rightClick()
      await expect(comfyPage.canvas).toHaveScreenshot('right-click-menu.png')
      await comfyPage.page.getByText('Add Group', { exact: true }).click()
      await comfyPage.nextFrame()
      await expect(comfyPage.canvas).toHaveScreenshot(
        'add-group-group-added.png'
      )
    })

    test('Can convert to group node', async ({ comfyPage }) => {
      await comfyPage.nodeOps.selectNodes(['CLIP Text Encode (Prompt)'])
      await expect(comfyPage.canvas).toHaveScreenshot('selected-2-nodes.png')
      await comfyPage.canvasOps.rightClick()
      await comfyPage.contextMenu.clickMenuItem(
        'Convert to Group Node (Deprecated)'
      )
      await comfyPage.nextFrame()
      await comfyPage.nodeOps.promptDialogInput.fill('GroupNode2CLIP')
      await comfyPage.page.keyboard.press('Enter')
      await comfyPage.nodeOps.promptDialogInput.waitFor({ state: 'hidden' })
      await comfyPage.nextFrame()
      await expect(comfyPage.canvas).toHaveScreenshot(
        'right-click-node-group-node.png'
      )
    })
  }
)

test.describe('Node Right Click Menu', { tag: ['@screenshot', '@ui'] }, () => {
  test('Can open properties panel', async ({ comfyPage }) => {
    await comfyPage.canvas.click({
      position: DefaultGraphPositions.emptyLatentWidgetClick,
      button: 'right'
    })
    await comfyPage.page.mouse.move(10, 10)
    await comfyPage.nextFrame()
    await expect(comfyPage.canvas).toHaveScreenshot('right-click-node.png')
    await comfyPage.page.getByText('Properties Panel').click()
    await comfyPage.nextFrame()
    await expect(comfyPage.canvas).toHaveScreenshot(
      'right-click-node-properties-panel.png'
    )
  })

  test('Can collapse', async ({ comfyPage }) => {
    await comfyPage.canvas.click({
      position: DefaultGraphPositions.emptyLatentWidgetClick,
      button: 'right'
    })
    await comfyPage.page.mouse.move(10, 10)
    await comfyPage.nextFrame()
    await expect(comfyPage.canvas).toHaveScreenshot('right-click-node.png')
    await comfyPage.page.getByText('Collapse').click()
    await comfyPage.nextFrame()
    await expect(comfyPage.canvas).toHaveScreenshot(
      'right-click-node-collapsed.png'
    )
  })

  test('Can collapse (Node Badge)', async ({ comfyPage }) => {
    await comfyPage.settings.setSetting(
      'Comfy.NodeBadge.NodeIdBadgeMode',
      NodeBadgeMode.ShowAll
    )
    await comfyPage.settings.setSetting(
      'Comfy.NodeBadge.NodeSourceBadgeMode',
      NodeBadgeMode.ShowAll
    )

    await comfyPage.canvas.click({
      position: DefaultGraphPositions.emptyLatentWidgetClick,
      button: 'right'
    })
    await comfyPage.page.mouse.move(10, 10)
    await comfyPage.nextFrame()
    await comfyPage.page.getByText('Collapse').click()
    await comfyPage.nextFrame()
    await expect(comfyPage.canvas).toHaveScreenshot(
      'right-click-node-collapsed-badge.png'
    )
  })

  test('Can bypass', async ({ comfyPage }) => {
    await comfyPage.canvas.click({
      position: DefaultGraphPositions.emptyLatentWidgetClick,
      button: 'right'
    })
    await comfyPage.page.mouse.move(10, 10)
    await comfyPage.nextFrame()
    await expect(comfyPage.canvas).toHaveScreenshot('right-click-node.png')
    await comfyPage.page.getByText('Bypass').click()
    await comfyPage.nextFrame()
    await expect(comfyPage.canvas).toHaveScreenshot(
      'right-click-node-bypassed.png'
    )
  })

  test('Can pin and unpin', async ({ comfyPage }) => {
    await comfyPage.canvas.click({
      position: DefaultGraphPositions.emptyLatentWidgetClick,
      button: 'right'
    })
    await comfyPage.page.mouse.move(10, 10)
    await comfyPage.nextFrame()
    await expect(comfyPage.canvas).toHaveScreenshot('right-click-node.png')
    await comfyPage.page.click('.litemenu-entry:has-text("Pin")')
    await comfyPage.nextFrame()

    // Get EmptyLatentImage node title position dynamically (for dragging)
    const emptyLatentNode = await comfyPage.nodeOps.getNodeRefById(5)
    const titlePos = await emptyLatentNode.getTitlePosition()
    await comfyPage.canvasOps.dragAndDrop(titlePos, { x: 16, y: 16 })
    await expect(comfyPage.canvas).toHaveScreenshot('node-pinned.png')
    await comfyPage.canvas.click({
      position: DefaultGraphPositions.emptyLatentWidgetClick,
      button: 'right'
    })
    await comfyPage.page.mouse.move(10, 10)
    await comfyPage.nextFrame()
    await expect(comfyPage.canvas).toHaveScreenshot(
      'right-click-pinned-node.png'
    )
    await comfyPage.page.click('.litemenu-entry:has-text("Unpin")')
    await comfyPage.nextFrame()
    await comfyPage.canvas.click({
      position: DefaultGraphPositions.emptyLatentWidgetClick,
      button: 'right'
    })
    await comfyPage.page.mouse.move(10, 10)
    await comfyPage.nextFrame()
    await expect(comfyPage.canvas).toHaveScreenshot(
      'right-click-unpinned-node.png'
    )
  })

  test('Can move after unpin', async ({ comfyPage }) => {
    await comfyPage.canvas.click({
      position: DefaultGraphPositions.emptyLatentWidgetClick,
      button: 'right'
    })
    await comfyPage.page.mouse.move(10, 10)
    await comfyPage.nextFrame()
    await comfyPage.page.click('.litemenu-entry:has-text("Pin")')
    await comfyPage.nextFrame()
    await comfyPage.canvas.click({
      position: DefaultGraphPositions.emptyLatentWidgetClick,
      button: 'right'
    })
    await comfyPage.page.mouse.move(10, 10)
    await comfyPage.nextFrame()
    await comfyPage.page.click('.litemenu-entry:has-text("Unpin")')
    await comfyPage.nextFrame()

    // Get EmptyLatentImage node title position dynamically (for dragging)
    const emptyLatentNode = await comfyPage.nodeOps.getNodeRefById(5)
    const titlePos = await emptyLatentNode.getTitlePosition()
    await comfyPage.canvasOps.dragAndDrop(titlePos, { x: 200, y: 590 })
    await expect(comfyPage.canvas).toHaveScreenshot(
      'right-click-unpinned-node-moved.png'
    )
  })

  test('Can pin/unpin selected nodes', async ({ comfyPage }) => {
    await comfyPage.nodeOps.selectNodes(['CLIP Text Encode (Prompt)'])
    await comfyPage.page.keyboard.down('Control')
    await comfyPage.canvas.click({
      position: DefaultGraphPositions.emptyLatentWidgetClick,
      button: 'right'
    })
    await comfyPage.page.mouse.move(10, 10)
    await comfyPage.nextFrame()
    await comfyPage.page.click('.litemenu-entry:has-text("Pin")')
    await comfyPage.page.keyboard.up('Control')
    await comfyPage.nextFrame()
    await expect(comfyPage.canvas).toHaveScreenshot('selected-nodes-pinned.png')
    await comfyPage.canvas.click({
      position: DefaultGraphPositions.emptyLatentWidgetClick,
      button: 'right'
    })
    await comfyPage.page.mouse.move(10, 10)
    await comfyPage.nextFrame()
    await comfyPage.page.click('.litemenu-entry:has-text("Unpin")')
    await comfyPage.nextFrame()
    await expect(comfyPage.canvas).toHaveScreenshot(
      'selected-nodes-unpinned.png'
    )
  })

  test('Can clone pinned nodes', async ({ comfyPage }) => {
    const nodeCount = await comfyPage.nodeOps.getGraphNodesCount()
    const node = (await comfyPage.nodeOps.getFirstNodeRef())!
    await node.clickContextMenuOption('Pin')
    await comfyPage.nextFrame()
    await node.click('title', { button: 'right' })
    await expect(
      comfyPage.page.locator('.litemenu-entry:has-text("Unpin")')
    ).toBeAttached()
    const cloneItem = comfyPage.page.locator(
      '.litemenu-entry:has-text("Clone")'
    )
    await cloneItem.click()
    await expect(cloneItem).toHaveCount(0)
    await comfyPage.nextFrame()
    expect(await comfyPage.nodeOps.getGraphNodesCount()).toBe(nodeCount + 1)
  })
})
