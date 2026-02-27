import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../../../../fixtures/ComfyPage'

test.beforeEach(async ({ comfyPage }) => {
  await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
})

test.describe('Vue Nodes - Delete Key Interaction', () => {
  test.beforeEach(async ({ comfyPage }) => {
    // Enable Vue nodes rendering
    await comfyPage.settings.setSetting('Comfy.VueNodes.Enabled', true)
    await comfyPage.settings.setSetting('Comfy.Graph.CanvasMenu', false)
    await comfyPage.setup()
  })

  test('Can select all and delete Vue nodes with Delete key', async ({
    comfyPage
  }) => {
    await comfyPage.vueNodes.waitForNodes()

    // Get initial Vue node count
    const initialNodeCount = await comfyPage.vueNodes.getNodeCount()
    expect(initialNodeCount).toBeGreaterThan(0)

    // Select all Vue nodes
    await comfyPage.keyboard.selectAll()

    // Verify all Vue nodes are selected
    const selectedCount = await comfyPage.vueNodes.getSelectedNodeCount()
    expect(selectedCount).toBe(initialNodeCount)

    // Delete with Delete key
    await comfyPage.vueNodes.deleteSelected()

    // Verify all Vue nodes were deleted
    const finalNodeCount = await comfyPage.vueNodes.getNodeCount()
    expect(finalNodeCount).toBe(0)
  })

  test('Can select specific Vue node and delete it', async ({ comfyPage }) => {
    await comfyPage.vueNodes.waitForNodes()

    // Get initial Vue node count
    const initialNodeCount = await comfyPage.vueNodes.getNodeCount()
    expect(initialNodeCount).toBeGreaterThan(0)

    // Get first Vue node ID and select it
    const nodeIds = await comfyPage.vueNodes.getNodeIds()
    await comfyPage.vueNodes.selectNode(nodeIds[0])

    // Verify selection
    const selectedCount = await comfyPage.vueNodes.getSelectedNodeCount()
    expect(selectedCount).toBe(1)

    // Delete with Delete key
    await comfyPage.vueNodes.deleteSelected()

    // Verify one Vue node was deleted
    const finalNodeCount = await comfyPage.vueNodes.getNodeCount()
    expect(finalNodeCount).toBe(initialNodeCount - 1)
  })

  test('Can select and delete Vue node with Backspace key', async ({
    comfyPage
  }) => {
    await comfyPage.vueNodes.waitForNodes()

    const initialNodeCount = await comfyPage.vueNodes.getNodeCount()

    // Select first Vue node
    const nodeIds = await comfyPage.vueNodes.getNodeIds()
    await comfyPage.vueNodes.selectNode(nodeIds[0])

    // Delete with Backspace key instead of Delete
    await comfyPage.vueNodes.deleteSelectedWithBackspace()

    // Verify Vue node was deleted
    const finalNodeCount = await comfyPage.vueNodes.getNodeCount()
    expect(finalNodeCount).toBe(initialNodeCount - 1)
  })

  test('Delete key does not delete node when typing in Vue node widgets', async ({
    comfyPage
  }) => {
    const initialNodeCount = await comfyPage.nodeOps.getGraphNodesCount()

    // Find a text input widget in a Vue node
    const textWidget = comfyPage.page
      .locator('input[type="text"], textarea')
      .first()

    // Click on text widget to focus it
    await textWidget.click()
    await textWidget.fill('test text')

    // Press Delete while focused on widget - should delete text, not node
    await textWidget.press('Delete')

    // Node count should remain the same
    const finalNodeCount = await comfyPage.nodeOps.getGraphNodesCount()
    expect(finalNodeCount).toBe(initialNodeCount)
  })

  test('Delete key does not delete node when nothing is selected', async ({
    comfyPage
  }) => {
    await comfyPage.vueNodes.waitForNodes()

    // Ensure no Vue nodes are selected
    await comfyPage.vueNodes.clearSelection()
    const selectedCount = await comfyPage.vueNodes.getSelectedNodeCount()
    expect(selectedCount).toBe(0)

    // Press Delete key - should not crash and should handle gracefully
    await comfyPage.page.keyboard.press('Delete')

    // Vue node count should remain the same
    const nodeCount = await comfyPage.vueNodes.getNodeCount()
    expect(nodeCount).toBeGreaterThan(0)
  })

  test('Can multi-select with Ctrl+click and delete multiple Vue nodes', async ({
    comfyPage
  }) => {
    await comfyPage.vueNodes.waitForNodes()
    const initialNodeCount = await comfyPage.vueNodes.getNodeCount()

    // Multi-select first two Vue nodes using Ctrl+click
    const nodeIds = await comfyPage.vueNodes.getNodeIds()
    const nodesToSelect = nodeIds.slice(0, 2)
    await comfyPage.vueNodes.selectNodes(nodesToSelect)

    // Verify expected nodes are selected
    const selectedCount = await comfyPage.vueNodes.getSelectedNodeCount()
    expect(selectedCount).toBe(nodesToSelect.length)

    // Delete selected Vue nodes
    await comfyPage.vueNodes.deleteSelected()

    // Verify expected nodes were deleted
    const finalNodeCount = await comfyPage.vueNodes.getNodeCount()
    expect(finalNodeCount).toBe(initialNodeCount - nodesToSelect.length)
  })
})
