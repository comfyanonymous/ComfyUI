import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../../fixtures/ComfyPage'

test.describe('Workflows sidebar', () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')
    await comfyPage.settings.setSetting(
      'Comfy.Workflow.WorkflowTabsPosition',
      'Sidebar'
    )

    // Open the sidebar
    const tab = comfyPage.menu.workflowsTab
    await tab.open()
  })

  test.afterEach(async ({ comfyPage }) => {
    await comfyPage.workflow.setupWorkflowsDirectory({})
  })

  test('Can create new blank workflow', async ({ comfyPage }) => {
    const tab = comfyPage.menu.workflowsTab
    expect(await tab.getOpenedWorkflowNames()).toEqual([
      '*Unsaved Workflow.json'
    ])

    await comfyPage.command.executeCommand('Comfy.NewBlankWorkflow')
    expect(await tab.getOpenedWorkflowNames()).toEqual([
      '*Unsaved Workflow.json',
      '*Unsaved Workflow (2).json'
    ])
  })

  test('Can show top level saved workflows', async ({ comfyPage }) => {
    await comfyPage.workflow.setupWorkflowsDirectory({
      'workflow1.json': 'default.json',
      'workflow2.json': 'default.json'
    })

    const tab = comfyPage.menu.workflowsTab
    await tab.open()
    expect(await tab.getTopLevelSavedWorkflowNames()).toEqual(
      expect.arrayContaining(['workflow1.json', 'workflow2.json'])
    )
  })

  test('Can duplicate workflow', async ({ comfyPage }) => {
    const tab = comfyPage.menu.workflowsTab
    await comfyPage.menu.topbar.saveWorkflow('workflow1.json')

    expect(await tab.getTopLevelSavedWorkflowNames()).toEqual(
      expect.arrayContaining(['workflow1.json'])
    )

    await comfyPage.command.executeCommand('Comfy.DuplicateWorkflow')
    expect(await tab.getOpenedWorkflowNames()).toEqual([
      'workflow1.json',
      '*workflow1 (Copy).json'
    ])

    await comfyPage.command.executeCommand('Comfy.DuplicateWorkflow')
    expect(await tab.getOpenedWorkflowNames()).toEqual([
      'workflow1.json',
      '*workflow1 (Copy).json',
      '*workflow1 (Copy) (2).json'
    ])

    await comfyPage.command.executeCommand('Comfy.DuplicateWorkflow')
    expect(await tab.getOpenedWorkflowNames()).toEqual([
      'workflow1.json',
      '*workflow1 (Copy).json',
      '*workflow1 (Copy) (2).json',
      '*workflow1 (Copy) (3).json'
    ])
  })

  test('Can open workflow after insert', async ({ comfyPage }) => {
    await comfyPage.workflow.setupWorkflowsDirectory({
      'workflow1.json': 'nodes/single_ksampler.json'
    })

    const tab = comfyPage.menu.workflowsTab
    await tab.open()
    await comfyPage.command.executeCommand('Comfy.LoadDefaultWorkflow')
    const originalNodeCount = await comfyPage.nodeOps.getNodeCount()

    await tab.insertWorkflow(tab.getPersistedItem('workflow1.json'))
    await expect
      .poll(() => comfyPage.nodeOps.getNodeCount())
      .toEqual(originalNodeCount + 1)

    await tab.getPersistedItem('workflow1.json').click()
    await expect.poll(() => comfyPage.nodeOps.getNodeCount()).toEqual(1)
  })

  test('Can rename nested workflow from opened workflow item', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.setupWorkflowsDirectory({
      foo: {
        'bar.json': 'default.json'
      }
    })

    const tab = comfyPage.menu.workflowsTab
    await tab.open()
    // Switch to the parent folder
    await tab.getPersistedItem('foo').click()
    // Switch to the nested workflow
    await tab.getPersistedItem('bar').click()

    const openedWorkflow = tab.getOpenedItem('foo/bar')
    await tab.renameWorkflow(openedWorkflow, 'foo/baz')
    expect(await tab.getOpenedWorkflowNames()).toEqual([
      '*Unsaved Workflow.json',
      'foo/baz.json'
    ])
  })

  test('Can save workflow as', async ({ comfyPage }) => {
    await comfyPage.command.executeCommand('Comfy.NewBlankWorkflow')
    await comfyPage.menu.topbar.saveWorkflowAs('workflow3.json')
    await expect
      .poll(() => comfyPage.menu.workflowsTab.getOpenedWorkflowNames())
      .toEqual(['*Unsaved Workflow.json', 'workflow3.json'])

    await comfyPage.menu.topbar.saveWorkflowAs('workflow4.json')
    await expect
      .poll(() => comfyPage.menu.workflowsTab.getOpenedWorkflowNames())
      .toEqual(['*Unsaved Workflow.json', 'workflow3.json', 'workflow4.json'])
  })

  test('Exported workflow does not contain localized slot names', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow('default')
    const exportedWorkflow = await comfyPage.workflow.getExportedWorkflow({
      api: false
    })
    expect(exportedWorkflow).toBeDefined()
    for (const node of exportedWorkflow.nodes) {
      for (const slot of node.inputs ?? []) {
        expect(slot.localized_name).toBeUndefined()
        expect(slot.label).toBeUndefined()
      }
      for (const slot of node.outputs ?? []) {
        expect(slot.localized_name).toBeUndefined()
        expect(slot.label).toBeUndefined()
      }
    }
  })

  test('Can export same workflow with different locales', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow('default')

    // Setup download listener before triggering the export
    const downloadPromise = comfyPage.page.waitForEvent('download')
    await comfyPage.menu.topbar.exportWorkflow('exported_default.json')

    // Wait for the download and get the file content
    const download = await downloadPromise
    expect(download.suggestedFilename()).toBe('exported_default.json')

    // Get the exported workflow content
    const downloadedContent = await comfyPage.workflow.getExportedWorkflow({
      api: false
    })

    await comfyPage.settings.setSetting('Comfy.Locale', 'zh')
    await comfyPage.setup()

    const downloadedContentZh = await comfyPage.workflow.getExportedWorkflow({
      api: false
    })

    // Compare the exported workflow with the original
    delete downloadedContent.id
    delete downloadedContentZh.id
    expect(downloadedContent).toBeDefined()
    expect(downloadedContent).toEqual(downloadedContentZh)
  })

  test('Can save workflow as with same name', async ({ comfyPage }) => {
    await comfyPage.menu.topbar.saveWorkflow('workflow5.json')
    expect(await comfyPage.menu.workflowsTab.getOpenedWorkflowNames()).toEqual([
      'workflow5.json'
    ])

    await comfyPage.menu.topbar.saveWorkflowAs('workflow5.json')
    await comfyPage.confirmDialog.click('overwrite')
    expect(await comfyPage.menu.workflowsTab.getOpenedWorkflowNames()).toEqual([
      'workflow5.json'
    ])
  })

  test('Can save temporary workflow with unmodified name', async ({
    comfyPage
  }) => {
    expect(await comfyPage.workflow.isCurrentWorkflowModified()).toBe(false)

    await comfyPage.menu.topbar.saveWorkflow('Unsaved Workflow')
    // Should not trigger the overwrite dialog
    expect(
      await comfyPage.page.locator('.comfy-modal-content:visible').count()
    ).toBe(0)

    expect(await comfyPage.workflow.isCurrentWorkflowModified()).toBe(false)
  })

  test('Can overwrite other workflows with save as', async ({ comfyPage }) => {
    const topbar = comfyPage.menu.topbar
    await topbar.saveWorkflow('workflow1.json')
    await topbar.saveWorkflowAs('workflow2.json')
    await comfyPage.nextFrame()
    await expect
      .poll(() => comfyPage.menu.workflowsTab.getOpenedWorkflowNames())
      .toEqual(['workflow1.json', 'workflow2.json'])
    await expect
      .poll(() => comfyPage.menu.workflowsTab.getActiveWorkflowName())
      .toEqual('workflow2.json')

    await topbar.saveWorkflowAs('workflow1.json')
    await comfyPage.confirmDialog.click('overwrite')
    // The old workflow1.json should be deleted and the new one should be saved.
    await expect
      .poll(() => comfyPage.menu.workflowsTab.getOpenedWorkflowNames())
      .toEqual(['workflow2.json', 'workflow1.json'])
    await expect
      .poll(() => comfyPage.menu.workflowsTab.getActiveWorkflowName())
      .toEqual('workflow1.json')
  })

  test('Does not report warning when switching between opened workflows', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow('missing/missing_nodes')
    await comfyPage.page
      .locator('.p-dialog')
      .getByRole('button', { name: 'Close' })
      .click({ force: true })
    await comfyPage.page.locator('.p-dialog').waitFor({ state: 'hidden' })

    // Load blank workflow
    await comfyPage.menu.workflowsTab.open()
    await comfyPage.command.executeCommand('Comfy.NewBlankWorkflow')

    // Switch back to the missing_nodes workflow
    await comfyPage.menu.workflowsTab.switchToWorkflow('missing_nodes')

    await expect(
      comfyPage.page.locator('.comfy-missing-nodes')
    ).not.toBeVisible()
  })

  test('Can close saved-workflows from the open workflows section', async ({
    comfyPage
  }) => {
    await comfyPage.menu.topbar.saveWorkflow(
      `tempWorkflow-${test.info().title}`
    )
    const closeButton = comfyPage.page.locator(
      '.comfyui-workflows-open .close-workflow-button'
    )
    await closeButton.click()
    expect(await comfyPage.menu.workflowsTab.getOpenedWorkflowNames()).toEqual([
      '*Unsaved Workflow.json'
    ])
  })

  test('Can close saved workflow with command', async ({ comfyPage }) => {
    const tab = comfyPage.menu.workflowsTab
    await comfyPage.menu.topbar.saveWorkflow('workflow1.json')
    await comfyPage.command.executeCommand('Workspace.CloseWorkflow')
    expect(await tab.getOpenedWorkflowNames()).toEqual([
      '*Unsaved Workflow.json'
    ])
  })

  test('Can delete workflows (confirm disabled)', async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.Workflow.ConfirmDelete', false)

    const { topbar, workflowsTab } = comfyPage.menu

    const filename = 'workflow18.json'
    await topbar.saveWorkflow(filename)
    expect(await workflowsTab.getOpenedWorkflowNames()).toEqual([filename])

    await workflowsTab.getOpenedItem(filename).click({ button: 'right' })
    await comfyPage.nextFrame()
    await comfyPage.contextMenu.clickMenuItem('Delete')
    await comfyPage.nextFrame()

    await expect(workflowsTab.getOpenedItem(filename)).not.toBeVisible()
    expect(await workflowsTab.getOpenedWorkflowNames()).toEqual([
      '*Unsaved Workflow.json'
    ])
  })

  test('Can delete workflows', async ({ comfyPage }) => {
    const { topbar, workflowsTab } = comfyPage.menu

    const filename = 'workflow18.json'
    await topbar.saveWorkflow(filename)
    expect(await workflowsTab.getOpenedWorkflowNames()).toEqual([filename])

    await workflowsTab.getOpenedItem(filename).click({ button: 'right' })
    await comfyPage.contextMenu.clickMenuItem('Delete')
    await comfyPage.nextFrame()

    await comfyPage.confirmDialog.click('delete')

    await expect(workflowsTab.getOpenedItem(filename)).not.toBeVisible()
    expect(await workflowsTab.getOpenedWorkflowNames()).toEqual([
      '*Unsaved Workflow.json'
    ])
  })

  test('Can duplicate workflow from context menu', async ({ comfyPage }) => {
    await comfyPage.workflow.setupWorkflowsDirectory({
      'workflow1.json': 'default.json'
    })

    const { workflowsTab } = comfyPage.menu
    await workflowsTab.open()

    await workflowsTab
      .getPersistedItem('workflow1.json')
      .click({ button: 'right' })
    await comfyPage.contextMenu.clickMenuItem('Duplicate')
    await comfyPage.nextFrame()

    expect(await workflowsTab.getOpenedWorkflowNames()).toEqual([
      '*Unsaved Workflow.json',
      '*workflow1 (Copy).json'
    ])
  })

  test('Can drop workflow from workflows sidebar', async ({ comfyPage }) => {
    await comfyPage.workflow.setupWorkflowsDirectory({
      'workflow1.json': 'default.json'
    })

    await comfyPage.menu.workflowsTab.open()

    // Wait for workflow to appear in Browse section after sync
    const workflowItem =
      comfyPage.menu.workflowsTab.getPersistedItem('workflow1.json')
    await expect(workflowItem).toBeVisible({ timeout: 3000 })

    const nodeCount = await comfyPage.nodeOps.getGraphNodesCount()

    // Get the bounding box of the canvas element
    const canvasBoundingBox = (await comfyPage.page
      .locator('#graph-canvas')
      .boundingBox())!

    // Calculate the center position of the canvas
    const targetPosition = {
      x: canvasBoundingBox.x + canvasBoundingBox.width / 2,
      y: canvasBoundingBox.y + canvasBoundingBox.height / 2
    }

    await comfyPage.page.dragAndDrop(
      '.comfyui-workflows-browse .node-label:has-text("workflow1.json")',
      '#graph-canvas',
      { targetPosition }
    )

    // Wait for nodes to be inserted after drag-drop with retryable assertion
    await expect
      .poll(() => comfyPage.nodeOps.getGraphNodesCount(), { timeout: 3000 })
      .toBe(nodeCount * 2)
  })
})
