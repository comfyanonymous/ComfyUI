import type { Locator } from '@playwright/test'
import { expect } from '@playwright/test'

import type { Keybinding } from '../../src/platform/keybindings/types'
import { comfyPageFixture as test } from '../fixtures/ComfyPage'
import { DefaultGraphPositions } from '../fixtures/constants/defaultGraphPositions'
import { TestIds } from '../fixtures/selectors'

test.beforeEach(async ({ comfyPage }) => {
  await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
})

test.describe('Load workflow warning', { tag: '@ui' }, () => {
  test('Should display a warning when loading a workflow with missing nodes', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow('missing/missing_nodes')

    const missingNodesWarning = comfyPage.page.getByTestId(
      TestIds.dialogs.missingNodes
    )
    await expect(missingNodesWarning).toBeVisible()
  })

  test('Should display a warning when loading a workflow with missing nodes in subgraphs', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow('missing/missing_nodes_in_subgraph')

    const missingNodesWarning = comfyPage.page.getByTestId(
      TestIds.dialogs.missingNodes
    )
    await expect(missingNodesWarning).toBeVisible()

    // Verify the missing node text includes subgraph context
    const warningText = await missingNodesWarning.textContent()
    expect(warningText).toContain('MISSING_NODE_TYPE_IN_SUBGRAPH')
    expect(warningText).toContain('in subgraph')
  })
})

test('Does not report warning on undo/redo', async ({ comfyPage }) => {
  await comfyPage.settings.setSetting('Comfy.NodeSearchBoxImpl', 'v1 (legacy)')
  const missingNodesWarning = comfyPage.page.getByTestId(
    TestIds.dialogs.missingNodes
  )

  await comfyPage.workflow.loadWorkflow('missing/missing_nodes')
  await expect(missingNodesWarning).toBeVisible()
  await comfyPage.page.keyboard.press('Escape')
  await expect(missingNodesWarning).not.toBeVisible()

  // Wait for any async operations to complete after dialog closes
  await comfyPage.nextFrame()

  // Make a change to the graph
  await comfyPage.canvasOps.doubleClick()
  await comfyPage.searchBox.fillAndSelectFirstNode('KSampler')

  // Undo and redo the change
  await comfyPage.keyboard.undo()
  await expect(async () => {
    await expect(missingNodesWarning).not.toBeVisible()
  }).toPass({ timeout: 5000 })

  await comfyPage.keyboard.redo()
  await expect(async () => {
    await expect(missingNodesWarning).not.toBeVisible()
  }).toPass({ timeout: 5000 })
})

test.describe('Execution error', () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')
    await comfyPage.setup()
  })

  test('Should display an error message when an execution error occurs', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow('nodes/execution_error')
    await comfyPage.command.executeCommand('Comfy.QueuePrompt')
    await comfyPage.nextFrame()

    // Wait for the error overlay to be visible
    const errorOverlay = comfyPage.page.locator('[data-testid="error-overlay"]')
    await expect(errorOverlay).toBeVisible()
  })
})

test.describe('Missing models warning', () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting(
      'Comfy.Workflow.ShowMissingModelsWarning',
      true
    )
    await comfyPage.page.evaluate((url: string) => {
      return fetch(`${url}/api/devtools/cleanup_fake_model`)
    }, comfyPage.url)
  })

  test('Should display a warning when missing models are found', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow('missing/missing_models')

    const dialogTitle = comfyPage.page.getByText(
      'This workflow is missing models'
    )
    await expect(dialogTitle).toBeVisible()

    const downloadAllButton = comfyPage.page.getByText('Download all')
    await expect(downloadAllButton).toBeVisible()
  })

  test('Should display a warning when missing models are found in node properties', async ({
    comfyPage
  }) => {
    // Load workflow that has a node with models metadata at the node level
    await comfyPage.workflow.loadWorkflow(
      'missing/missing_models_from_node_properties'
    )

    const dialogTitle = comfyPage.page.getByText(
      'This workflow is missing models'
    )
    await expect(dialogTitle).toBeVisible()

    const downloadAllButton = comfyPage.page.getByText('Download all')
    await expect(downloadAllButton).toBeVisible()
  })

  test('Should not display a warning when no missing models are found', async ({
    comfyPage
  }) => {
    const modelFoldersRes = {
      status: 200,
      body: JSON.stringify([
        {
          name: 'text_encoders',
          folders: ['ComfyUI/models/text_encoders']
        }
      ])
    }
    await comfyPage.page.route(
      '**/api/experiment/models',
      (route) => route.fulfill(modelFoldersRes),
      { times: 1 }
    )

    // Reload page to trigger indexing of model folders
    await comfyPage.setup()

    const clipModelsRes = {
      status: 200,
      body: JSON.stringify([
        {
          name: 'fake_model.safetensors',
          pathIndex: 0
        }
      ])
    }
    await comfyPage.page.route(
      '**/api/experiment/models/text_encoders',
      (route) => route.fulfill(clipModelsRes),
      { times: 1 }
    )

    await comfyPage.workflow.loadWorkflow('missing/missing_models')

    const dialogTitle = comfyPage.page.getByText(
      'This workflow is missing models'
    )
    await expect(dialogTitle).not.toBeVisible()
  })

  test('Should not display warning when model metadata exists but widget values have changed', async ({
    comfyPage
  }) => {
    // This tests the scenario where outdated model metadata exists in the workflow
    // but the actual selected models (widget values) have changed
    await comfyPage.workflow.loadWorkflow(
      'missing/model_metadata_widget_mismatch'
    )

    // The missing models warning should NOT appear
    const dialogTitle = comfyPage.page.getByText(
      'This workflow is missing models'
    )
    await expect(dialogTitle).not.toBeVisible()
  })

  // Flaky test after parallelization
  // https://github.com/Comfy-Org/ComfyUI_frontend/pull/1400
  test.skip('Should download missing model when clicking download button', async ({
    comfyPage
  }) => {
    // The fake_model.safetensors is served by
    // https://github.com/Comfy-Org/ComfyUI_devtools/blob/main/__init__.py
    await comfyPage.workflow.loadWorkflow('missing/missing_models')

    const dialogTitle = comfyPage.page.getByText(
      'This workflow is missing models'
    )
    await expect(dialogTitle).toBeVisible()

    const downloadAllButton = comfyPage.page.getByText('Download all')
    await expect(downloadAllButton).toBeVisible()
    const downloadPromise = comfyPage.page.waitForEvent('download')
    await downloadAllButton.click()

    const download = await downloadPromise
    expect(download.suggestedFilename()).toBe('fake_model.safetensors')
  })

  test.describe('Do not show again checkbox', () => {
    let checkbox: Locator
    let closeButton: Locator

    test.beforeEach(async ({ comfyPage }) => {
      await comfyPage.settings.setSetting(
        'Comfy.Workflow.ShowMissingModelsWarning',
        true
      )
      await comfyPage.workflow.loadWorkflow('missing/missing_models')

      checkbox = comfyPage.page.getByLabel("Don't show this again")
      closeButton = comfyPage.page.getByLabel('Close')
    })

    test('Should disable warning dialog when checkbox is checked', async ({
      comfyPage
    }) => {
      const changeSettingPromise = comfyPage.page.waitForRequest(
        '**/api/settings/Comfy.Workflow.ShowMissingModelsWarning'
      )
      await checkbox.click()
      await changeSettingPromise

      await closeButton.click()

      const settingValue = await comfyPage.settings.getSetting(
        'Comfy.Workflow.ShowMissingModelsWarning'
      )
      expect(settingValue).toBe(false)
    })

    test('Should keep warning dialog enabled when checkbox is unchecked', async ({
      comfyPage
    }) => {
      await closeButton.click()

      const settingValue = await comfyPage.settings.getSetting(
        'Comfy.Workflow.ShowMissingModelsWarning'
      )
      expect(settingValue).toBe(true)
    })
  })
})

test.describe('Settings', () => {
  test('@mobile Should be visible on mobile', async ({ comfyPage }) => {
    await comfyPage.page.keyboard.press('Control+,')
    const settingsDialog = comfyPage.page.locator(
      '[data-testid="settings-dialog"]'
    )
    await expect(settingsDialog).toBeVisible()
    const contentArea = settingsDialog.locator('main')
    await expect(contentArea).toBeVisible()
    const isUsableHeight = await contentArea.evaluate(
      (el) => el.clientHeight > 30
    )
    expect(isUsableHeight).toBeTruthy()
  })

  test('Can open settings with hotkey', async ({ comfyPage }) => {
    await comfyPage.page.keyboard.down('ControlOrMeta')
    await comfyPage.page.keyboard.press(',')
    await comfyPage.page.keyboard.up('ControlOrMeta')
    const settingsLocator = comfyPage.page.locator(
      '[data-testid="settings-dialog"]'
    )
    await expect(settingsLocator).toBeVisible()
    await comfyPage.page.keyboard.press('Escape')
    await expect(settingsLocator).not.toBeVisible()
  })

  test('Can change canvas zoom speed setting', async ({ comfyPage }) => {
    const maxSpeed = 2.5
    await comfyPage.settings.setSetting('Comfy.Graph.ZoomSpeed', maxSpeed)
    await test.step('Setting should persist', async () => {
      expect(await comfyPage.settings.getSetting('Comfy.Graph.ZoomSpeed')).toBe(
        maxSpeed
      )
    })
  })

  test('Should persist keybinding setting', async ({ comfyPage }) => {
    // Open the settings dialog
    await comfyPage.page.keyboard.press('Control+,')
    await comfyPage.page.waitForSelector('[data-testid="settings-dialog"]')

    // Open the keybinding tab
    const settingsDialog = comfyPage.page.locator(
      '[data-testid="settings-dialog"]'
    )
    await settingsDialog
      .locator('nav [role="button"]', { hasText: 'Keybinding' })
      .click()
    await comfyPage.page.waitForSelector(
      '[placeholder="Search Keybindings..."]'
    )

    // Focus the 'New Blank Workflow' row
    const newBlankWorkflowRow = comfyPage.page.locator('tr', {
      has: comfyPage.page.getByRole('cell', { name: 'New Blank Workflow' })
    })
    await newBlankWorkflowRow.click()

    // Click edit button
    const editKeybindingButton = newBlankWorkflowRow.locator('.pi-pencil')
    await editKeybindingButton.click()

    // Set new keybinding
    const input = comfyPage.page.getByPlaceholder('Press keys for new binding')
    await input.press('Alt+n')

    const requestPromise = comfyPage.page.waitForRequest(
      (req) =>
        req.url().includes('/api/settings') &&
        !req.url().includes('/api/settings/') &&
        req.method() === 'POST'
    )

    // Save keybinding
    const saveButton = comfyPage.page
      .getByLabel('New Blank Workflow')
      .getByText('Save')
    await saveButton.click()

    const request = await requestPromise
    const expectedSetting: Keybinding = {
      commandId: 'Comfy.NewBlankWorkflow',
      combo: {
        key: 'n',
        ctrl: false,
        alt: true,
        shift: false
      }
    }
    expect(request.postData()).toContain(JSON.stringify(expectedSetting))
  })
})

test.describe('Support', () => {
  test('Should open external zendesk link with OSS tag', async ({
    comfyPage
  }) => {
    await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')

    // Prevent loading the external page
    await comfyPage.page
      .context()
      .route('https://support.comfy.org/**', (route) =>
        route.fulfill({ body: '<html></html>', contentType: 'text/html' })
      )

    const popupPromise = comfyPage.page.waitForEvent('popup')
    await comfyPage.menu.topbar.triggerTopbarCommand(['Help', 'Support'])
    const popup = await popupPromise

    const url = new URL(popup.url())
    expect(url.hostname).toBe('support.comfy.org')
    expect(url.searchParams.get('tf_42243568391700')).toBe('oss')

    await popup.close()
  })
})

test.describe('Error dialog', () => {
  test('Should display an error dialog when graph configure fails', async ({
    comfyPage
  }) => {
    await comfyPage.page.evaluate(() => {
      const graph = window.graph!
      ;(graph as { configure: () => void }).configure = () => {
        throw new Error('Error on configure!')
      }
    })

    await comfyPage.workflow.loadWorkflow('default')

    const errorDialog = comfyPage.page.locator('.comfy-error-report')
    await expect(errorDialog).toBeVisible()
  })

  test('Should display an error dialog when prompt execution fails', async ({
    comfyPage
  }) => {
    await comfyPage.page.evaluate(async () => {
      const app = window.app!
      app.api.queuePrompt = () => {
        throw new Error('Error on queuePrompt!')
      }
      await app.queuePrompt(0)
    })
    const errorDialog = comfyPage.page.locator('.comfy-error-report')
    await expect(errorDialog).toBeVisible()
  })
})

test.describe('Signin dialog', () => {
  test('Paste content to signin dialog should not paste node on canvas', async ({
    comfyPage
  }) => {
    const nodeNum = await comfyPage.nodeOps.getNodeCount()
    await comfyPage.canvas.click({
      position: DefaultGraphPositions.emptyLatentWidgetClick
    })
    await comfyPage.page.mouse.move(10, 10)
    await comfyPage.nextFrame()
    await comfyPage.clipboard.copy()

    const textBox = comfyPage.widgetTextBox
    await textBox.click()
    await textBox.fill('test_password')
    await textBox.press('Control+a')
    await textBox.press('Control+c')

    await comfyPage.page.evaluate(() => {
      void window.app!.extensionManager.dialog.showSignInDialog()
    })

    const input = comfyPage.page.locator('#comfy-org-sign-in-password')
    await input.waitFor({ state: 'visible' })
    await input.press('Control+v')
    await expect(input).toHaveValue('test_password')

    expect(await comfyPage.nodeOps.getNodeCount()).toBe(nodeNum)
  })
})
