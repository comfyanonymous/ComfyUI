import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'

test.beforeEach(async ({ comfyPage }) => {
  await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
})

// If an input is optional by node definition, it should be shown as
// a hollow circle no matter what shape it was defined in the workflow JSON.
test.describe('Optional input', { tag: ['@screenshot', '@node'] }, () => {
  test('No shape specified', async ({ comfyPage }) => {
    await comfyPage.workflow.loadWorkflow('inputs/optional_input_no_shape')
    await expect(comfyPage.canvas).toHaveScreenshot('optional_input.png')
  })

  test('Wrong shape specified', async ({ comfyPage }) => {
    await comfyPage.workflow.loadWorkflow('inputs/optional_input_wrong_shape')
    await expect(comfyPage.canvas).toHaveScreenshot('optional_input.png')
  })

  test('Correct shape specified', async ({ comfyPage }) => {
    await comfyPage.workflow.loadWorkflow('inputs/optional_input_correct_shape')
    await expect(comfyPage.canvas).toHaveScreenshot('optional_input.png')
  })

  test('Force input', async ({ comfyPage }) => {
    await comfyPage.workflow.loadWorkflow('inputs/force_input')
    await expect(comfyPage.canvas).toHaveScreenshot('force_input.png')
  })

  test('Default input', async ({ comfyPage }) => {
    await comfyPage.workflow.loadWorkflow('inputs/default_input')
    await expect(comfyPage.canvas).toHaveScreenshot('default_input.png')
  })

  test('Only optional inputs', async ({ comfyPage }) => {
    await comfyPage.workflow.loadWorkflow('inputs/only_optional_inputs')
    expect(await comfyPage.nodeOps.getGraphNodesCount()).toBe(1)
    await expect(
      comfyPage.page.locator('.comfy-missing-nodes')
    ).not.toBeVisible()

    // If the node's multiline text widget is visible, then it was loaded successfully
    await expect(comfyPage.page.locator('.comfy-multiline-input')).toHaveCount(
      1
    )
  })
  test('Old workflow with converted input', async ({ comfyPage }) => {
    await comfyPage.workflow.loadWorkflow('inputs/old_workflow_converted_input')
    const node = await comfyPage.nodeOps.getNodeRefById('1')
    const inputs = (await node.getProperty('inputs')) as {
      name: string
      link?: number | null
    }[]
    const vaeInput = inputs.find((w) => w.name === 'vae')
    const convertedInput = inputs.find((w) => w.name === 'strength')

    expect(vaeInput).toBeDefined()
    expect(convertedInput).toBeDefined()
    expect(vaeInput!.link).toBeNull()
    expect(convertedInput!.link).not.toBeNull()
  })
  test('Renamed converted input', async ({ comfyPage }) => {
    await comfyPage.workflow.loadWorkflow('inputs/renamed_converted_widget')
    const node = await comfyPage.nodeOps.getNodeRefById('3')
    const inputs = (await node.getProperty('inputs')) as { name: string }[]
    const renamedInput = inputs.find((w) => w.name === 'breadth')
    expect(renamedInput).toBeUndefined()
  })
  test('slider', async ({ comfyPage }) => {
    await comfyPage.workflow.loadWorkflow('inputs/simple_slider')
    await expect(comfyPage.canvas).toHaveScreenshot('simple_slider.png')
  })
  test('unknown converted widget', async ({ comfyPage }) => {
    await comfyPage.settings.setSetting(
      'Comfy.Workflow.ShowMissingNodesWarning',
      false
    )
    await comfyPage.workflow.loadWorkflow(
      'missing/missing_nodes_converted_widget'
    )
    await expect(comfyPage.canvas).toHaveScreenshot(
      'missing_nodes_converted_widget.png'
    )
  })
  test('dynamically added input', async ({ comfyPage }) => {
    await comfyPage.workflow.loadWorkflow('inputs/dynamically_added_input')
    await expect(comfyPage.canvas).toHaveScreenshot(
      'dynamically_added_input.png'
    )
  })
})
