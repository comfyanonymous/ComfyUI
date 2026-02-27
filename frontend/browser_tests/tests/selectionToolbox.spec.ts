import { expect } from '@playwright/test'
import type { Page } from '@playwright/test'

import { comfyPageFixture } from '../fixtures/ComfyPage'
import { TestIds } from '../fixtures/selectors'

const test = comfyPageFixture

test.beforeEach(async ({ comfyPage }) => {
  await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
})
const BLUE_COLOR = 'rgb(51, 51, 85)'
const RED_COLOR = 'rgb(85, 51, 51)'

const getColorPickerButton = (comfyPage: { page: Page }) =>
  comfyPage.page.getByTestId(TestIds.selectionToolbox.colorPickerButton)

const getColorPickerCurrentColor = (comfyPage: { page: Page }) =>
  comfyPage.page.getByTestId(TestIds.selectionToolbox.colorPickerCurrentColor)

const getColorPickerGroup = (comfyPage: { page: Page }) =>
  comfyPage.page.getByRole('group').filter({
    has: comfyPage.page.getByTestId(TestIds.selectionToolbox.colorBlue)
  })

test.describe('Selection Toolbox', { tag: ['@screenshot', '@ui'] }, () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.Canvas.SelectionToolbox', true)
  })

  test('shows selection toolbox', async ({ comfyPage }) => {
    // By default, selection toolbox should be enabled
    await expect(comfyPage.selectionToolbox).not.toBeVisible()

    // Select multiple nodes
    await comfyPage.nodeOps.selectNodes([
      'KSampler',
      'CLIP Text Encode (Prompt)'
    ])

    // Selection toolbox should be visible with multiple nodes selected
    await expect(comfyPage.selectionToolbox).toBeVisible()
    // Border is now drawn on canvas, check via screenshot
    await expect(comfyPage.canvas).toHaveScreenshot(
      'selection-toolbox-multiple-nodes-border.png'
    )
  })

  test('shows at correct position when node is pasted', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow('nodes/single_ksampler')
    await comfyPage.nodeOps.selectNodes(['KSampler'])
    await comfyPage.clipboard.copy()
    await comfyPage.page.mouse.move(100, 100)
    await comfyPage.clipboard.paste()

    const toolboxContainer = comfyPage.selectionToolbox
    await expect(toolboxContainer).toBeVisible()

    // Verify toolbox is positioned (canvas-based positioning has different coordinates)
    const boundingBox = await toolboxContainer.boundingBox()
    expect(boundingBox).not.toBeNull()
    // Canvas-based positioning can vary, just verify toolbox appears in reasonable bounds
    expect(boundingBox!.x).toBeGreaterThan(-200) // Not too far off-screen left
    expect(boundingBox!.x).toBeLessThan(1000) // Not too far off-screen right
    expect(boundingBox!.y).toBeGreaterThan(-100) // Not too far off-screen top
  })

  test('hide when select and drag happen at the same time', async ({
    comfyPage
  }) => {
    await comfyPage.workflow.loadWorkflow('nodes/single_ksampler')
    const node = (await comfyPage.nodeOps.getNodeRefsByTitle('KSampler'))[0]
    const nodePos = await node.getPosition()

    // Drag on the title of the node
    await comfyPage.page.mouse.move(nodePos.x + 100, nodePos.y - 15)
    await comfyPage.page.mouse.down()
    await comfyPage.page.mouse.move(nodePos.x + 200, nodePos.y + 200)
    await comfyPage.nextFrame()
    await expect(comfyPage.selectionToolbox).not.toBeVisible()
  })

  test('shows border only with multiple selections', async ({ comfyPage }) => {
    // Select single node
    await comfyPage.nodeOps.selectNodes(['KSampler'])

    // Selection toolbox should be visible but without border
    await expect(comfyPage.selectionToolbox).toBeVisible()
    // Border is now drawn on canvas, check via screenshot
    await expect(comfyPage.canvas).toHaveScreenshot(
      'selection-toolbox-single-node-no-border.png'
    )

    // Select multiple nodes
    await comfyPage.nodeOps.selectNodes([
      'KSampler',
      'CLIP Text Encode (Prompt)'
    ])

    // Selection border should show with multiple selections (canvas-based)
    await expect(comfyPage.canvas).toHaveScreenshot(
      'selection-toolbox-multiple-selections-border.png'
    )

    // Deselect to single node
    await comfyPage.nodeOps.selectNodes(['CLIP Text Encode (Prompt)'])

    // Border should be hidden again (canvas-based)
    await expect(comfyPage.canvas).toHaveScreenshot(
      'selection-toolbox-single-selection-no-border.png'
    )
  })

  test('displays bypass button in toolbox when nodes are selected', async ({
    comfyPage
  }) => {
    // A group + a KSampler node
    await comfyPage.workflow.loadWorkflow('groups/single_group')

    // Select group + node should show bypass button
    await comfyPage.page.focus('canvas')
    await comfyPage.page.keyboard.press('Control+A')
    await expect(
      comfyPage.page.locator(
        '.selection-toolbox *[data-testid="bypass-button"]'
      )
    ).toBeVisible()

    // Deselect node (Only group is selected) should hide bypass button
    await comfyPage.nodeOps.selectNodes(['KSampler'])
    await expect(
      comfyPage.page.locator(
        '.selection-toolbox *[data-testid="bypass-button"]'
      )
    ).not.toBeVisible()
  })

  test.describe('Color Picker', () => {
    test('displays color picker button and allows color selection', async ({
      comfyPage
    }) => {
      // Select a node
      await comfyPage.nodeOps.selectNodes(['KSampler'])

      // Color picker button should be visible
      const colorPickerButton = getColorPickerButton(comfyPage)
      await expect(colorPickerButton).toBeVisible()

      // Click color picker button
      await colorPickerButton.click()

      // Color picker dropdown should be visible
      const colorPickerGroup = getColorPickerGroup(comfyPage)
      await expect(colorPickerGroup).toBeVisible()

      // Select a color (e.g., blue)
      const blueColorOption = colorPickerGroup.getByTestId(
        TestIds.selectionToolbox.colorBlue
      )
      await blueColorOption.click()

      // Dropdown should close after selection
      await expect(colorPickerGroup).not.toBeVisible()

      // Node should have the selected color class/style
      // Note: Exact verification method depends on how color is applied to nodes
      const selectedNode = (
        await comfyPage.nodeOps.getNodeRefsByTitle('KSampler')
      )[0]
      expect(await selectedNode.getProperty('color')).not.toBeNull()
    })

    test('color picker shows current color of selected nodes', async ({
      comfyPage
    }) => {
      // Select multiple nodes
      await comfyPage.nodeOps.selectNodes([
        'KSampler',
        'CLIP Text Encode (Prompt)'
      ])

      const colorPickerButton = getColorPickerButton(comfyPage)
      const colorPickerCurrentColor = getColorPickerCurrentColor(comfyPage)

      // Initially should show default color
      await expect(colorPickerButton).not.toHaveAttribute('color')

      // Click color picker and select a color
      await colorPickerButton.click()
      const redColorOption = getColorPickerGroup(comfyPage).getByTestId(
        TestIds.selectionToolbox.colorRed
      )
      await redColorOption.click()

      // Button should now show the selected color
      await expect(colorPickerCurrentColor).toHaveCSS('color', RED_COLOR)
    })

    test('color picker shows mixed state for differently colored selections', async ({
      comfyPage
    }) => {
      // Select first node and color it
      await comfyPage.nodeOps.selectNodes(['KSampler'])
      await getColorPickerButton(comfyPage).click()
      await getColorPickerGroup(comfyPage)
        .getByTestId(TestIds.selectionToolbox.colorBlue)
        .click()
      await comfyPage.nodeOps.selectNodes(['KSampler'])

      // Select second node and color it differently
      await comfyPage.nodeOps.selectNodes(['CLIP Text Encode (Prompt)'])
      await getColorPickerButton(comfyPage).click()
      await getColorPickerGroup(comfyPage)
        .getByTestId(TestIds.selectionToolbox.colorRed)
        .click()

      // Select both nodes
      await comfyPage.nodeOps.selectNodes([
        'KSampler',
        'CLIP Text Encode (Prompt)'
      ])

      // Color picker should show null/mixed state
      const colorPickerButton = getColorPickerButton(comfyPage)
      await expect(colorPickerButton).not.toHaveAttribute('color')
    })

    test('color picker shows correct color when selecting pre-colored node', async ({
      comfyPage
    }) => {
      // First color a node
      await comfyPage.nodeOps.selectNodes(['KSampler'])
      await getColorPickerButton(comfyPage).click()
      await getColorPickerGroup(comfyPage)
        .getByTestId(TestIds.selectionToolbox.colorBlue)
        .click()

      // Clear selection
      await comfyPage.nodeOps.selectNodes(['KSampler'])

      // Re-select the node
      await comfyPage.nodeOps.selectNodes(['KSampler'])

      // Color picker button should show the correct color
      const colorPickerCurrentColor = getColorPickerCurrentColor(comfyPage)
      await expect(colorPickerCurrentColor).toHaveCSS('color', BLUE_COLOR)
    })

    test('colorization via color picker can be undone', async ({
      comfyPage
    }) => {
      // Select a node and color it
      await comfyPage.nodeOps.selectNodes(['KSampler'])
      await getColorPickerButton(comfyPage).click()
      await getColorPickerGroup(comfyPage)
        .getByTestId(TestIds.selectionToolbox.colorBlue)
        .click()

      // Undo the colorization
      await comfyPage.page.keyboard.press('Control+Z')
      await comfyPage.nextFrame()

      // Node should be uncolored again
      const selectedNode = (
        await comfyPage.nodeOps.getNodeRefsByTitle('KSampler')
      )[0]
      expect(await selectedNode.getProperty('color')).toBeUndefined()
    })
  })
})
