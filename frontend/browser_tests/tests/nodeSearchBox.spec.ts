import {
  comfyExpect as expect,
  comfyPageFixture as test
} from '../fixtures/ComfyPage'
import type { ComfyPage } from '../fixtures/ComfyPage'

test.beforeEach(async ({ comfyPage }) => {
  await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
})

test.describe('Node search box', { tag: '@node' }, () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting(
      'Comfy.LinkRelease.Action',
      'search box'
    )
    await comfyPage.settings.setSetting(
      'Comfy.LinkRelease.ActionShift',
      'search box'
    )
    await comfyPage.settings.setSetting(
      'Comfy.NodeSearchBoxImpl',
      'v1 (legacy)'
    )
  })

  test(`Can trigger on empty canvas double click`, async ({ comfyPage }) => {
    await comfyPage.canvasOps.doubleClick()
    await expect(comfyPage.searchBox.input).toHaveCount(1)
  })

  test(`Can trigger on group body double click`, async ({ comfyPage }) => {
    await comfyPage.workflow.loadWorkflow('groups/single_group_only')
    await comfyPage.page.mouse.dblclick(50, 50, { delay: 5 })
    await comfyPage.nextFrame()
    await expect(comfyPage.searchBox.input).toHaveCount(1)
  })

  test('Can trigger on link release', async ({ comfyPage }) => {
    await comfyPage.canvasOps.disconnectEdge()
    await expect(comfyPage.searchBox.input).toHaveCount(1)
  })

  test('New user (1.24.1+) gets search box by default on link release', async ({
    comfyPage
  }) => {
    // Start fresh to test new user behavior
    await comfyPage.setup({ clearStorage: true })
    // Simulate new user with 1.24.1+ installed version
    await comfyPage.settings.setSetting('Comfy.InstalledVersion', '1.24.1')
    await comfyPage.settings.setSetting(
      'Comfy.NodeSearchBoxImpl',
      'v1 (legacy)'
    )
    // Don't set LinkRelease settings explicitly to test versioned defaults

    await comfyPage.canvasOps.disconnectEdge()
    await expect(comfyPage.searchBox.input).toHaveCount(1)
    await expect(comfyPage.searchBox.input).toBeVisible()
  })

  test('Can add node', { tag: '@screenshot' }, async ({ comfyPage }) => {
    await comfyPage.canvasOps.doubleClick()
    await expect(comfyPage.searchBox.input).toHaveCount(1)
    await comfyPage.searchBox.fillAndSelectFirstNode('KSampler')
    await expect(comfyPage.canvas).toHaveScreenshot('added-node.png')
  })

  test('Can auto link node', { tag: '@screenshot' }, async ({ comfyPage }) => {
    await comfyPage.canvasOps.disconnectEdge()
    // Select the second item as the first item is always reroute
    await comfyPage.searchBox.fillAndSelectFirstNode('CLIPTextEncode', {
      suggestionIndex: 0
    })
    await expect(comfyPage.canvas).toHaveScreenshot('auto-linked-node.png')
  })

  test(
    'Can auto link batch moved node',
    { tag: '@screenshot' },
    async ({ comfyPage }) => {
      await comfyPage.workflow.loadWorkflow('links/batch_move_links')

      // Get the CLIP output slot (index 1) from the first CheckpointLoaderSimple node (id: 4)
      const checkpointNode = await comfyPage.nodeOps.getNodeRefById(4)
      const clipOutputSlot = await checkpointNode.getOutput(1)
      const outputSlotPos = await clipOutputSlot.getPosition()

      // Use a position in the empty canvas area (top-left corner)
      const emptySpacePos = { x: 5, y: 5 }

      await comfyPage.page.keyboard.down('Shift')
      await comfyPage.canvasOps.dragAndDrop(outputSlotPos, emptySpacePos)
      await comfyPage.page.keyboard.up('Shift')

      // Select the second item as the first item is always reroute
      await comfyPage.searchBox.fillAndSelectFirstNode('Load Checkpoint', {
        suggestionIndex: 0
      })
      await expect(comfyPage.canvas).toHaveScreenshot(
        'auto-linked-node-batch.png'
      )
    }
  )

  test(
    'Link release connecting to node with no slots',
    { tag: '@screenshot' },
    async ({ comfyPage }) => {
      await comfyPage.canvasOps.disconnectEdge()
      await expect(comfyPage.searchBox.input).toHaveCount(1)
      await comfyPage.page.locator('.p-chip-remove-icon').click()
      await comfyPage.searchBox.fillAndSelectFirstNode('KSampler')
      await expect(comfyPage.canvas).toHaveScreenshot(
        'added-node-no-connection.png'
      )
    }
  )

  test('Has correct aria-labels on search results', async ({ comfyPage }) => {
    const node = 'Load Checkpoint'
    await comfyPage.canvasOps.doubleClick()
    await comfyPage.searchBox.input.waitFor({ state: 'visible' })
    await comfyPage.searchBox.input.fill(node)
    await comfyPage.searchBox.dropdown.waitFor({ state: 'visible' })

    const firstResult = comfyPage.searchBox.dropdown.locator('li').first()
    await expect(firstResult).toHaveAttribute('aria-label', node)
  })

  test('@mobile Can trigger on empty canvas tap', async ({ comfyPage }) => {
    await comfyPage.closeMenu()
    await comfyPage.workflow.loadWorkflow('nodes/single_ksampler')
    const screenCenter = {
      x: 200,
      y: 400
    }
    await comfyPage.canvas.tap({
      position: screenCenter
    })
    await comfyPage.canvas.tap({
      position: screenCenter
    })
    await expect(comfyPage.searchBox.input).not.toHaveCount(0)
  })

  test.describe('Filtering', () => {
    const expectFilterChips = async (
      comfyPage: ComfyPage,
      expectedTexts: string[]
    ) => {
      const chips = comfyPage.searchBox.filterChips

      // Check that the number of chips matches the expected count
      await expect(chips).toHaveCount(expectedTexts.length)

      // Verify the text and visibility of each filter chip
      await Promise.all(
        expectedTexts.map(async (text, index) => {
          const chip = chips.nth(index)
          await expect(chip).toContainText(text)
          await expect(chip).toBeVisible()
        })
      )
    }

    test.beforeEach(async ({ comfyPage }) => {
      await comfyPage.canvasOps.doubleClick()
    })

    test('Can add filter', async ({ comfyPage }) => {
      await comfyPage.searchBox.addFilter('MODEL', 'Input Type')
      await expectFilterChips(comfyPage, ['MODEL'])
    })

    // Flaky test.
    // Sample test failure:
    // https://github.com/Comfy-Org/ComfyUI_frontend/actions/runs/12696912248/job/35391990861?pr=2210
    /*
    1) [chromium-2x] › nodeSearchBox.spec.ts:135:5 › Node search box › Filtering › Outer click dismisses filter panel but keeps search box visible

    Error: expect(locator).not.toBeVisible()

    Locator: getByRole('dialog').locator('div').filter({ hasText: 'Add node filter condition' })
    Expected: not visible
    Received: visible
    Call log:
      - expect.not.toBeVisible with timeout 5000ms
      - waiting for getByRole('dialog').locator('div').filter({ hasText: 'Add node filter condition' })


      143 |
      144 |       // Verify the filter selection panel is hidden
    > 145 |       expect(panel.header).not.toBeVisible()
          |                                ^
      146 |
      147 |       // Verify the node search dialog is still visible
      148 |       expect(comfyPage.searchBox.input).toBeVisible()

        at /home/runner/work/ComfyUI_frontend/ComfyUI_frontend/ComfyUI_frontend/browser_tests/nodeSearchBox.spec.ts:145:32
     */
    test.skip('Outer click dismisses filter panel but keeps search box visible', async ({
      comfyPage
    }) => {
      await comfyPage.searchBox.filterButton.click()
      const panel = comfyPage.searchBox.filterSelectionPanel
      await panel.header.waitFor({ state: 'visible' })
      const panelBounds = await panel.header.boundingBox()
      await comfyPage.page.mouse.click(panelBounds!.x - 10, panelBounds!.y - 10)

      // Verify the filter selection panel is hidden
      await expect(panel.header).not.toBeVisible()

      // Verify the node search dialog is still visible
      await expect(comfyPage.searchBox.input).toBeVisible()
    })

    test('Can add multiple filters', async ({ comfyPage }) => {
      await comfyPage.searchBox.addFilter('MODEL', 'Input Type')
      await comfyPage.searchBox.addFilter('CLIP', 'Output Type')
      await expectFilterChips(comfyPage, ['MODEL', 'CLIP'])
    })

    test('Does not add duplicate filter with same type and value', async ({
      comfyPage
    }) => {
      await comfyPage.searchBox.addFilter('MODEL', 'Input Type')
      await comfyPage.searchBox.addFilter('MODEL', 'Input Type')
      await expectFilterChips(comfyPage, ['MODEL'])
    })

    test('Can remove filter', async ({ comfyPage }) => {
      await comfyPage.searchBox.addFilter('MODEL', 'Input Type')
      await comfyPage.searchBox.removeFilter(0)
      await expectFilterChips(comfyPage, [])
    })

    test.describe('Removing from multiple filters', () => {
      test.beforeEach(async ({ comfyPage }) => {
        await comfyPage.searchBox.addFilter('MODEL', 'Input Type')
        await comfyPage.searchBox.addFilter('CLIP', 'Output Type')
        await comfyPage.searchBox.addFilter('utils', 'Category')
      })

      test('Can remove first filter', async ({ comfyPage }) => {
        await comfyPage.searchBox.removeFilter(0)
        await expectFilterChips(comfyPage, ['CLIP', 'utils'])
        await comfyPage.searchBox.removeFilter(0)
        await expectFilterChips(comfyPage, ['utils'])
        await comfyPage.searchBox.removeFilter(0)
        await expectFilterChips(comfyPage, [])
      })

      test('Can remove middle filter', async ({ comfyPage }) => {
        await comfyPage.searchBox.removeFilter(1)
        await expectFilterChips(comfyPage, ['MODEL', 'utils'])
      })

      test('Can remove last filter', async ({ comfyPage }) => {
        await comfyPage.searchBox.removeFilter(2)
        await expectFilterChips(comfyPage, ['MODEL', 'CLIP'])
      })
    })
  })

  test.describe('Input focus behavior', () => {
    test.beforeEach(async ({ comfyPage }) => {
      await comfyPage.canvasOps.doubleClick()
    })

    test('focuses input after adding a filter', async ({ comfyPage }) => {
      await comfyPage.searchBox.addFilter('MODEL', 'Input Type')
      await expect(comfyPage.searchBox.input).toHaveFocus()
    })

    test('focuses input after removing a filter', async ({ comfyPage }) => {
      await comfyPage.searchBox.addFilter('MODEL', 'Input Type')
      await comfyPage.searchBox.removeFilter(0)
      await expect(comfyPage.searchBox.input).toHaveFocus()
    })
  })
})

test.describe('Release context menu', { tag: '@node' }, () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting(
      'Comfy.LinkRelease.Action',
      'context menu'
    )
    await comfyPage.settings.setSetting(
      'Comfy.LinkRelease.ActionShift',
      'search box'
    )
    await comfyPage.settings.setSetting(
      'Comfy.NodeSearchBoxImpl',
      'v1 (legacy)'
    )
  })

  test(
    'Can trigger on link release',
    { tag: '@screenshot' },
    async ({ comfyPage }) => {
      await comfyPage.canvasOps.disconnectEdge()
      const contextMenu = comfyPage.page.locator('.litecontextmenu')
      // Wait for context menu with correct title (slot name | slot type)
      // The title shows the output slot name and type from the disconnected link
      await expect(contextMenu.locator('.litemenu-title')).toContainText(
        'CLIP | CLIP'
      )
      await comfyPage.page.mouse.move(10, 10)
      await comfyPage.nextFrame()
      await expect(comfyPage.canvas).toHaveScreenshot(
        'link-release-context-menu.png'
      )
    }
  )

  test(
    'Can search and add node from context menu',
    { tag: '@screenshot' },
    async ({ comfyPage, comfyMouse }) => {
      await comfyPage.canvasOps.disconnectEdge()
      await comfyMouse.move({ x: 10, y: 10 })
      await comfyPage.contextMenu.clickMenuItem('Search')
      await comfyPage.nextFrame()
      await comfyPage.searchBox.fillAndSelectFirstNode('CLIP Prompt')
      await expect(comfyPage.canvas).toHaveScreenshot(
        'link-context-menu-search.png'
      )
    }
  )

  test('Existing user (pre-1.24.1) gets context menu by default on link release', async ({
    comfyPage
  }) => {
    // Start fresh to test existing user behavior
    await comfyPage.setup({ clearStorage: true })
    // Simulate existing user with pre-1.24.1 version
    await comfyPage.settings.setSetting('Comfy.InstalledVersion', '1.23.0')
    await comfyPage.settings.setSetting(
      'Comfy.NodeSearchBoxImpl',
      'v1 (legacy)'
    )
    // Don't set LinkRelease settings explicitly to test versioned defaults

    await comfyPage.canvasOps.disconnectEdge()
    // Context menu should appear, search box should not
    await expect(comfyPage.searchBox.input).toHaveCount(0)
    const contextMenu = comfyPage.page.locator('.litecontextmenu')
    await expect(contextMenu).toBeVisible()
  })

  test('Explicit setting overrides versioned defaults', async ({
    comfyPage
  }) => {
    // Start fresh and simulate new user who should get search box by default
    await comfyPage.setup({ clearStorage: true })
    await comfyPage.settings.setSetting('Comfy.InstalledVersion', '1.24.1')
    // But explicitly set to context menu (overriding versioned default)
    await comfyPage.settings.setSetting(
      'Comfy.LinkRelease.Action',
      'context menu'
    )
    await comfyPage.settings.setSetting(
      'Comfy.NodeSearchBoxImpl',
      'v1 (legacy)'
    )

    await comfyPage.canvasOps.disconnectEdge()
    // Context menu should appear due to explicit setting, not search box
    await expect(comfyPage.searchBox.input).toHaveCount(0)
    const contextMenu = comfyPage.page.locator('.litecontextmenu')
    await expect(contextMenu).toBeVisible()
  })
})
