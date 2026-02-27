import {
  comfyExpect as expect,
  comfyPageFixture as test
} from '../fixtures/ComfyPage'

test.describe('Node search box V2', { tag: '@node' }, () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
    await comfyPage.settings.setSetting('Comfy.NodeSearchBoxImpl', 'default')
    await comfyPage.settings.setSetting(
      'Comfy.LinkRelease.Action',
      'search box'
    )
    await comfyPage.settings.setSetting(
      'Comfy.LinkRelease.ActionShift',
      'search box'
    )
    await comfyPage.searchBoxV2.reload(comfyPage)
  })

  test('Can open search and add node', async ({ comfyPage }) => {
    const { searchBoxV2 } = comfyPage
    const initialCount = await comfyPage.nodeOps.getGraphNodesCount()

    await comfyPage.canvasOps.doubleClick()
    await expect(searchBoxV2.input).toBeVisible()

    await searchBoxV2.input.fill('KSampler')
    await expect(searchBoxV2.results.first()).toBeVisible()

    await comfyPage.page.keyboard.press('Enter')
    await expect(searchBoxV2.input).not.toBeVisible()

    const newCount = await comfyPage.nodeOps.getGraphNodesCount()
    expect(newCount).toBe(initialCount + 1)
  })

  test('Can add first default result with Enter', async ({ comfyPage }) => {
    const { searchBoxV2 } = comfyPage
    const initialCount = await comfyPage.nodeOps.getGraphNodesCount()

    await comfyPage.canvasOps.doubleClick()
    await expect(searchBoxV2.input).toBeVisible()

    // Default results should be visible without typing
    await expect(searchBoxV2.results.first()).toBeVisible()

    // Enter should add the first (selected) result
    await comfyPage.page.keyboard.press('Enter')
    await expect(searchBoxV2.input).not.toBeVisible()

    const newCount = await comfyPage.nodeOps.getGraphNodesCount()
    expect(newCount).toBe(initialCount + 1)
  })

  test.describe('Category navigation', () => {
    test('Favorites shows only bookmarked nodes', async ({ comfyPage }) => {
      const { searchBoxV2 } = comfyPage
      await comfyPage.settings.setSetting('Comfy.NodeLibrary.Bookmarks.V2', [
        'KSampler'
      ])
      await searchBoxV2.reload(comfyPage)

      await comfyPage.canvasOps.doubleClick()
      await expect(searchBoxV2.input).toBeVisible()

      await searchBoxV2.categoryButton('favorites').click()

      await expect(searchBoxV2.results).toHaveCount(1)
      await expect(searchBoxV2.results.first()).toContainText('KSampler')
    })

    test('Category filters results to matching nodes', async ({
      comfyPage
    }) => {
      const { searchBoxV2 } = comfyPage

      await comfyPage.canvasOps.doubleClick()
      await expect(searchBoxV2.input).toBeVisible()

      await searchBoxV2.categoryButton('sampling').click()

      await expect(searchBoxV2.results.first()).toBeVisible()
      const count = await searchBoxV2.results.count()
      expect(count).toBeGreaterThan(0)
    })
  })

  test.describe('Filter workflow', () => {
    test('Can filter by input type via filter bar', async ({ comfyPage }) => {
      const { searchBoxV2 } = comfyPage

      await comfyPage.canvasOps.doubleClick()
      await expect(searchBoxV2.input).toBeVisible()

      // Click "Input" filter chip in the filter bar
      await searchBoxV2.filterBarButton('Input').click()

      // Filter options should appear
      await expect(searchBoxV2.filterOptions.first()).toBeVisible()

      // Type to narrow and select MODEL
      await searchBoxV2.input.fill('MODEL')
      await searchBoxV2.filterOptions
        .filter({ hasText: 'MODEL' })
        .first()
        .click()

      // Filter chip should appear and results should be filtered
      await expect(
        searchBoxV2.dialog.getByText('Input:', { exact: false }).locator('..')
      ).toContainText('MODEL')
      await expect(searchBoxV2.results.first()).toBeVisible()
    })
  })

  test.describe('Keyboard navigation', () => {
    test('Can navigate and select with keyboard', async ({ comfyPage }) => {
      const { searchBoxV2 } = comfyPage
      const initialCount = await comfyPage.nodeOps.getGraphNodesCount()

      await comfyPage.canvasOps.doubleClick()
      await expect(searchBoxV2.input).toBeVisible()

      await searchBoxV2.input.fill('KSampler')
      const results = searchBoxV2.results
      await expect(results.first()).toBeVisible()

      // First result selected by default
      await expect(results.first()).toHaveAttribute('aria-selected', 'true')

      // ArrowDown moves selection
      await comfyPage.page.keyboard.press('ArrowDown')
      await expect(results.nth(1)).toHaveAttribute('aria-selected', 'true')
      await expect(results.first()).toHaveAttribute('aria-selected', 'false')

      // ArrowUp moves back
      await comfyPage.page.keyboard.press('ArrowUp')
      await expect(results.first()).toHaveAttribute('aria-selected', 'true')

      // Enter selects and adds node
      await comfyPage.page.keyboard.press('Enter')
      await expect(searchBoxV2.input).not.toBeVisible()

      const newCount = await comfyPage.nodeOps.getGraphNodesCount()
      expect(newCount).toBe(initialCount + 1)
    })
  })
})
