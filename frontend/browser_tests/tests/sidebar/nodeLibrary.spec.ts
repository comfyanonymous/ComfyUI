import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../../fixtures/ComfyPage'

test.describe('Node library sidebar', () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')
    await comfyPage.settings.setSetting('Comfy.NodeLibrary.NewDesign', false)
    await comfyPage.settings.setSetting('Comfy.NodeLibrary.Bookmarks.V2', [])
    await comfyPage.settings.setSetting(
      'Comfy.NodeLibrary.BookmarksCustomization',
      {}
    )
    // Open the sidebar
    const tab = comfyPage.menu.nodeLibraryTab
    await tab.open()
  })

  test('Node preview and drag to canvas', async ({ comfyPage }) => {
    const tab = comfyPage.menu.nodeLibraryTab
    await tab.getFolder('sampling').click()

    // Hover over a node to display the preview
    const nodeSelector = '.p-tree-node-leaf'
    await comfyPage.page.hover(nodeSelector)

    // Verify the preview is displayed
    const previewVisible = await comfyPage.page.isVisible(
      '.node-lib-node-preview'
    )
    expect(previewVisible).toBe(true)

    const count = await comfyPage.nodeOps.getGraphNodesCount()
    // Drag the node onto the canvas
    const canvasSelector = '#graph-canvas'

    // Get the bounding box of the canvas element
    const canvasBoundingBox = (await comfyPage.page
      .locator(canvasSelector)
      .boundingBox())!

    // Calculate the center position of the canvas
    const targetPosition = {
      x: canvasBoundingBox.x + canvasBoundingBox.width / 2,
      y: canvasBoundingBox.y + canvasBoundingBox.height / 2
    }

    await comfyPage.page.dragAndDrop(nodeSelector, canvasSelector, {
      targetPosition
    })

    // Verify the node is added to the canvas
    expect(await comfyPage.nodeOps.getGraphNodesCount()).toBe(count + 1)
  })

  test('Bookmark node', async ({ comfyPage }) => {
    const tab = comfyPage.menu.nodeLibraryTab
    await tab.getFolder('sampling').click()

    // Bookmark the node
    await tab.getNode('KSampler (Advanced)').locator('.bookmark-button').click()

    // Verify the bookmark is added to the bookmarks tab
    expect(
      await comfyPage.settings.getSetting('Comfy.NodeLibrary.Bookmarks.V2')
    ).toEqual(['KSamplerAdvanced'])
    // Verify the bookmark node with the same name is added to the tree.
    expect(await tab.getNode('KSampler (Advanced)').count()).toBe(2)

    // Hover on the bookmark node to display the preview
    await comfyPage.page.hover('.node-lib-bookmark-tree-explorer .tree-leaf')
    expect(await comfyPage.page.isVisible('.node-lib-node-preview')).toBe(true)
  })

  test('Ignores unrecognized node', async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.NodeLibrary.Bookmarks.V2', [
      'foo'
    ])

    const tab = comfyPage.menu.nodeLibraryTab
    expect(await tab.getFolder('sampling').count()).toBe(1)
    expect(await tab.getNode('foo').count()).toBe(0)
  })

  test('Displays empty bookmarks folder', async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.NodeLibrary.Bookmarks.V2', [
      'foo/'
    ])
    const tab = comfyPage.menu.nodeLibraryTab
    expect(await tab.getFolder('foo').count()).toBe(1)
  })

  test('Can add new bookmark folder', async ({ comfyPage }) => {
    const tab = comfyPage.menu.nodeLibraryTab
    await tab.newFolderButton.click()
    const textInput = comfyPage.page.locator('.editable-text input')
    await textInput.waitFor({ state: 'visible' })
    await textInput.fill('New Folder')
    await textInput.press('Enter')
    expect(await tab.getFolder('New Folder').count()).toBe(1)
    expect(
      await comfyPage.settings.getSetting('Comfy.NodeLibrary.Bookmarks.V2')
    ).toEqual(['New Folder/'])
  })

  test('Can add nested bookmark folder', async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.NodeLibrary.Bookmarks.V2', [
      'foo/'
    ])
    const tab = comfyPage.menu.nodeLibraryTab

    await tab.getFolder('foo').click({ button: 'right' })
    await comfyPage.page.getByRole('menuitem', { name: 'New Folder' }).click()
    const textInput = comfyPage.page.locator('.editable-text input')
    await textInput.waitFor({ state: 'visible' })
    await textInput.fill('bar')
    await textInput.press('Enter')

    expect(await tab.getFolder('bar').count()).toBe(1)
    expect(
      await comfyPage.settings.getSetting('Comfy.NodeLibrary.Bookmarks.V2')
    ).toEqual(['foo/', 'foo/bar/'])
  })

  test('Can delete bookmark folder', async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.NodeLibrary.Bookmarks.V2', [
      'foo/'
    ])
    const tab = comfyPage.menu.nodeLibraryTab

    await tab.getFolder('foo').click({ button: 'right' })
    await comfyPage.page.getByLabel('Delete').click()

    expect(
      await comfyPage.settings.getSetting('Comfy.NodeLibrary.Bookmarks.V2')
    ).toEqual([])
  })

  test('Can rename bookmark folder', async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.NodeLibrary.Bookmarks.V2', [
      'foo/'
    ])
    const tab = comfyPage.menu.nodeLibraryTab

    await tab.getFolder('foo').click({ button: 'right' })
    await comfyPage.page
      .locator('.p-contextmenu-item-label:has-text("Rename")')
      .click()
    await comfyPage.page.keyboard.insertText('bar')
    await comfyPage.page.keyboard.press('Enter')

    expect(
      await comfyPage.settings.getSetting('Comfy.NodeLibrary.Bookmarks.V2')
    ).toEqual(['bar/'])
  })

  test('Can add bookmark by dragging node to bookmark folder', async ({
    comfyPage
  }) => {
    await comfyPage.settings.setSetting('Comfy.NodeLibrary.Bookmarks.V2', [
      'foo/'
    ])
    const tab = comfyPage.menu.nodeLibraryTab
    await tab.getFolder('sampling').click()
    await comfyPage.page.dragAndDrop(
      tab.nodeSelector('KSampler (Advanced)'),
      tab.folderSelector('foo')
    )
    expect(
      await comfyPage.settings.getSetting('Comfy.NodeLibrary.Bookmarks.V2')
    ).toEqual(['foo/', 'foo/KSamplerAdvanced'])
  })

  test('Can add bookmark by clicking bookmark button', async ({
    comfyPage
  }) => {
    const tab = comfyPage.menu.nodeLibraryTab
    await tab.getFolder('sampling').click()
    await tab.getNode('KSampler (Advanced)').locator('.bookmark-button').click()
    expect(
      await comfyPage.settings.getSetting('Comfy.NodeLibrary.Bookmarks.V2')
    ).toEqual(['KSamplerAdvanced'])
  })

  test('Can unbookmark node (Top level bookmark)', async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.NodeLibrary.Bookmarks.V2', [
      'KSamplerAdvanced'
    ])
    const tab = comfyPage.menu.nodeLibraryTab
    await tab.getNode('KSampler (Advanced)').locator('.bookmark-button').click()
    expect(
      await comfyPage.settings.getSetting('Comfy.NodeLibrary.Bookmarks.V2')
    ).toEqual([])
  })

  test('Can unbookmark node (Library node bookmark)', async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.NodeLibrary.Bookmarks.V2', [
      'KSamplerAdvanced'
    ])
    const tab = comfyPage.menu.nodeLibraryTab
    await tab.getFolder('sampling').click()
    await tab
      .getNodeInFolder('KSampler (Advanced)', 'sampling')
      .locator('.bookmark-button')
      .click()
    expect(
      await comfyPage.settings.getSetting('Comfy.NodeLibrary.Bookmarks.V2')
    ).toEqual([])
  })
  test('Can customize icon', async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.NodeLibrary.Bookmarks.V2', [
      'foo/'
    ])
    const tab = comfyPage.menu.nodeLibraryTab
    await tab.getFolder('foo').click({ button: 'right' })
    await comfyPage.page.getByLabel('Customize').click()
    const dialog = comfyPage.page.getByRole('dialog', {
      name: 'Customize Folder'
    })
    // Select Folder icon (2nd button in Icon group)
    const iconGroup = dialog.getByText('Icon').locator('..').getByRole('group')
    await iconGroup.getByRole('button').nth(1).click()
    // Select Blue color (2nd button in Color group)
    const colorGroup = dialog
      .getByText('Color')
      .locator('..')
      .getByRole('group')
    await colorGroup.getByRole('button').nth(1).click()
    await dialog.getByRole('button', { name: 'Confirm' }).click()
    await comfyPage.nextFrame()
    expect(
      await comfyPage.settings.getSetting(
        'Comfy.NodeLibrary.BookmarksCustomization'
      )
    ).toEqual({
      'foo/': {
        icon: 'pi-folder',
        color: '#007bff'
      }
    })
  })
  // If color is left as default, it should not be saved
  test('Can customize icon (default field)', async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.NodeLibrary.Bookmarks.V2', [
      'foo/'
    ])
    const tab = comfyPage.menu.nodeLibraryTab
    await tab.getFolder('foo').click({ button: 'right' })
    await comfyPage.page.getByLabel('Customize').click()
    const dialog = comfyPage.page.getByRole('dialog', {
      name: 'Customize Folder'
    })
    // Select Folder icon (2nd button in Icon group)
    const iconGroup = dialog.getByText('Icon').locator('..').getByRole('group')
    await iconGroup.getByRole('button').nth(1).click()
    await dialog.getByRole('button', { name: 'Confirm' }).click()
    await comfyPage.nextFrame()
    expect(
      await comfyPage.settings.getSetting(
        'Comfy.NodeLibrary.BookmarksCustomization'
      )
    ).toEqual({
      'foo/': {
        icon: 'pi-folder'
      }
    })
  })

  test('Can customize bookmark color after interacting with color options', async ({
    comfyPage
  }) => {
    // Open customization dialog
    await comfyPage.settings.setSetting('Comfy.NodeLibrary.Bookmarks.V2', [
      'foo/'
    ])
    const tab = comfyPage.menu.nodeLibraryTab
    await tab.getFolder('foo').click({ button: 'right' })
    await comfyPage.page.getByLabel('Customize').click()

    // Click a color option multiple times
    const customColorOption = comfyPage.page.locator(
      '.p-togglebutton-content > .pi-palette'
    )
    await customColorOption.click()
    await customColorOption.click()

    // Use the color picker
    await comfyPage.page
      .getByLabel('Customize Folder')
      .getByRole('textbox')
      .click()
    await comfyPage.page.locator('.p-colorpicker-color-background').click()

    // Finalize the customization
    const dialog = comfyPage.page.getByRole('dialog', {
      name: 'Customize Folder'
    })
    // Select Folder icon (2nd button in Icon group)
    const iconGroup = dialog.getByText('Icon').locator('..').getByRole('group')
    await iconGroup.getByRole('button').nth(1).click()
    await dialog.getByRole('button', { name: 'Confirm' }).click()
    await comfyPage.nextFrame()

    // Verify the color selection is saved
    const setting = await comfyPage.settings.getSetting<
      Record<string, { icon?: string; color?: string }>
    >('Comfy.NodeLibrary.BookmarksCustomization')
    await expect(setting).toHaveProperty(['foo/', 'color'])
    await expect(setting['foo/'].color).not.toBeNull()
    await expect(setting['foo/'].color).not.toBeUndefined()
    await expect(setting['foo/'].color).not.toBe('')
  })

  test('Can rename customized bookmark folder', async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.NodeLibrary.Bookmarks.V2', [
      'foo/'
    ])
    await comfyPage.settings.setSetting(
      'Comfy.NodeLibrary.BookmarksCustomization',
      {
        'foo/': {
          icon: 'pi-folder',
          color: '#007bff'
        }
      }
    )
    const tab = comfyPage.menu.nodeLibraryTab
    await tab.getFolder('foo').click({ button: 'right' })
    await comfyPage.page
      .locator('.p-contextmenu-item-label:has-text("Rename")')
      .click()
    await comfyPage.page.keyboard.insertText('bar')
    await comfyPage.page.keyboard.press('Enter')
    await comfyPage.nextFrame()
    await expect(async () => {
      expect(
        await comfyPage.settings.getSetting('Comfy.NodeLibrary.Bookmarks.V2')
      ).toEqual(['bar/'])
      expect(
        await comfyPage.settings.getSetting(
          'Comfy.NodeLibrary.BookmarksCustomization'
        )
      ).toEqual({
        'bar/': {
          icon: 'pi-folder',
          color: '#007bff'
        }
      })
    }).toPass({
      timeout: 2_000
    })
  })

  test('Can delete customized bookmark folder', async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.NodeLibrary.Bookmarks.V2', [
      'foo/'
    ])
    await comfyPage.settings.setSetting(
      'Comfy.NodeLibrary.BookmarksCustomization',
      {
        'foo/': {
          icon: 'pi-folder',
          color: '#007bff'
        }
      }
    )
    const tab = comfyPage.menu.nodeLibraryTab
    await tab.getFolder('foo').click({ button: 'right' })
    await comfyPage.page.getByLabel('Delete').click()
    await comfyPage.nextFrame()
    expect(
      await comfyPage.settings.getSetting('Comfy.NodeLibrary.Bookmarks.V2')
    ).toEqual([])
    expect(
      await comfyPage.settings.getSetting(
        'Comfy.NodeLibrary.BookmarksCustomization'
      )
    ).toEqual({})
  })

  test('Can filter nodes in both trees', async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.NodeLibrary.Bookmarks.V2', [
      'foo/',
      'foo/KSamplerAdvanced',
      'KSampler'
    ])

    const tab = comfyPage.menu.nodeLibraryTab
    await tab.nodeLibrarySearchBoxInput.fill('KSampler')
    await expect(tab.getNode('KSampler (Advanced)')).toHaveCount(2)
  })
})
