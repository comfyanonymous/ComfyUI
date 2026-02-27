import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'

test.describe('Menu', { tag: '@ui' }, () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')
  })

  test('Can register sidebar tab', async ({ comfyPage }) => {
    const initialChildrenCount = await comfyPage.menu.buttons.count()

    await comfyPage.page.evaluate(async () => {
      window.app!.extensionManager.registerSidebarTab({
        id: 'search',
        icon: 'pi pi-search',
        title: 'search',
        tooltip: 'search',
        type: 'custom',
        render: (el) => {
          el.innerHTML = '<div>Custom search tab</div>'
        }
      })
    })
    await comfyPage.nextFrame()

    const newChildrenCount = await comfyPage.menu.buttons.count()
    expect(newChildrenCount).toBe(initialChildrenCount + 1)
  })

  test.describe('Workflows topbar tabs', () => {
    test.beforeEach(async ({ comfyPage }) => {
      await comfyPage.settings.setSetting(
        'Comfy.Workflow.WorkflowTabsPosition',
        'Topbar'
      )
      await comfyPage.workflow.setupWorkflowsDirectory({})
    })

    test('Can show opened workflows', async ({ comfyPage }) => {
      expect(await comfyPage.menu.topbar.getTabNames()).toEqual([
        'Unsaved Workflow'
      ])
    })

    test('Can close saved-workflow tabs', async ({ comfyPage }) => {
      const workflowName = `tempWorkflow-${test.info().title}`
      await comfyPage.menu.topbar.saveWorkflow(workflowName)
      expect(await comfyPage.menu.topbar.getTabNames()).toEqual([workflowName])
      await comfyPage.menu.topbar.closeWorkflowTab(workflowName)
      await comfyPage.nextFrame()
      expect(await comfyPage.menu.topbar.getTabNames()).toEqual([
        'Unsaved Workflow'
      ])
    })
  })

  test.describe('Topbar submmenus', () => {
    test('@mobile Items fully visible on mobile screen width', async ({
      comfyPage
    }) => {
      await comfyPage.menu.topbar.openTopbarMenu()
      const topLevelMenuItem = comfyPage.page
        .locator('a.p-menubar-item-link')
        .first()
      const isTextCutoff = await topLevelMenuItem.evaluate((el) => {
        return el.scrollWidth > el.clientWidth
      })
      expect(isTextCutoff).toBe(false)
    })

    test('Clicking on active state items does not close menu', async ({
      comfyPage
    }) => {
      // Open the menu
      await comfyPage.menu.topbar.openTopbarMenu()
      const menu = comfyPage.page.locator('.comfy-command-menu')

      // Navigate to View menu
      const viewMenuItem = comfyPage.page.locator(
        '.p-menubar-item-label:text-is("View")'
      )
      await viewMenuItem.hover()

      // Wait for submenu to appear
      const viewSubmenu = comfyPage.page
        .locator('.p-tieredmenu-submenu:visible')
        .first()
      await viewSubmenu.waitFor({ state: 'visible' })

      // Find Bottom Panel menu item
      const bottomPanelMenuItem = viewSubmenu
        .locator('.p-tieredmenu-item:has-text("Bottom Panel")')
        .first()
      const bottomPanelItem = bottomPanelMenuItem.locator(
        '.p-menubar-item-label:text-is("Bottom Panel")'
      )
      await bottomPanelItem.waitFor({ state: 'visible' })

      // Get checkmark icon element
      const checkmark = bottomPanelMenuItem.locator('.pi-check')

      // Check initial state of bottom panel (it's initially hidden)
      const { bottomPanel } = comfyPage
      await expect(bottomPanel.root).not.toBeVisible()

      // Checkmark should be invisible initially (panel is hidden)
      await expect(checkmark).toHaveClass(/invisible/)

      await bottomPanelItem.click()

      // Verify menu is still visible after clicking
      await expect(menu).toBeVisible()
      await expect(viewSubmenu).toBeVisible()

      // Verify bottom panel is now visible
      await expect(bottomPanel.root).toBeVisible()

      // Checkmark should now be visible (panel is shown)
      await expect(checkmark).not.toHaveClass(/invisible/)

      // Click Bottom Panel again to toggle it off
      await bottomPanelItem.click()

      // Verify menu is still visible after second click
      await expect(menu).toBeVisible()
      await expect(viewSubmenu).toBeVisible()

      // Verify bottom panel is hidden again
      await expect(bottomPanel.root).not.toBeVisible()

      // Checkmark should be invisible again (panel is hidden)
      await expect(checkmark).toHaveClass(/invisible/)

      // Click in top-right corner to close menu (avoid hamburger menu at top-left)
      const viewport = comfyPage.page.viewportSize()!
      await comfyPage.page
        .locator('body')
        .click({ position: { x: viewport.width - 10, y: 10 } })

      // Verify menu is now closed
      await expect(menu).not.toBeVisible()
    })

    test('Displays keybinding next to item', async ({ comfyPage }) => {
      await comfyPage.menu.topbar.openTopbarMenu()
      const workflowMenuItem = comfyPage.menu.topbar.getMenuItem('File')
      await workflowMenuItem.hover()
      const exportTag = comfyPage.page.locator('.keybinding-tag', {
        hasText: 'Ctrl + s'
      })
      expect(await exportTag.count()).toBe(1)
    })

    test('Can catch error when executing command', async ({ comfyPage }) => {
      await comfyPage.page.evaluate(() => {
        window.app!.registerExtension({
          name: 'TestExtension1',
          commands: [
            {
              id: 'foo',
              label: 'foo-command',
              function: () => {
                throw new Error('foo!')
              }
            }
          ],
          menuCommands: [
            {
              path: ['ext'],
              commands: ['foo']
            }
          ]
        })
      })
      await comfyPage.menu.topbar.triggerTopbarCommand(['ext', 'foo-command'])
      await expect.poll(() => comfyPage.toast.getVisibleToastCount()).toBe(1)
    })

    test('Can navigate Theme menu and switch between Dark and Light themes', async ({
      comfyPage
    }) => {
      const { topbar } = comfyPage.menu

      // Take initial screenshot with default theme
      await comfyPage.attachScreenshot('theme-initial')

      // Open the topbar menu
      const menu = await topbar.openTopbarMenu()
      await expect(menu).toBeVisible()

      // Get theme menu items
      const {
        submenu: themeSubmenu,
        darkTheme: darkThemeItem,
        lightTheme: lightThemeItem
      } = await topbar.getThemeMenuItems()

      await expect(darkThemeItem).toBeVisible()
      await expect(lightThemeItem).toBeVisible()

      // Switch to Light theme
      await topbar.switchTheme('light')

      // Verify menu stays open and Light theme shows as active
      await expect(async () => {
        await expect(menu).toBeVisible()
        await expect(themeSubmenu).toBeVisible()
        expect(await topbar.isMenuItemActive(lightThemeItem)).toBe(true)
      }).toPass({ timeout: 5000 })

      // Screenshot with light theme active
      await comfyPage.attachScreenshot('theme-menu-light-active')

      // Verify ColorPalette setting is set to "light"
      await expect
        .poll(() => comfyPage.settings.getSetting('Comfy.ColorPalette'))
        .toBe('light')

      // Close menu to see theme change
      await topbar.closeTopbarMenu()

      // Re-open menu and get theme items again
      await topbar.openTopbarMenu()
      const themeItems2 = await topbar.getThemeMenuItems()

      // Switch back to Dark theme
      await topbar.switchTheme('dark')

      // Verify menu stays open and Dark theme shows as active
      await expect(async () => {
        await expect(menu).toBeVisible()
        await expect(themeItems2.submenu).toBeVisible()
        expect(await topbar.isMenuItemActive(themeItems2.darkTheme)).toBe(true)
        expect(await topbar.isMenuItemActive(themeItems2.lightTheme)).toBe(
          false
        )
      }).toPass({ timeout: 5000 })

      // Screenshot with dark theme active
      await comfyPage.attachScreenshot('theme-menu-dark-active')

      // Verify ColorPalette setting is set to "dark"
      await expect
        .poll(() => comfyPage.settings.getSetting('Comfy.ColorPalette'))
        .toBe('dark')

      // Close menu
      await topbar.closeTopbarMenu()
    })
  })

  // Only test 'Top' to reduce test time.
  // ['Bottom', 'Top']
  ;['Top'].forEach(async (position) => {
    test(`Can migrate deprecated menu positions (${position})`, async ({
      comfyPage
    }) => {
      await comfyPage.settings.setSetting('Comfy.UseNewMenu', position)
      expect(await comfyPage.settings.getSetting('Comfy.UseNewMenu')).toBe(
        'Top'
      )
    })

    test(`Can migrate deprecated menu positions on initial load (${position})`, async ({
      comfyPage
    }) => {
      await comfyPage.settings.setSetting('Comfy.UseNewMenu', position)
      await comfyPage.setup()
      expect(await comfyPage.settings.getSetting('Comfy.UseNewMenu')).toBe(
        'Top'
      )
    })
  })
})
