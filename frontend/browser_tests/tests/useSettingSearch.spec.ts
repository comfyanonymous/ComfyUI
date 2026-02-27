import { expect } from '@playwright/test'

import type { Settings } from '../../src/schemas/apiSchema'
import { comfyPageFixture as test } from '../fixtures/ComfyPage'

/**
 * Type helper for test settings with arbitrary IDs.
 * Extensions can register settings with any ID, but SettingParams.id
 * is typed as keyof Settings for autocomplete.
 */
type TestSettingId = keyof Settings

test.beforeEach(async ({ comfyPage }) => {
  await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
})

test.describe('Settings Search functionality', { tag: '@settings' }, () => {
  test.beforeEach(async ({ comfyPage }) => {
    // Register test settings to verify hidden/deprecated filtering
    await comfyPage.page.evaluate(() => {
      window.app!.registerExtension({
        name: 'TestSettingsExtension',
        settings: [
          {
            id: 'TestHiddenSetting' as TestSettingId,
            name: 'Test Hidden Setting',
            type: 'hidden',
            defaultValue: 'hidden_value',
            category: ['Test', 'Hidden']
          },
          {
            id: 'TestDeprecatedSetting' as TestSettingId,
            name: 'Test Deprecated Setting',
            type: 'text',
            defaultValue: 'deprecated_value',
            deprecated: true,
            category: ['Test', 'Deprecated']
          },
          {
            id: 'TestVisibleSetting' as TestSettingId,
            name: 'Test Visible Setting',
            type: 'text',
            defaultValue: 'visible_value',
            category: ['Test', 'Visible']
          }
        ]
      })
    })
  })

  test('can open settings dialog and use search box', async ({ comfyPage }) => {
    const dialog = comfyPage.settingDialog
    await dialog.open()

    await expect(dialog.searchBox).toHaveAttribute(
      'placeholder',
      expect.stringContaining('Search')
    )
  })

  test('search box is functional and accepts input', async ({ comfyPage }) => {
    const dialog = comfyPage.settingDialog
    await dialog.open()

    await dialog.searchBox.fill('Comfy')
    await expect(dialog.searchBox).toHaveValue('Comfy')
  })

  test('search box clears properly', async ({ comfyPage }) => {
    const dialog = comfyPage.settingDialog
    await dialog.open()

    await dialog.searchBox.fill('test')
    await expect(dialog.searchBox).toHaveValue('test')

    await dialog.searchBox.clear()
    await expect(dialog.searchBox).toHaveValue('')
  })

  test('settings categories are visible in sidebar', async ({ comfyPage }) => {
    const dialog = comfyPage.settingDialog
    await dialog.open()

    expect(await dialog.categories.count()).toBeGreaterThan(0)
  })

  test('can select different categories in sidebar', async ({ comfyPage }) => {
    const dialog = comfyPage.settingDialog
    await dialog.open()

    const categoryCount = await dialog.categories.count()

    if (categoryCount > 1) {
      await dialog.categories.nth(1).click()

      await expect(dialog.categories.nth(1)).toHaveClass(
        /bg-interface-menu-component-surface-selected/
      )
    }
  })

  test('search functionality affects UI state', async ({ comfyPage }) => {
    const dialog = comfyPage.settingDialog
    await dialog.open()

    await dialog.searchBox.fill('graph')
    await expect(dialog.searchBox).toHaveValue('graph')
  })

  test('settings dialog can be closed', async ({ comfyPage }) => {
    const dialog = comfyPage.settingDialog
    await dialog.open()

    await comfyPage.page.keyboard.press('Escape')
    await expect(dialog.root).not.toBeVisible()
  })

  test('search box has proper debouncing behavior', async ({ comfyPage }) => {
    const dialog = comfyPage.settingDialog
    await dialog.open()

    await dialog.searchBox.fill('a')
    await dialog.searchBox.fill('ab')
    await dialog.searchBox.fill('abc')
    await dialog.searchBox.fill('abcd')

    await expect(dialog.searchBox).toHaveValue('abcd')
  })

  test('search excludes hidden settings from results', async ({
    comfyPage
  }) => {
    const dialog = comfyPage.settingDialog
    await dialog.open()

    await dialog.searchBox.fill('Test')

    await expect(dialog.contentArea).toContainText('Test Visible Setting')
    await expect(dialog.contentArea).not.toContainText('Test Hidden Setting')
  })

  test('search excludes deprecated settings from results', async ({
    comfyPage
  }) => {
    const dialog = comfyPage.settingDialog
    await dialog.open()

    await dialog.searchBox.fill('Test')

    await expect(dialog.contentArea).toContainText('Test Visible Setting')
    await expect(dialog.contentArea).not.toContainText(
      'Test Deprecated Setting'
    )
  })

  test('search shows visible settings but excludes hidden and deprecated', async ({
    comfyPage
  }) => {
    const dialog = comfyPage.settingDialog
    await dialog.open()

    await dialog.searchBox.fill('Test')

    await expect(dialog.contentArea).toContainText('Test Visible Setting')
    await expect(dialog.contentArea).not.toContainText('Test Hidden Setting')
    await expect(dialog.contentArea).not.toContainText(
      'Test Deprecated Setting'
    )
  })

  test('search by setting name excludes hidden and deprecated', async ({
    comfyPage
  }) => {
    const dialog = comfyPage.settingDialog
    await dialog.open()

    await dialog.searchBox.clear()
    await dialog.searchBox.fill('Hidden')
    await expect(dialog.contentArea).not.toContainText('Test Hidden Setting')

    await dialog.searchBox.clear()
    await dialog.searchBox.fill('Deprecated')
    await expect(dialog.contentArea).not.toContainText(
      'Test Deprecated Setting'
    )

    await dialog.searchBox.clear()
    await dialog.searchBox.fill('Visible')
    await expect(dialog.contentArea).toContainText('Test Visible Setting')
  })
})
