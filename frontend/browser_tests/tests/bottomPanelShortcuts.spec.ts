import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'

test.describe('Bottom Panel Shortcuts', { tag: '@ui' }, () => {
  test.beforeEach(async ({ comfyPage }) => {
    await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Top')
  })

  test('should toggle shortcuts panel visibility', async ({ comfyPage }) => {
    const { bottomPanel } = comfyPage

    await expect(bottomPanel.root).not.toBeVisible()
    await bottomPanel.keyboardShortcutsButton.click()
    await expect(bottomPanel.root).toBeVisible()
    await bottomPanel.keyboardShortcutsButton.click()
    await expect(bottomPanel.root).not.toBeVisible()
  })

  test('should display essentials shortcuts tab', async ({ comfyPage }) => {
    const { bottomPanel } = comfyPage

    await bottomPanel.keyboardShortcutsButton.click()

    await expect(bottomPanel.shortcuts.essentialsTab).toBeVisible()
    await expect(bottomPanel.shortcuts.essentialsTab).toHaveAttribute(
      'aria-selected',
      'true'
    )

    await expect(bottomPanel.shortcuts.subcategoryTitles.first()).toBeVisible()
    await expect(bottomPanel.shortcuts.keyBadges.first()).toBeVisible()

    await expect(
      comfyPage.page.getByRole('heading', { name: 'Workflow' })
    ).toBeVisible()
    await expect(
      comfyPage.page.getByRole('heading', { name: 'Node' })
    ).toBeVisible()
    await expect(
      comfyPage.page.getByRole('heading', { name: 'Queue' })
    ).toBeVisible()
  })

  test('should display view controls shortcuts tab', async ({ comfyPage }) => {
    const { bottomPanel } = comfyPage

    await bottomPanel.keyboardShortcutsButton.click()
    await bottomPanel.shortcuts.viewControlsTab.click()

    await expect(bottomPanel.shortcuts.viewControlsTab).toHaveAttribute(
      'aria-selected',
      'true'
    )

    await expect(bottomPanel.shortcuts.keyBadges.first()).toBeVisible()

    await expect(
      comfyPage.page.getByRole('heading', { name: 'View' })
    ).toBeVisible()
    await expect(
      comfyPage.page.getByRole('heading', { name: 'Panel Controls' })
    ).toBeVisible()
  })

  test('should switch between shortcuts tabs', async ({ comfyPage }) => {
    const { bottomPanel } = comfyPage

    await bottomPanel.keyboardShortcutsButton.click()

    await expect(bottomPanel.shortcuts.essentialsTab).toHaveAttribute(
      'aria-selected',
      'true'
    )

    await bottomPanel.shortcuts.viewControlsTab.click()

    await expect(bottomPanel.shortcuts.viewControlsTab).toHaveAttribute(
      'aria-selected',
      'true'
    )
    await expect(bottomPanel.shortcuts.essentialsTab).not.toHaveAttribute(
      'aria-selected',
      'true'
    )

    await bottomPanel.shortcuts.essentialsTab.click()

    await expect(bottomPanel.shortcuts.essentialsTab).toHaveAttribute(
      'aria-selected',
      'true'
    )
    await expect(bottomPanel.shortcuts.viewControlsTab).not.toHaveAttribute(
      'aria-selected',
      'true'
    )
  })

  test('should display formatted keyboard shortcuts', async ({ comfyPage }) => {
    const { bottomPanel } = comfyPage

    await bottomPanel.keyboardShortcutsButton.click()

    const keyBadges = bottomPanel.shortcuts.keyBadges
    await keyBadges.first().waitFor({ state: 'visible' })
    const count = await keyBadges.count()
    expect(count).toBeGreaterThanOrEqual(1)

    const badgeText = await keyBadges.allTextContents()
    const hasModifiers = badgeText.some((text) =>
      ['Ctrl', 'Cmd', 'Shift', 'Alt'].includes(text)
    )
    expect(hasModifiers).toBeTruthy()
  })

  test('should maintain panel state when switching between panels', async ({
    comfyPage
  }) => {
    const { bottomPanel } = comfyPage

    // Open shortcuts panel first
    await bottomPanel.keyboardShortcutsButton.click()
    await expect(bottomPanel.root).toBeVisible()
    await expect(
      comfyPage.page.locator('[id*="tab_shortcuts-essentials"]')
    ).toBeVisible()

    // Try to open terminal panel - may show terminal OR close shortcuts
    // depending on whether terminal tabs have loaded (async loading)
    await bottomPanel.toggleButton.click()

    // Check if terminal tabs loaded (Logs tab visible) or fell back to shortcuts toggle
    const logsTab = comfyPage.page.getByRole('tab', { name: /Logs/i })
    const hasTerminalTabs = await logsTab.isVisible().catch(() => false)

    if (hasTerminalTabs) {
      // Terminal panel is visible - verify we can switch back to shortcuts
      await expect(bottomPanel.root).toBeVisible()

      // Switch back to shortcuts
      await bottomPanel.keyboardShortcutsButton.click()

      // Should show shortcuts content again
      await expect(
        comfyPage.page.locator('[id*="tab_shortcuts-essentials"]')
      ).toBeVisible()
    } else {
      // Terminal tabs not loaded - button toggled shortcuts off, reopen for verification
      await bottomPanel.keyboardShortcutsButton.click()
      await expect(bottomPanel.root).toBeVisible()
      await expect(
        comfyPage.page.locator('[id*="tab_shortcuts-essentials"]')
      ).toBeVisible()
    }
  })

  test('should handle keyboard navigation', async ({ comfyPage }) => {
    const { bottomPanel } = comfyPage

    await bottomPanel.keyboardShortcutsButton.click()
    await bottomPanel.shortcuts.essentialsTab.focus()

    await comfyPage.page.keyboard.press('ArrowRight')

    await expect(bottomPanel.shortcuts.viewControlsTab).toBeFocused()

    await comfyPage.page.keyboard.press('Enter')

    await expect(bottomPanel.shortcuts.viewControlsTab).toHaveAttribute(
      'aria-selected',
      'true'
    )
  })

  test('should close panel by clicking shortcuts button again', async ({
    comfyPage
  }) => {
    const { bottomPanel } = comfyPage

    await bottomPanel.keyboardShortcutsButton.click()
    await expect(bottomPanel.root).toBeVisible()

    await bottomPanel.keyboardShortcutsButton.click()
    await expect(bottomPanel.root).not.toBeVisible()
  })

  test('should display shortcuts in organized columns', async ({
    comfyPage
  }) => {
    const { bottomPanel } = comfyPage

    await bottomPanel.keyboardShortcutsButton.click()

    await expect(
      comfyPage.page.locator('[data-testid="shortcuts-columns"]')
    ).toBeVisible()

    const subcategoryTitles = bottomPanel.shortcuts.subcategoryTitles
    const titleCount = await subcategoryTitles.count()
    expect(titleCount).toBeGreaterThanOrEqual(2)
  })

  test('should open shortcuts panel with Ctrl+Shift+K', async ({
    comfyPage
  }) => {
    const { bottomPanel } = comfyPage

    await expect(bottomPanel.root).not.toBeVisible()

    await comfyPage.page.keyboard.press('Control+Shift+KeyK')

    await expect(bottomPanel.root).toBeVisible()
    await expect(bottomPanel.shortcuts.essentialsTab).toHaveAttribute(
      'aria-selected',
      'true'
    )
  })

  test('should open settings dialog when clicking manage shortcuts button', async ({
    comfyPage
  }) => {
    const { bottomPanel } = comfyPage

    await bottomPanel.keyboardShortcutsButton.click()

    await expect(bottomPanel.shortcuts.manageButton).toBeVisible()
    await bottomPanel.shortcuts.manageButton.click()

    await expect(comfyPage.settingDialog.root).toBeVisible()
    await expect(comfyPage.settingDialog.category('Keybinding')).toBeVisible()
  })
})
