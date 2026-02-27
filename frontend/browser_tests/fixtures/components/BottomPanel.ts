import type { Locator, Page } from '@playwright/test'

class ShortcutsTab {
  readonly essentialsTab: Locator
  readonly viewControlsTab: Locator
  readonly manageButton: Locator
  readonly keyBadges: Locator
  readonly subcategoryTitles: Locator

  constructor(readonly page: Page) {
    this.essentialsTab = page.getByRole('tab', { name: /Essential/i })
    this.viewControlsTab = page.getByRole('tab', { name: /View Controls/i })
    this.manageButton = page.getByRole('button', { name: /Manage Shortcuts/i })
    this.keyBadges = page.locator('.key-badge')
    this.subcategoryTitles = page.locator('.subcategory-title')
  }
}

export class BottomPanel {
  readonly root: Locator
  readonly keyboardShortcutsButton: Locator
  readonly toggleButton: Locator
  readonly shortcuts: ShortcutsTab

  constructor(readonly page: Page) {
    this.root = page.locator('.bottom-panel')
    this.keyboardShortcutsButton = page.getByRole('button', {
      name: /Keyboard Shortcuts/i
    })
    this.toggleButton = page.getByRole('button', {
      name: /Toggle Bottom Panel/i
    })
    this.shortcuts = new ShortcutsTab(page)
  }
}
