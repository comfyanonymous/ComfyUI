import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'

test.beforeEach(async ({ comfyPage }) => {
  await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
})

test.describe('Keybindings', { tag: '@keyboard' }, () => {
  test('Should not trigger non-modifier keybinding when typing in input fields', async ({
    comfyPage
  }) => {
    await comfyPage.command.registerKeybinding({ key: 'k' }, () => {
      window.TestCommand = true
    })

    const textBox = comfyPage.widgetTextBox
    await textBox.click()
    await textBox.fill('k')
    await expect(textBox).toHaveValue('k')
    expect(await comfyPage.page.evaluate(() => window.TestCommand)).toBe(
      undefined
    )
  })

  test('Should not trigger modifier keybinding when typing in input fields', async ({
    comfyPage
  }) => {
    await comfyPage.command.registerKeybinding({ key: 'k', ctrl: true }, () => {
      window.TestCommand = true
    })

    const textBox = comfyPage.widgetTextBox
    await textBox.click()
    await textBox.fill('q')
    await textBox.press('Control+k')
    await expect(textBox).toHaveValue('q')
    expect(await comfyPage.page.evaluate(() => window.TestCommand)).toBe(true)
  })

  test('Should not trigger keybinding reserved by text input when typing in input fields', async ({
    comfyPage
  }) => {
    await comfyPage.command.registerKeybinding({ key: 'Ctrl+v' }, () => {
      window.TestCommand = true
    })

    const textBox = comfyPage.widgetTextBox
    await textBox.click()
    await textBox.press('Control+v')
    await expect(textBox).toBeFocused()
    expect(await comfyPage.page.evaluate(() => window.TestCommand)).toBe(
      undefined
    )
  })
})
