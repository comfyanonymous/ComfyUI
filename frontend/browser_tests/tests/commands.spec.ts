import { expect } from '@playwright/test'

import { comfyPageFixture as test } from '../fixtures/ComfyPage'

test.beforeEach(async ({ comfyPage }) => {
  await comfyPage.settings.setSetting('Comfy.UseNewMenu', 'Disabled')
})

test.describe('Keybindings', { tag: '@keyboard' }, () => {
  test('Should execute command', async ({ comfyPage }) => {
    await comfyPage.command.registerCommand('TestCommand', () => {
      window.foo = true
    })

    await comfyPage.command.executeCommand('TestCommand')
    expect(await comfyPage.page.evaluate(() => window.foo)).toBe(true)
  })

  test('Should execute async command', async ({ comfyPage }) => {
    await comfyPage.command.registerCommand('TestCommand', async () => {
      await new Promise<void>((resolve) =>
        setTimeout(() => {
          window.foo = true
          resolve()
        }, 5)
      )
    })

    await comfyPage.command.executeCommand('TestCommand')
    expect(await comfyPage.page.evaluate(() => window.foo)).toBe(true)
  })

  test('Should handle command errors', async ({ comfyPage }) => {
    await comfyPage.command.registerCommand('TestCommand', () => {
      throw new Error('Test error')
    })

    await comfyPage.command.executeCommand('TestCommand')
    expect(await comfyPage.toast.getToastErrorCount()).toBe(1)
  })

  test('Should handle async command errors', async ({ comfyPage }) => {
    await comfyPage.command.registerCommand('TestCommand', async () => {
      await new Promise<void>((_resolve, reject) =>
        setTimeout(() => {
          reject(new Error('Test error'))
        }, 5)
      )
    })

    await comfyPage.command.executeCommand('TestCommand')
    expect(await comfyPage.toast.getToastErrorCount()).toBe(1)
  })
})
