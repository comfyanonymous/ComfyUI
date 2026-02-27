import type { Page } from '@playwright/test'

import type { KeyCombo } from '../../../src/platform/keybindings/types'

export class CommandHelper {
  constructor(private readonly page: Page) {}

  async executeCommand(
    commandId: string,
    metadata?: Record<string, unknown>
  ): Promise<void> {
    await this.page.evaluate(
      ({ commandId, metadata }) => {
        const app = window.app
        if (!app) throw new Error('window.app is not available')

        return app.extensionManager.command.execute(commandId, {
          metadata
        })
      },
      { commandId, metadata }
    )
  }

  async registerCommand(
    commandId: string,
    command: (() => void) | (() => Promise<void>)
  ): Promise<void> {
    // SECURITY: eval() is intentionally used here to deserialize/execute functions
    // passed from controlled test code across the Node/Playwright browser boundary.
    // Execution happens in isolated Playwright browser contexts with test-only data.
    // This pattern is unsafe for production and must not be copied elsewhere.
    await this.page.evaluate(
      ({ commandId, commandStr }) => {
        const app = window.app!
        const randomSuffix = Math.random().toString(36).substring(2, 8)
        const extensionName = `TestExtension_${randomSuffix}`

        app.registerExtension({
          name: extensionName,
          commands: [
            {
              id: commandId,
              function: eval(commandStr)
            }
          ]
        })
      },
      { commandId, commandStr: command.toString() }
    )
  }

  async registerKeybinding(
    keyCombo: KeyCombo,
    command: () => void
  ): Promise<void> {
    // SECURITY: eval() is intentionally used here to deserialize/execute functions
    // passed from controlled test code across the Node/Playwright browser boundary.
    // Execution happens in isolated Playwright browser contexts with test-only data.
    // This pattern is unsafe for production and must not be copied elsewhere.
    await this.page.evaluate(
      ({ keyCombo, commandStr }) => {
        const app = window.app!
        const randomSuffix = Math.random().toString(36).substring(2, 8)
        const extensionName = `TestExtension_${randomSuffix}`
        const commandId = `TestCommand_${randomSuffix}`

        app.registerExtension({
          name: extensionName,
          keybindings: [
            {
              combo: keyCombo,
              commandId: commandId
            }
          ],
          commands: [
            {
              id: commandId,
              function: eval(commandStr)
            }
          ]
        })
      },
      { keyCombo, commandStr: command.toString() }
    )
  }
}
