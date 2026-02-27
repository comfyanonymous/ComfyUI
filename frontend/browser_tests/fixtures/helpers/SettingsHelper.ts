import type { Page } from '@playwright/test'

export class SettingsHelper {
  constructor(private readonly page: Page) {}

  async setSetting(settingId: string, settingValue: unknown): Promise<void> {
    await this.page.evaluate(
      async ({ id, value }) => {
        await window.app!.extensionManager.setting.set(id, value)
      },
      { id: settingId, value: settingValue }
    )
  }

  async getSetting<T = unknown>(settingId: string): Promise<T> {
    return (await this.page.evaluate(async (id) => {
      return await window.app!.extensionManager.setting.get(id)
    }, settingId)) as T
  }
}
