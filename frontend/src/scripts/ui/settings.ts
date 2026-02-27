import { t } from '@/i18n'
import { useSettingStore } from '@/platform/settings/settingStore'
import type { SettingParams } from '@/platform/settings/types'
import { useToastStore } from '@/platform/updates/common/toastStore'
import type { Settings } from '@/schemas/apiSchema'
import type { ComfyApp } from '@/scripts/app'

import { ComfyDialog } from './dialog'

export class ComfySettingsDialog extends ComfyDialog<HTMLDialogElement> {
  app: ComfyApp

  constructor(app: ComfyApp) {
    super()
    this.app = app
  }

  dispatchChange<T>(id: string, value: T, oldValue?: T) {
    this.dispatchEvent(
      new CustomEvent(id + '.change', {
        detail: {
          value,
          oldValue
        }
      })
    )
  }

  /**
   * @deprecated Use `settingStore.settingValues` instead.
   */
  get settingsValues() {
    return useSettingStore().settingValues
  }

  /**
   * @deprecated Use `settingStore.settingsById` instead.
   */
  get settingsLookup() {
    return useSettingStore().settingsById
  }

  /**
   * @deprecated Use `settingStore.settingsById` instead.
   */
  get settingsParamLookup() {
    return useSettingStore().settingsById
  }

  /**
   * @deprecated Use `settingStore.get` instead.
   */
  getSettingValue<K extends keyof Settings>(
    id: K,
    defaultValue?: Settings[K]
  ): Settings[K] {
    if (defaultValue !== undefined) {
      console.warn(
        `Parameter defaultValue is deprecated. The default value in settings definition will be used instead.`
      )
    }
    return useSettingStore().get(id)
  }

  /**
   * @deprecated Use `settingStore.getDefaultValue` instead.
   */
  getSettingDefaultValue<K extends keyof Settings>(
    id: K
  ): Settings[K] | undefined {
    return useSettingStore().getDefaultValue(id)
  }

  /**
   * @deprecated Use `settingStore.set` instead.
   */
  async setSettingValueAsync<K extends keyof Settings>(
    id: K,
    value: Settings[K]
  ) {
    await useSettingStore().set(id, value)
  }

  /**
   * @deprecated Use `settingStore.set` instead.
   */
  setSettingValue<K extends keyof Settings>(id: K, value: Settings[K]) {
    useSettingStore()
      .set(id, value)
      .catch((err) => {
        useToastStore().addAlert(
          t('toastMessages.errorSaveSetting', { id, err })
        )
      })
  }

  /**
   * @deprecated Deprecated for external callers/extensions. Use
   * `ComfyExtension.settings` field instead.
   *
   * Example:
   * ```ts
   * app.registerExtension({
   *   name: 'My Extension',
   *   settings: [
   *     {
   *       id: 'My.Setting',
   *       name: 'My Setting',
   *       type: 'text',
   *       defaultValue: 'Hello, world!'
   *     }
   *   ]
   * })
   * ```
   */
  addSetting(params: SettingParams) {
    const settingStore = useSettingStore()
    settingStore.addSetting(params)

    return {
      get value() {
        return settingStore.get(params.id)
      },
      set value(v) {
        settingStore.set(params.id, v)
      }
    }
  }
}
