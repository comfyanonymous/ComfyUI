import type { Settings } from '@/schemas/apiSchema'

type SettingInputType =
  | 'boolean'
  | 'number'
  | 'slider'
  | 'knob'
  | 'combo'
  | 'radio'
  | 'text'
  | 'image'
  | 'color'
  | 'url'
  | 'hidden'
  | 'backgroundImage'

type SettingCustomRenderer = (
  name: string,
  setter: (v: unknown) => void,
  value: unknown,
  attrs?: Record<string, unknown>
) => HTMLElement

export interface SettingOption {
  text: string
  value?: string | number
}

export interface SettingParams<TValue = unknown> extends FormItem {
  id: keyof Settings
  defaultValue: TValue | (() => TValue)
  defaultsByInstallVersion?: Record<`${number}.${number}.${number}`, TValue>
  onChange?(newValue: TValue, oldValue?: TValue): void
  // By default category is id.split('.'). However, changing id to assign
  // new category has poor backward compatibility. Use this field to overwrite
  // default category from id.
  // Note: Like id, category value need to be unique.
  category?: string[]
  experimental?: boolean
  deprecated?: boolean
  migrateDeprecatedValue?: (value: TValue) => TValue
  // Version of the setting when it was added
  versionAdded?: string
  // Version of the setting when it was last modified
  versionModified?: string
  // sortOrder for sorting settings within a group. Higher values appear first.
  // Default is 0 if not specified.
  sortOrder?: number
  hideInVueNodes?: boolean
}

/**
 * The base form item for rendering in a form.
 */
export interface FormItem {
  name: string
  type: SettingInputType | SettingCustomRenderer
  tooltip?: string
  attrs?: Record<string, unknown>
  options?: Array<string | SettingOption>
}

export interface ISettingGroup {
  label: string
  category?: string
  settings: SettingParams[]
}

export type SettingPanelType =
  | 'about'
  | 'credits'
  | 'extension'
  | 'keybinding'
  | 'secrets'
  | 'server-config'
  | 'subscription'
  | 'user'
  | 'workspace'
