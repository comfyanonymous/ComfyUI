import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import {
  getSettingInfo,
  useSettingStore
} from '@/platform/settings/settingStore'
import type { SettingParams } from '@/platform/settings/types'
import type { Settings } from '@/schemas/apiSchema'
import { api } from '@/scripts/api'
import { app } from '@/scripts/app'

// Mock the api
vi.mock('@/scripts/api', () => ({
  api: {
    getSettings: vi.fn(),
    storeSetting: vi.fn(),
    storeSettings: vi.fn()
  }
}))

// Mock the app
vi.mock('@/scripts/app', () => ({
  app: {
    ui: {
      settings: {
        dispatchChange: vi.fn()
      }
    }
  }
}))

describe('useSettingStore', () => {
  let store: ReturnType<typeof useSettingStore>

  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    store = useSettingStore()
    vi.clearAllMocks()
  })

  it('should initialize with empty settings', () => {
    expect(store.settingValues).toEqual({})
    expect(store.settingsById).toEqual({})
  })

  describe('load', () => {
    it('should load settings from API', async () => {
      const mockSettings = { 'test.setting': 'value' }
      vi.mocked(api.getSettings).mockResolvedValue(
        mockSettings as Partial<Settings> as Settings
      )

      await store.load()

      expect(store.settingValues).toEqual(mockSettings)
      expect(api.getSettings).toHaveBeenCalled()
    })

    it('should set error if settings are loaded after registration', async () => {
      const setting: SettingParams = {
        id: 'test.setting',
        name: 'test.setting',
        type: 'text',
        defaultValue: 'default'
      }
      store.addSetting(setting)

      await store.load()

      expect(store.error).toBeInstanceOf(Error)
      if (store.error instanceof Error) {
        expect(store.error.message).toBe(
          'Setting values must be loaded before any setting is registered.'
        )
      }
    })
  })

  describe('addSetting', () => {
    it('should register a new setting', () => {
      const setting: SettingParams = {
        id: 'test.setting',
        name: 'test.setting',
        type: 'text',
        defaultValue: 'default'
      }

      store.addSetting(setting)

      expect(store.settingsById['test.setting']).toEqual(setting)
    })

    it('should warn and skip for duplicate setting ID', () => {
      const setting: SettingParams = {
        id: 'test.setting',
        name: 'test.setting',
        type: 'text',
        defaultValue: 'default'
      }
      const consoleWarnSpy = vi
        .spyOn(console, 'warn')
        .mockImplementation(() => {})

      store.addSetting(setting)
      store.addSetting(setting)

      expect(consoleWarnSpy).toHaveBeenCalledWith(
        'Setting already registered: test.setting'
      )
      consoleWarnSpy.mockRestore()
    })

    it('should migrate deprecated values', () => {
      const setting: SettingParams = {
        id: 'test.setting',
        name: 'test.setting',
        type: 'text',
        defaultValue: 'default',
        migrateDeprecatedValue: (val: unknown) => (val as string).toUpperCase()
      }

      store.settingValues['test.setting'] = 'oldvalue'
      store.addSetting(setting)

      expect(store.settingValues['test.setting']).toBe('OLDVALUE')
    })
  })

  describe('getDefaultValue', () => {
    beforeEach(() => {
      // Set up installed version for most tests
      store.settingValues['Comfy.InstalledVersion'] = '1.30.0'
    })

    it('should return regular default value when no defaultsByInstallVersion', () => {
      const setting: SettingParams = {
        id: 'test.setting',
        name: 'Test Setting',
        type: 'text',
        defaultValue: 'regular-default'
      }
      store.addSetting(setting)

      const result = store.getDefaultValue('test.setting')
      expect(result).toBe('regular-default')
    })

    it('should return versioned default when user version matches', () => {
      const setting: SettingParams = {
        id: 'test.setting',
        name: 'Test Setting',
        type: 'text',
        defaultValue: 'regular-default',
        defaultsByInstallVersion: {
          '1.21.3': 'version-1.21.3-default',
          '1.40.3': 'version-1.40.3-default'
        }
      }
      store.addSetting(setting)

      const result = store.getDefaultValue('test.setting')
      // installedVersion is 1.30.0, so should get 1.21.3 default
      expect(result).toBe('version-1.21.3-default')
    })

    it('should return latest versioned default when user version is higher', () => {
      store.settingValues['Comfy.InstalledVersion'] = '1.50.0'

      const setting: SettingParams = {
        id: 'test.setting',
        name: 'Test Setting',
        type: 'text',
        defaultValue: 'regular-default',
        defaultsByInstallVersion: {
          '1.21.3': 'version-1.21.3-default',
          '1.40.3': 'version-1.40.3-default'
        }
      }
      store.addSetting(setting)

      const result = store.getDefaultValue('test.setting')
      // installedVersion is 1.50.0, so should get 1.40.3 default
      expect(result).toBe('version-1.40.3-default')
    })

    it('should return regular default when user version is lower than all versioned defaults', () => {
      store.settingValues['Comfy.InstalledVersion'] = '1.10.0'

      const setting: SettingParams = {
        id: 'test.setting',
        name: 'Test Setting',
        type: 'text',
        defaultValue: 'regular-default',
        defaultsByInstallVersion: {
          '1.21.3': 'version-1.21.3-default',
          '1.40.3': 'version-1.40.3-default'
        }
      }
      store.addSetting(setting)

      const result = store.getDefaultValue('test.setting')
      // installedVersion is 1.10.0, lower than all versioned defaults
      expect(result).toBe('regular-default')
    })

    it('should return regular default when no installed version (existing users)', () => {
      // Clear installed version to simulate existing user
      delete store.settingValues['Comfy.InstalledVersion']

      const setting: SettingParams = {
        id: 'test.setting',
        name: 'Test Setting',
        type: 'text',
        defaultValue: 'regular-default',
        defaultsByInstallVersion: {
          '1.21.3': 'version-1.21.3-default',
          '1.40.3': 'version-1.40.3-default'
        }
      }
      store.addSetting(setting)

      const result = store.getDefaultValue('test.setting')
      // No installed version, should use backward compatibility
      expect(result).toBe('regular-default')
    })

    it('should handle function-based versioned defaults', () => {
      const setting: SettingParams = {
        id: 'test.setting',
        name: 'Test Setting',
        type: 'text',
        defaultValue: 'regular-default',
        defaultsByInstallVersion: {
          '1.21.3': () => 'dynamic-version-1.21.3-default',
          '1.40.3': () => 'dynamic-version-1.40.3-default'
        }
      }
      store.addSetting(setting)

      const result = store.getDefaultValue('test.setting')
      // installedVersion is 1.30.0, so should get 1.21.3 default (executed)
      expect(result).toBe('dynamic-version-1.21.3-default')
    })

    it('should handle function-based regular defaults with versioned defaults', () => {
      store.settingValues['Comfy.InstalledVersion'] = '1.10.0'

      const setting: SettingParams = {
        id: 'test.setting',
        name: 'Test Setting',
        type: 'text',
        defaultValue: () => 'dynamic-regular-default',
        defaultsByInstallVersion: {
          '1.21.3': 'version-1.21.3-default',
          '1.40.3': 'version-1.40.3-default'
        }
      }
      store.addSetting(setting)

      const result = store.getDefaultValue('test.setting')
      // installedVersion is 1.10.0, should fallback to function-based regular default
      expect(result).toBe('dynamic-regular-default')
    })

    it('should handle complex version comparison correctly', () => {
      const setting: SettingParams = {
        id: 'test.setting',
        name: 'Test Setting',
        type: 'text',
        defaultValue: 'regular-default',
        defaultsByInstallVersion: {
          '1.21.3': 'version-1.21.3-default',
          '1.21.10': 'version-1.21.10-default',
          '1.40.3': 'version-1.40.3-default'
        }
      }
      store.addSetting(setting)

      // Test with 1.21.5 - should get 1.21.3 default
      store.settingValues['Comfy.InstalledVersion'] = '1.21.5'
      expect(store.getDefaultValue('test.setting')).toBe(
        'version-1.21.3-default'
      )

      // Test with 1.21.15 - should get 1.21.10 default
      store.settingValues['Comfy.InstalledVersion'] = '1.21.15'
      expect(store.getDefaultValue('test.setting')).toBe(
        'version-1.21.10-default'
      )

      // Test with 1.21.3 exactly - should get 1.21.3 default
      store.settingValues['Comfy.InstalledVersion'] = '1.21.3'
      expect(store.getDefaultValue('test.setting')).toBe(
        'version-1.21.3-default'
      )
    })

    it('should work with get() method using versioned defaults', () => {
      const setting: SettingParams = {
        id: 'test.setting',
        name: 'Test Setting',
        type: 'text',
        defaultValue: 'regular-default',
        defaultsByInstallVersion: {
          '1.21.3': 'version-1.21.3-default',
          '1.40.3': 'version-1.40.3-default'
        }
      }
      store.addSetting(setting)

      // get() should use getDefaultValue internally
      const result = store.get('test.setting')
      expect(result).toBe('version-1.21.3-default')
    })

    it('should handle mixed function and static versioned defaults', () => {
      const setting: SettingParams = {
        id: 'test.setting',
        name: 'Test Setting',
        type: 'text',
        defaultValue: 'regular-default',
        defaultsByInstallVersion: {
          '1.21.3': () => 'dynamic-1.21.3-default',
          '1.40.3': 'static-1.40.3-default'
        }
      }
      store.addSetting(setting)

      // Test with 1.30.0 - should get dynamic 1.21.3 default
      store.settingValues['Comfy.InstalledVersion'] = '1.30.0'
      expect(store.getDefaultValue('test.setting')).toBe(
        'dynamic-1.21.3-default'
      )

      // Test with 1.50.0 - should get static 1.40.3 default
      store.settingValues['Comfy.InstalledVersion'] = '1.50.0'
      expect(store.getDefaultValue('test.setting')).toBe(
        'static-1.40.3-default'
      )
    })

    it('should handle version sorting correctly', () => {
      const setting: SettingParams = {
        id: 'test.setting',
        name: 'Test Setting',
        type: 'text',
        defaultValue: 'regular-default',
        defaultsByInstallVersion: {
          '1.40.3': 'version-1.40.3-default',
          '1.21.3': 'version-1.21.3-default', // Unsorted order
          '1.35.0': 'version-1.35.0-default'
        }
      }
      store.addSetting(setting)

      // Test with 1.37.0 - should get 1.35.0 default (highest version <= 1.37.0)
      store.settingValues['Comfy.InstalledVersion'] = '1.37.0'
      expect(store.getDefaultValue('test.setting')).toBe(
        'version-1.35.0-default'
      )
    })
  })

  describe('get and set', () => {
    it('should get default value when setting not exists', () => {
      const setting: SettingParams = {
        id: 'test.setting',
        name: 'test.setting',
        type: 'text',
        defaultValue: 'default'
      }
      store.addSetting(setting)

      expect(store.get('test.setting')).toBe('default')
    })

    it('should set value and trigger onChange', async () => {
      const onChangeMock = vi.fn()
      const dispatchChangeMock = vi.mocked(app.ui.settings.dispatchChange)
      const setting: SettingParams = {
        id: 'test.setting',
        name: 'test.setting',
        type: 'text',
        defaultValue: 'default',
        onChange: onChangeMock
      }
      store.addSetting(setting)
      // Adding the new setting should trigger onChange
      expect(onChangeMock).toHaveBeenCalledTimes(1)
      expect(dispatchChangeMock).toHaveBeenCalledTimes(1)

      await store.set('test.setting', 'newvalue')

      expect(store.get('test.setting')).toBe('newvalue')
      expect(onChangeMock).toHaveBeenCalledWith('newvalue', 'default')
      expect(onChangeMock).toHaveBeenCalledTimes(2)
      expect(dispatchChangeMock).toHaveBeenCalledTimes(2)
      expect(api.storeSetting).toHaveBeenCalledWith('test.setting', 'newvalue')

      // Set the same value again, it should not trigger onChange
      await store.set('test.setting', 'newvalue')
      expect(onChangeMock).toHaveBeenCalledTimes(2)
      expect(dispatchChangeMock).toHaveBeenCalledTimes(2)

      // Set a different value, it should trigger onChange
      await store.set('test.setting', 'differentvalue')
      expect(onChangeMock).toHaveBeenCalledWith('differentvalue', 'newvalue')
      expect(onChangeMock).toHaveBeenCalledTimes(3)
      expect(dispatchChangeMock).toHaveBeenCalledTimes(3)
      expect(api.storeSetting).toHaveBeenCalledWith(
        'test.setting',
        'differentvalue'
      )
    })

    describe('object mutation prevention', () => {
      beforeEach(() => {
        const setting: SettingParams = {
          id: 'test.setting',
          name: 'Test setting',
          type: 'hidden',
          defaultValue: {}
        }
        store.addSetting(setting)
      })

      it('should prevent mutations of objects after set', async () => {
        const originalObject = { foo: 'bar', nested: { value: 123 } }

        await store.set('test.setting', originalObject)

        // Attempt to mutate the original object
        originalObject.foo = 'changed'
        originalObject.nested.value = 456

        // Get the stored value
        const storedValue = store.get('test.setting')

        // Verify the stored value wasn't affected by the mutation
        expect(storedValue).toEqual({ foo: 'bar', nested: { value: 123 } })
      })

      it('should prevent mutations of retrieved objects', async () => {
        const initialValue = { foo: 'bar', nested: { value: 123 } }

        // Set initial value
        await store.set('test.setting', initialValue)

        // Get the value and try to mutate it
        const retrievedValue = store.get('test.setting')
        retrievedValue.foo = 'changed'
        if (retrievedValue.nested) {
          retrievedValue.nested.value = 456
        }

        // Get the value again
        const newRetrievedValue = store.get('test.setting')

        // Verify the stored value wasn't affected by the mutation
        expect(newRetrievedValue).toEqual({
          foo: 'bar',
          nested: { value: 123 }
        })
      })

      it('should prevent mutations of arrays after set', async () => {
        const originalArray = [1, 2, { value: 3 }]

        await store.set('test.setting', originalArray)

        // Attempt to mutate the original array
        originalArray.push(4)
        if (typeof originalArray[2] === 'object') {
          originalArray[2].value = 999
        }

        // Get the stored value
        const storedValue = store.get('test.setting')

        // Verify the stored value wasn't affected by the mutation
        expect(storedValue).toEqual([1, 2, { value: 3 }])
      })

      it('should prevent mutations of retrieved arrays', async () => {
        const initialArray = [1, 2, { value: 3 }]

        // Set initial value
        await store.set('test.setting', initialArray)

        // Get the value and try to mutate it
        const retrievedArray = store.get('test.setting')
        retrievedArray.push(4)
        if (typeof retrievedArray[2] === 'object') {
          retrievedArray[2].value = 999
        }

        // Get the value again
        const newRetrievedValue = store.get('test.setting')

        // Verify the stored value wasn't affected by the mutation
        expect(newRetrievedValue).toEqual([1, 2, { value: 3 }])
      })
    })
  })

  describe('setMany', () => {
    it('should set multiple values and make a single API call', async () => {
      const onChange1 = vi.fn()
      const onChange2 = vi.fn()
      store.addSetting({
        id: 'Comfy.Release.Version',
        name: 'Release Version',
        type: 'hidden',
        defaultValue: '',
        onChange: onChange1
      })
      store.addSetting({
        id: 'Comfy.Release.Status',
        name: 'Release Status',
        type: 'hidden',
        defaultValue: 'skipped',
        onChange: onChange2
      })
      vi.clearAllMocks()

      await store.setMany({
        'Comfy.Release.Version': '1.0.0',
        'Comfy.Release.Status': 'changelog seen'
      })

      expect(store.get('Comfy.Release.Version')).toBe('1.0.0')
      expect(store.get('Comfy.Release.Status')).toBe('changelog seen')
      expect(onChange1).toHaveBeenCalledWith('1.0.0', '')
      expect(onChange2).toHaveBeenCalledWith('changelog seen', 'skipped')
      expect(api.storeSettings).toHaveBeenCalledTimes(1)
      expect(api.storeSettings).toHaveBeenCalledWith({
        'Comfy.Release.Version': '1.0.0',
        'Comfy.Release.Status': 'changelog seen'
      })
      expect(api.storeSetting).not.toHaveBeenCalled()
    })

    it('should skip unchanged values', async () => {
      store.addSetting({
        id: 'Comfy.Release.Version',
        name: 'Release Version',
        type: 'hidden',
        defaultValue: ''
      })
      store.addSetting({
        id: 'Comfy.Release.Status',
        name: 'Release Status',
        type: 'hidden',
        defaultValue: 'skipped'
      })
      await store.set('Comfy.Release.Version', 'existing')
      vi.clearAllMocks()

      await store.setMany({
        'Comfy.Release.Version': 'existing',
        'Comfy.Release.Status': 'changelog seen'
      })

      expect(api.storeSettings).toHaveBeenCalledWith({
        'Comfy.Release.Status': 'changelog seen'
      })
    })

    it('should not call API when all values are unchanged', async () => {
      store.addSetting({
        id: 'Comfy.Release.Version',
        name: 'Release Version',
        type: 'hidden',
        defaultValue: ''
      })
      await store.set('Comfy.Release.Version', 'existing')
      vi.clearAllMocks()

      await store.setMany({ 'Comfy.Release.Version': 'existing' })

      expect(api.storeSettings).not.toHaveBeenCalled()
    })
  })
})

describe('getSettingInfo', () => {
  const baseSetting: SettingParams = {
    id: 'test.setting',
    name: 'test.setting',
    type: 'text',
    defaultValue: 'default'
  }

  it('should handle settings with explicit category array', () => {
    const setting: SettingParams = {
      ...baseSetting,
      id: 'test.setting',
      category: ['Main', 'Sub', 'Detail']
    }

    const result = getSettingInfo(setting)

    expect(result).toEqual({
      category: 'Main',
      subCategory: 'Sub'
    })
  })

  it('should handle settings with id-based categorization', () => {
    const setting: SettingParams = {
      ...baseSetting,
      id: 'main.sub.setting.name'
    }

    const result = getSettingInfo(setting)

    expect(result).toEqual({
      category: 'main',
      subCategory: 'sub'
    })
  })

  it('should use "Other" as default subCategory when missing', () => {
    const setting: SettingParams = {
      ...baseSetting,
      id: 'single.setting',
      category: ['single']
    }

    const result = getSettingInfo(setting)

    expect(result).toEqual({
      category: 'single',
      subCategory: 'Other'
    })
  })

  it('should use "Other" as default category when missing', () => {
    const setting: SettingParams = {
      ...baseSetting,
      id: 'single.setting',
      category: []
    }

    const result = getSettingInfo(setting)

    expect(result).toEqual({
      category: 'Other',
      subCategory: 'Other'
    })
  })
})
