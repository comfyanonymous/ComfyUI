import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it } from 'vitest'

import type { ServerConfig, ServerConfigValue } from '@/constants/serverConfig'
import type { FormItem } from '@/platform/settings/types'
import { useServerConfigStore } from '@/stores/serverConfigStore'

const dummyFormItem: FormItem = {
  name: '',
  type: 'text'
}

describe('useServerConfigStore', () => {
  let store: ReturnType<typeof useServerConfigStore>

  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    store = useServerConfigStore()
  })

  it('should initialize with empty configs', () => {
    expect(store.serverConfigs).toHaveLength(0)
    expect(Object.keys(store.serverConfigById)).toHaveLength(0)
    expect(Object.keys(store.serverConfigsByCategory)).toHaveLength(0)
    expect(Object.keys(store.serverConfigValues)).toHaveLength(0)
    expect(Object.keys(store.launchArgs)).toHaveLength(0)
  })

  it('should load server configs with default values', () => {
    const configs: ServerConfig<ServerConfigValue>[] = [
      {
        ...dummyFormItem,
        id: 'test.config1',
        defaultValue: 'default1',
        category: ['Test']
      },
      {
        ...dummyFormItem,
        id: 'test.config2',
        defaultValue: 'default2'
      }
    ]

    store.loadServerConfig(configs, {})

    expect(store.serverConfigs).toHaveLength(2)
    expect(store.serverConfigById['test.config1'].value).toBe('default1')
    expect(store.serverConfigById['test.config2'].value).toBe('default2')
  })

  it('should load server configs with provided values', () => {
    const configs: ServerConfig<ServerConfigValue>[] = [
      {
        ...dummyFormItem,
        id: 'test.config1',
        defaultValue: 'default1',
        category: ['Test']
      }
    ]

    store.loadServerConfig(configs, {
      'test.config1': 'custom1'
    })

    expect(store.serverConfigs).toHaveLength(1)
    expect(store.serverConfigById['test.config1'].value).toBe('custom1')
  })

  it('should organize configs by category', () => {
    const configs: ServerConfig<ServerConfigValue>[] = [
      {
        ...dummyFormItem,
        id: 'test.config1',
        defaultValue: 'default1',
        category: ['Test']
      },
      {
        ...dummyFormItem,
        id: 'test.config2',
        defaultValue: 'default2',
        category: ['Other']
      },
      {
        ...dummyFormItem,
        id: 'test.config3',
        defaultValue: 'default3'
      }
    ]

    store.loadServerConfig(configs, {})

    expect(Object.keys(store.serverConfigsByCategory)).toHaveLength(3)
    expect(store.serverConfigsByCategory['Test']).toHaveLength(1)
    expect(store.serverConfigsByCategory['Other']).toHaveLength(1)
    expect(store.serverConfigsByCategory['General']).toHaveLength(1)
  })

  it('should generate server config values excluding defaults', () => {
    const configs: ServerConfig<ServerConfigValue>[] = [
      {
        ...dummyFormItem,
        id: 'test.config1',
        defaultValue: 'default1'
      },
      {
        ...dummyFormItem,
        id: 'test.config2',
        defaultValue: 'default2'
      }
    ]

    store.loadServerConfig(configs, {
      'test.config1': 'custom1',
      'test.config2': 'default2'
    })

    expect(Object.keys(store.serverConfigValues)).toHaveLength(2)
    expect(store.serverConfigValues['test.config1']).toBe('custom1')
    expect(store.serverConfigValues['test.config2']).toBeUndefined()
  })

  it('should generate launch arguments with custom getValue function', () => {
    const configs: ServerConfig<ServerConfigValue>[] = [
      {
        ...dummyFormItem,
        id: 'test.config1',
        defaultValue: 'default1',
        getValue: (value: ServerConfigValue) => ({ customArg: value })
      },
      {
        ...dummyFormItem,
        id: 'test.config2',
        defaultValue: 'default2'
      }
    ]

    store.loadServerConfig(configs, {
      'test.config1': 'custom1',
      'test.config2': 'custom2'
    })

    expect(Object.keys(store.launchArgs)).toHaveLength(2)
    expect(store.launchArgs['customArg']).toBe('custom1')
    expect(store.launchArgs['test.config2']).toBe('custom2')
  })

  it('should not include default values in launch arguments', () => {
    const configs: ServerConfig<ServerConfigValue>[] = [
      {
        ...dummyFormItem,
        id: 'test.config1',
        defaultValue: 'default1'
      },
      {
        ...dummyFormItem,
        id: 'test.config2',
        defaultValue: 'default2'
      }
    ]

    store.loadServerConfig(configs, {
      'test.config1': 'custom1',
      'test.config2': 'default2'
    })

    expect(Object.keys(store.launchArgs)).toHaveLength(1)
    expect(store.launchArgs['test.config1']).toBe('custom1')
    expect(store.launchArgs['test.config2']).toBeUndefined()
  })

  it('should not include nullish values in launch arguments', () => {
    const configs: ServerConfig<ServerConfigValue>[] = [
      { ...dummyFormItem, id: 'test.config1', defaultValue: 'default1' },
      { ...dummyFormItem, id: 'test.config2', defaultValue: 'default2' },
      { ...dummyFormItem, id: 'test.config3', defaultValue: 'default3' },
      { ...dummyFormItem, id: 'test.config4', defaultValue: null }
    ]

    store.loadServerConfig(configs, {
      'test.config1': undefined,
      'test.config2': null,
      'test.config3': '',
      'test.config4': 0
    })

    expect(Object.keys(store.launchArgs)).toEqual([
      'test.config3',
      'test.config4'
    ])
    expect(Object.values(store.launchArgs)).toEqual(['', '0'])
    expect(store.serverConfigById['test.config3'].value).toBe('')
    expect(store.serverConfigById['test.config4'].value).toBe(0)
    expect(Object.values(store.serverConfigValues)).toEqual([
      undefined,
      undefined,
      '',
      0
    ])
  })

  it('should convert true to empty string in launch arguments', () => {
    store.loadServerConfig(
      [
        {
          ...dummyFormItem,
          id: 'test.config1',
          defaultValue: 0
        }
      ],
      {
        'test.config1': true
      }
    )
    expect(store.launchArgs['test.config1']).toBe('')
    expect(store.commandLineArgs).toBe('--test.config1')
  })

  it('should convert number to string in launch arguments', () => {
    store.loadServerConfig(
      [
        {
          ...dummyFormItem,
          id: 'test.config1',
          defaultValue: 1
        }
      ],
      {
        'test.config1': 123
      }
    )
    expect(store.launchArgs['test.config1']).toBe('123')
    expect(store.commandLineArgs).toBe('--test.config1 123')
  })

  it('should drop nullish values in launch arguments', () => {
    store.loadServerConfig(
      [
        {
          ...dummyFormItem,
          id: 'test.config1',
          defaultValue: 1
        }
      ],
      {
        'test.config1': null
      }
    )
    expect(Object.keys(store.launchArgs)).toHaveLength(0)
  })

  it('should track modified configs', () => {
    const configs = [
      {
        ...dummyFormItem,
        id: 'test.config1',
        defaultValue: 'default1'
      },
      {
        ...dummyFormItem,
        id: 'test.config2',
        defaultValue: 'default2'
      }
    ]

    store.loadServerConfig(configs, {
      'test.config1': 'initial1'
    })

    // Initially no modified configs
    expect(store.modifiedConfigs).toHaveLength(0)

    // Modify config1's value after loading
    store.serverConfigById['test.config1'].value = 'custom1'

    // Now config1 should be in modified configs
    expect(store.modifiedConfigs).toHaveLength(1)
    expect(store.modifiedConfigs[0].id).toBe('test.config1')
    expect(store.modifiedConfigs[0].value).toBe('custom1')
    expect(store.modifiedConfigs[0].initialValue).toBe('initial1')

    // Change config1 back to default
    store.serverConfigById['test.config1'].value = 'initial1'

    // Should go back to no modified configs
    expect(store.modifiedConfigs).toHaveLength(0)
  })
})
