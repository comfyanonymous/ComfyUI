import type { NodeReplacementResponse } from './types'

import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { ServerFeatureFlag } from '@/composables/useFeatureFlags'
import { useSettingStore } from '@/platform/settings/settingStore'
import { api } from '@/scripts/api'
import { fetchNodeReplacements } from './nodeReplacementService'
import { useNodeReplacementStore } from './nodeReplacementStore'

vi.mock('@/platform/settings/settingStore', () => ({
  useSettingStore: vi.fn()
}))

vi.mock('./nodeReplacementService', () => ({
  fetchNodeReplacements: vi.fn()
}))

vi.mock('@/scripts/api', () => ({
  api: {
    getServerFeature: vi.fn()
  }
}))

function mockSettingStore(enabled: boolean) {
  vi.mocked(useSettingStore, { partial: true }).mockReturnValue({
    get: vi.fn().mockImplementation((key: string) => {
      if (key === 'Comfy.NodeReplacement.Enabled') {
        return enabled
      }
      return false
    }),
    load: vi.fn().mockResolvedValue(undefined)
  })
}

function createStore(settingEnabled = true, serverFeatureEnabled = true) {
  setActivePinia(createPinia())
  mockSettingStore(settingEnabled)
  vi.mocked(api.getServerFeature).mockImplementation(
    (flag: string, defaultValue?: unknown) => {
      if (flag === ServerFeatureFlag.NODE_REPLACEMENTS) {
        return serverFeatureEnabled
      }
      return defaultValue
    }
  )
  return useNodeReplacementStore()
}

describe('useNodeReplacementStore', () => {
  let store: ReturnType<typeof useNodeReplacementStore>

  beforeEach(() => {
    vi.clearAllMocks()
    store = createStore()
  })

  it('should initialize with empty replacements', () => {
    expect(store.replacements).toEqual({})
    expect(store.isLoaded).toBe(false)
  })

  describe('getReplacementFor', () => {
    it('should return first replacement for existing node type', () => {
      store.replacements = {
        OldNode: [
          {
            new_node_id: 'NewNodeA',
            old_node_id: 'OldNode',
            old_widget_ids: null,
            input_mapping: null,
            output_mapping: null
          },
          {
            new_node_id: 'NewNodeB',
            old_node_id: 'OldNode',
            old_widget_ids: null,
            input_mapping: null,
            output_mapping: null
          }
        ]
      }

      const result = store.getReplacementFor('OldNode')

      expect(result).not.toBeNull()
      expect(result?.new_node_id).toBe('NewNodeA')
    })

    it('should return null for non-existing node type', () => {
      store.replacements = {
        OldNode: [
          {
            new_node_id: 'NewNode',
            old_node_id: 'OldNode',
            old_widget_ids: null,
            input_mapping: null,
            output_mapping: null
          }
        ]
      }

      const result = store.getReplacementFor('NonExistentNode')

      expect(result).toBeNull()
    })

    it('should return null for empty replacement array', () => {
      store.replacements = {
        OldNode: []
      }

      const result = store.getReplacementFor('OldNode')

      expect(result).toBeNull()
    })

    it('should return null when feature is disabled', () => {
      store = createStore(false)
      store.replacements = {
        OldNode: [
          {
            new_node_id: 'NewNode',
            old_node_id: 'OldNode',
            old_widget_ids: null,
            input_mapping: null,
            output_mapping: null
          }
        ]
      }

      const result = store.getReplacementFor('OldNode')

      expect(result).toBeNull()
    })
  })

  describe('hasReplacement', () => {
    it('should return true when replacement exists', () => {
      store.replacements = {
        OldNode: [
          {
            new_node_id: 'NewNode',
            old_node_id: 'OldNode',
            old_widget_ids: null,
            input_mapping: null,
            output_mapping: null
          }
        ]
      }

      expect(store.hasReplacement('OldNode')).toBe(true)
    })

    it('should return false when node type does not exist', () => {
      store.replacements = {}

      expect(store.hasReplacement('NonExistentNode')).toBe(false)
    })

    it('should return false when replacement array is empty', () => {
      store.replacements = {
        OldNode: []
      }

      expect(store.hasReplacement('OldNode')).toBe(false)
    })

    it('should return false when feature is disabled', () => {
      store = createStore(false)
      store.replacements = {
        OldNode: [
          {
            new_node_id: 'NewNode',
            old_node_id: 'OldNode',
            old_widget_ids: null,
            input_mapping: null,
            output_mapping: null
          }
        ]
      }

      expect(store.hasReplacement('OldNode')).toBe(false)
    })
  })

  describe('isEnabled', () => {
    it('should return true when setting is enabled', () => {
      expect(store.isEnabled).toBe(true)
    })

    it('should return false when setting is disabled', () => {
      store = createStore(false)
      expect(store.isEnabled).toBe(false)
    })
  })

  describe('load', () => {
    const mockReplacements: NodeReplacementResponse = {
      OldNode: [
        {
          new_node_id: 'NewNode',
          old_node_id: 'OldNode',
          old_widget_ids: null,
          input_mapping: null,
          output_mapping: null
        }
      ]
    }

    beforeEach(() => {
      vi.mocked(fetchNodeReplacements).mockReset()
    })

    it('should fetch and assign replacements on successful load', async () => {
      vi.mocked(fetchNodeReplacements).mockResolvedValue(mockReplacements)
      store = createStore()

      await store.load()

      expect(fetchNodeReplacements).toHaveBeenCalledOnce()
      expect(store.replacements).toEqual(mockReplacements)
      expect(store.isLoaded).toBe(true)
    })

    it('should log error but not throw when fetch fails', async () => {
      const consoleErrorSpy = vi
        .spyOn(console, 'error')
        .mockImplementation(() => {})
      const error = new Error('Network error')
      vi.mocked(fetchNodeReplacements).mockRejectedValue(error)
      store = createStore()

      await expect(store.load()).resolves.toBeUndefined()

      expect(consoleErrorSpy).toHaveBeenCalledWith(
        'Failed to load node replacements:',
        error
      )
      expect(store.isLoaded).toBe(false)

      consoleErrorSpy.mockRestore()
    })

    it('should not fetch when setting is disabled', async () => {
      vi.mocked(fetchNodeReplacements).mockResolvedValue({})
      store = createStore(false)

      await store.load()

      expect(fetchNodeReplacements).not.toHaveBeenCalled()
      expect(store.isLoaded).toBe(false)
    })

    it('should not fetch when server feature flag is disabled', async () => {
      vi.mocked(fetchNodeReplacements).mockResolvedValue(mockReplacements)
      store = createStore(true, false)

      await store.load()

      expect(fetchNodeReplacements).not.toHaveBeenCalled()
      expect(store.isLoaded).toBe(false)
    })

    it('should not re-fetch when called twice', async () => {
      vi.mocked(fetchNodeReplacements).mockResolvedValue(mockReplacements)
      store = createStore()

      await store.load()
      await store.load()

      expect(fetchNodeReplacements).toHaveBeenCalledOnce()
    })
  })
})
