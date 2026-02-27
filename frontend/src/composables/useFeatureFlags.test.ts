import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { isReactive, isReadonly } from 'vue'

import {
  ServerFeatureFlag,
  useFeatureFlags
} from '@/composables/useFeatureFlags'
import * as distributionTypes from '@/platform/distribution/types'
import { api } from '@/scripts/api'

// Mock the API module
vi.mock('@/scripts/api', () => ({
  api: {
    getServerFeature: vi.fn()
  }
}))

// Mock the distribution types module
vi.mock('@/platform/distribution/types', () => ({
  isCloud: false,
  isNightly: false
}))

describe('useFeatureFlags', () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  describe('flags object', () => {
    it('should provide reactive readonly flags', () => {
      const { flags } = useFeatureFlags()

      expect(isReadonly(flags)).toBe(true)
      expect(isReactive(flags)).toBe(true)
    })

    it('should access supportsPreviewMetadata', () => {
      vi.mocked(api.getServerFeature).mockImplementation(
        (path, defaultValue) => {
          if (path === ServerFeatureFlag.SUPPORTS_PREVIEW_METADATA) return true
          return defaultValue
        }
      )

      const { flags } = useFeatureFlags()
      expect(flags.supportsPreviewMetadata).toBe(true)
      expect(api.getServerFeature).toHaveBeenCalledWith(
        ServerFeatureFlag.SUPPORTS_PREVIEW_METADATA
      )
    })

    it('should access maxUploadSize', () => {
      vi.mocked(api.getServerFeature).mockImplementation(
        (path, defaultValue) => {
          if (path === ServerFeatureFlag.MAX_UPLOAD_SIZE) return 209715200 // 200MB
          return defaultValue
        }
      )

      const { flags } = useFeatureFlags()
      expect(flags.maxUploadSize).toBe(209715200)
      expect(api.getServerFeature).toHaveBeenCalledWith(
        ServerFeatureFlag.MAX_UPLOAD_SIZE
      )
    })

    it('should access supportsManagerV4', () => {
      vi.mocked(api.getServerFeature).mockImplementation(
        (path, defaultValue) => {
          if (path === ServerFeatureFlag.MANAGER_SUPPORTS_V4) return true
          return defaultValue
        }
      )

      const { flags } = useFeatureFlags()
      expect(flags.supportsManagerV4).toBe(true)
      expect(api.getServerFeature).toHaveBeenCalledWith(
        ServerFeatureFlag.MANAGER_SUPPORTS_V4
      )
    })

    it('should return undefined when features are not available and no default provided', () => {
      vi.mocked(api.getServerFeature).mockImplementation(
        (_path, defaultValue) => defaultValue
      )

      const { flags } = useFeatureFlags()
      expect(flags.supportsPreviewMetadata).toBeUndefined()
      expect(flags.maxUploadSize).toBeUndefined()
      expect(flags.supportsManagerV4).toBeUndefined()
    })
  })

  describe('featureFlag', () => {
    it('should create reactive computed for custom feature flags', () => {
      vi.mocked(api.getServerFeature).mockImplementation(
        (path, defaultValue) => {
          if (path === 'custom.feature') return 'custom-value'
          return defaultValue
        }
      )

      const { featureFlag } = useFeatureFlags()
      const customFlag = featureFlag('custom.feature', 'default')

      expect(customFlag.value).toBe('custom-value')
      expect(api.getServerFeature).toHaveBeenCalledWith(
        'custom.feature',
        'default'
      )
    })

    it('should handle nested paths', () => {
      vi.mocked(api.getServerFeature).mockImplementation(
        (path, defaultValue) => {
          if (path === 'extension.custom.nested.feature') return true
          return defaultValue
        }
      )

      const { featureFlag } = useFeatureFlags()
      const nestedFlag = featureFlag('extension.custom.nested.feature', false)

      expect(nestedFlag.value).toBe(true)
    })

    it('should work with ServerFeatureFlag enum', () => {
      vi.mocked(api.getServerFeature).mockImplementation(
        (path, defaultValue) => {
          if (path === ServerFeatureFlag.MAX_UPLOAD_SIZE) return 104857600
          return defaultValue
        }
      )

      const { featureFlag } = useFeatureFlags()
      const maxUploadSize = featureFlag(ServerFeatureFlag.MAX_UPLOAD_SIZE)

      expect(maxUploadSize.value).toBe(104857600)
    })
  })

  describe('linearToggleEnabled', () => {
    it('should return true when isNightly is true', () => {
      vi.mocked(distributionTypes).isNightly = true

      const { flags } = useFeatureFlags()
      expect(flags.linearToggleEnabled).toBe(true)
      expect(api.getServerFeature).not.toHaveBeenCalled()
    })

    it('should check remote config and server feature when isNightly is false', () => {
      vi.mocked(distributionTypes).isNightly = false
      vi.mocked(api.getServerFeature).mockImplementation(
        (path, defaultValue) => {
          if (path === ServerFeatureFlag.LINEAR_TOGGLE_ENABLED) return true
          return defaultValue
        }
      )

      const { flags } = useFeatureFlags()
      expect(flags.linearToggleEnabled).toBe(true)
      expect(api.getServerFeature).toHaveBeenCalledWith(
        ServerFeatureFlag.LINEAR_TOGGLE_ENABLED,
        false
      )
    })

    it('should return false when isNightly is false and flag is disabled', () => {
      vi.mocked(distributionTypes).isNightly = false
      vi.mocked(api.getServerFeature).mockImplementation(
        (_path, defaultValue) => defaultValue
      )

      const { flags } = useFeatureFlags()
      expect(flags.linearToggleEnabled).toBe(false)
    })
  })

  describe('dev override via localStorage', () => {
    afterEach(() => {
      localStorage.clear()
    })

    it('resolveFlag returns localStorage override over remoteConfig and server value', () => {
      vi.mocked(api.getServerFeature).mockReturnValue(false)
      localStorage.setItem('ff:model_upload_button_enabled', 'true')

      const { flags } = useFeatureFlags()
      expect(flags.modelUploadButtonEnabled).toBe(true)
    })

    it('resolveFlag falls through to server when no override is set', () => {
      vi.mocked(api.getServerFeature).mockImplementation(
        (path, defaultValue) => {
          if (path === ServerFeatureFlag.ASSET_RENAME_ENABLED) return true
          return defaultValue
        }
      )

      const { flags } = useFeatureFlags()
      expect(flags.assetRenameEnabled).toBe(true)
    })

    it('direct server flags delegate override to api.getServerFeature', () => {
      vi.mocked(api.getServerFeature).mockImplementation((path) => {
        if (path === ServerFeatureFlag.SUPPORTS_PREVIEW_METADATA)
          return 'overridden'
        return undefined
      })

      const { flags } = useFeatureFlags()
      expect(flags.supportsPreviewMetadata).toBe('overridden')
    })

    it('teamWorkspacesEnabled override bypasses isCloud and isAuthenticatedConfigLoaded guards', () => {
      vi.mocked(distributionTypes).isCloud = false
      localStorage.setItem('ff:team_workspaces_enabled', 'true')

      const { flags } = useFeatureFlags()
      expect(flags.teamWorkspacesEnabled).toBe(true)
    })
  })
})
