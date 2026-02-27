import { until } from '@vueuse/core'
import { setActivePinia } from 'pinia'
import { compare } from 'semver'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { ref } from 'vue'

import type { ReleaseNote } from '@/platform/updates/common/releaseService'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useReleaseStore } from '@/platform/updates/common/releaseStore'
import { useReleaseService } from '@/platform/updates/common/releaseService'
import { useSystemStatsStore } from '@/stores/systemStatsStore'
import { createTestingPinia } from '@pinia/testing'
import type { SystemStats } from '@/types'

// Mock the dependencies
vi.mock('semver', () => ({
  compare: vi.fn(),
  valid: vi.fn(() => '1.0.0')
}))

const mockData = vi.hoisted(() => ({ isDesktop: true }))

vi.mock('@/platform/distribution/types', () => ({
  get isDesktop() {
    return mockData.isDesktop
  },
  isCloud: false
}))

vi.mock('@/platform/updates/common/releaseService', () => {
  const getReleases = vi.fn()
  const isLoading = ref(false)
  const error = ref<string | null>(null)
  return {
    useReleaseService: () => ({
      getReleases,
      isLoading,
      error
    })
  }
})

vi.mock('@/platform/settings/settingStore', () => {
  const get = vi.fn((key: string) => {
    if (key === 'Comfy.Notification.ShowVersionUpdates') return true
    return null
  })
  const set = vi.fn()
  const setMany = vi.fn()
  return {
    useSettingStore: () => ({ get, set, setMany })
  }
})

const mockSystemStatsState = vi.hoisted(() => ({
  systemStats: {
    system: {
      comfyui_version: '1.0.0',
      argv: []
    }
  } satisfies {
    system: Partial<SystemStats['system']>
  },
  isInitialized: true,
  reset() {
    this.systemStats = {
      system: {
        comfyui_version: '1.0.0',
        argv: []
      } satisfies Partial<SystemStats['system']>
    }
    this.isInitialized = true
  }
}))
vi.mock('@/stores/systemStatsStore', () => {
  const refetchSystemStats = vi.fn()
  const getFormFactor = vi.fn(() => 'git-windows')
  return {
    useSystemStatsStore: () => ({
      get systemStats() {
        return mockSystemStatsState.systemStats
      },
      set systemStats(val) {
        mockSystemStatsState.systemStats = val
      },
      get isInitialized() {
        return mockSystemStatsState.isInitialized
      },
      set isInitialized(val) {
        mockSystemStatsState.isInitialized = val
      },
      refetchSystemStats,
      getFormFactor
    })
  }
})
vi.mock('@vueuse/core', () => ({
  until: vi.fn(() => Promise.resolve()),
  useStorage: vi.fn(() => ({ value: {} })),
  createSharedComposable: vi.fn((fn) => fn)
}))

describe('useReleaseStore', () => {
  const mockRelease = {
    id: 1,
    project: 'comfyui' as const,
    version: '1.2.0',
    content: 'New features and improvements',
    published_at: '2023-12-01T00:00:00Z',
    attention: 'high' as const
  }

  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))

    vi.resetAllMocks()
    mockSystemStatsState.reset()
  })

  describe('initial state', () => {
    it('should initialize with default state', () => {
      const store = useReleaseStore()
      expect(store.releases).toEqual([])
      expect(store.isLoading).toBe(false)
      expect(store.error).toBeNull()
    })
  })

  describe('computed properties', () => {
    it('should return most recent release', () => {
      const store = useReleaseStore()
      const olderRelease = {
        ...mockRelease,
        id: 2,
        version: '1.1.0',
        published_at: '2023-11-01T00:00:00Z'
      }

      store.releases = [mockRelease, olderRelease]
      expect(store.recentRelease).toEqual(mockRelease)
    })

    it('should return 3 most recent releases', () => {
      const store = useReleaseStore()
      const releases = [
        mockRelease,
        { ...mockRelease, id: 2, version: '1.1.0' },
        { ...mockRelease, id: 3, version: '1.0.0' },
        { ...mockRelease, id: 4, version: '0.9.0' }
      ]

      store.releases = releases
      expect(store.recentReleases).toEqual(releases.slice(0, 3))
    })

    it('should show update button (shouldShowUpdateButton)', () => {
      const store = useReleaseStore()
      vi.mocked(compare).mockReturnValue(1) // newer version available

      store.releases = [mockRelease]
      expect(store.shouldShowUpdateButton).toBe(true)
    })

    it('should not show update button when no new version', () => {
      const store = useReleaseStore()
      vi.mocked(compare).mockReturnValue(-1) // current version is newer

      store.releases = [mockRelease]
      expect(store.shouldShowUpdateButton).toBe(false)
    })
  })

  describe('showVersionUpdates setting', () => {
    describe('when notifications are enabled', () => {
      beforeEach(() => {
        const settingStore = useSettingStore()
        vi.mocked(settingStore.get).mockImplementation((key: string) => {
          if (key === 'Comfy.Notification.ShowVersionUpdates') return true
          return null
        })
      })

      it('should show toast for medium/high attention releases', () => {
        const store = useReleaseStore()
        vi.mocked(compare).mockReturnValue(1)
        store.releases = [mockRelease]

        expect(store.shouldShowToast).toBe(true)
      })

      it('should not show toast for low attention releases', () => {
        const store = useReleaseStore()
        vi.mocked(compare).mockReturnValue(1)

        const lowAttentionRelease = {
          ...mockRelease,
          attention: 'low' as const
        }

        store.releases = [lowAttentionRelease]

        expect(store.shouldShowToast).toBe(false)
      })

      it('should show red dot for new versions', () => {
        const store = useReleaseStore()
        store.releases = [mockRelease]
        vi.mocked(compare).mockReturnValue(1)

        expect(store.shouldShowRedDot).toBe(true)
      })

      it('should show popup for latest version', () => {
        const store = useReleaseStore()
        store.releases = [mockRelease]
        const systemStatsStore = useSystemStatsStore()
        systemStatsStore.systemStats!.system.comfyui_version = '1.2.0'

        vi.mocked(compare).mockReturnValue(0)

        expect(store.shouldShowPopup).toBe(true)
      })

      it('should fetch releases during initialization', async () => {
        const store = useReleaseStore()
        const releaseService = useReleaseService()
        vi.mocked(releaseService.getReleases).mockResolvedValue([mockRelease])

        await store.initialize()

        expect(releaseService.getReleases).toHaveBeenCalledWith({
          project: 'comfyui',
          current_version: '1.0.0',
          form_factor: 'git-windows',
          locale: 'en'
        })
      })
    })

    describe('when notifications are disabled', () => {
      beforeEach(() => {
        const settingStore = useSettingStore()
        vi.mocked(settingStore.get).mockImplementation((key: string) => {
          if (key === 'Comfy.Notification.ShowVersionUpdates') return false
          return null
        })
      })

      it('should not show toast even with new version available', () => {
        const store = useReleaseStore()
        store.releases = [mockRelease]
        vi.mocked(compare).mockReturnValue(1)

        expect(store.shouldShowToast).toBe(false)
      })

      it('should not show red dot even with new version available', () => {
        const store = useReleaseStore()
        store.releases = [mockRelease]
        vi.mocked(compare).mockReturnValue(1)

        expect(store.shouldShowRedDot).toBe(false)
      })

      it('should not show popup even for latest version', () => {
        const store = useReleaseStore()
        store.releases = [mockRelease]
        const systemStatsStore = useSystemStatsStore()
        systemStatsStore.systemStats!.system.comfyui_version = '1.2.0'

        vi.mocked(compare).mockReturnValue(0)

        expect(store.shouldShowPopup).toBe(false)
      })

      it('should skip fetching releases during initialization', async () => {
        const store = useReleaseStore()
        const releaseService = useReleaseService()
        await store.initialize()

        expect(releaseService.getReleases).not.toHaveBeenCalled()
      })

      it('should not fetch releases when calling fetchReleases directly', async () => {
        const store = useReleaseStore()
        const releaseService = useReleaseService()
        await store.fetchReleases()

        expect(releaseService.getReleases).not.toHaveBeenCalled()
        expect(store.isLoading).toBe(false)
      })
    })
  })

  describe('release initialization', () => {
    it('should fetch releases successfully', async () => {
      const store = useReleaseStore()
      const releaseService = useReleaseService()
      vi.mocked(releaseService.getReleases).mockResolvedValue([mockRelease])

      await store.initialize()

      expect(releaseService.getReleases).toHaveBeenCalledWith({
        project: 'comfyui',
        current_version: '1.0.0',
        form_factor: 'git-windows',
        locale: 'en'
      })
      expect(store.releases).toEqual([mockRelease])
    })

    it('should include form_factor in API call', async () => {
      const store = useReleaseStore()
      const releaseService = useReleaseService()
      const systemStatsStore = useSystemStatsStore()
      vi.mocked(systemStatsStore.getFormFactor).mockReturnValue('desktop-mac')
      vi.mocked(releaseService.getReleases).mockResolvedValue([mockRelease])

      await store.initialize()

      expect(releaseService.getReleases).toHaveBeenCalledWith({
        project: 'comfyui',
        current_version: '1.0.0',
        form_factor: 'desktop-mac',
        locale: 'en'
      })
    })

    it('should skip fetching when --disable-api-nodes is present', async () => {
      const store = useReleaseStore()
      const releaseService = useReleaseService()
      const systemStatsStore = useSystemStatsStore()
      systemStatsStore.systemStats!.system.argv = ['--disable-api-nodes']

      await store.initialize()

      expect(releaseService.getReleases).not.toHaveBeenCalled()
      expect(store.isLoading).toBe(false)
    })

    it('should skip fetching when --disable-api-nodes is one of multiple args', async () => {
      const store = useReleaseStore()
      const releaseService = useReleaseService()
      const systemStatsStore = useSystemStatsStore()
      systemStatsStore.systemStats!.system.argv = [
        '--port',
        '8080',
        '--disable-api-nodes',
        '--verbose'
      ]

      await store.initialize()

      expect(releaseService.getReleases).not.toHaveBeenCalled()
      expect(store.isLoading).toBe(false)
    })

    it('should fetch normally when --disable-api-nodes is not present', async () => {
      const store = useReleaseStore()
      const releaseService = useReleaseService()
      const systemStatsStore = useSystemStatsStore()
      systemStatsStore.systemStats!.system.argv = [
        '--port',
        '8080',
        '--verbose'
      ]
      vi.mocked(releaseService.getReleases).mockResolvedValue([mockRelease])

      await store.initialize()

      expect(releaseService.getReleases).toHaveBeenCalled()
      expect(store.releases).toEqual([mockRelease])
    })

    it('should fetch normally when argv is undefined', async () => {
      const store = useReleaseStore()
      const releaseService = useReleaseService()
      const systemStatsStore = useSystemStatsStore()
      // TODO: Consider deleting this test since the types have to be violated for it to be relevant
      delete (systemStatsStore.systemStats!.system as { argv?: string[] }).argv
      vi.mocked(releaseService.getReleases).mockResolvedValue([mockRelease])

      await store.initialize()

      expect(releaseService.getReleases).toHaveBeenCalled()
      expect(store.releases).toEqual([mockRelease])
    })

    it('should handle API errors gracefully', async () => {
      const store = useReleaseStore()
      const releaseService = useReleaseService()
      vi.mocked(releaseService.getReleases).mockResolvedValue(null)
      releaseService.error.value = 'API Error'

      await store.initialize()

      expect(store.releases).toEqual([])
      expect(store.error).toBe('API Error')
    })

    it('should handle non-Error objects', async () => {
      const store = useReleaseStore()
      const releaseService = useReleaseService()
      vi.mocked(releaseService.getReleases).mockRejectedValue('String error')

      await store.initialize()

      expect(store.error).toBe('Unknown error occurred')
    })

    it('should set loading state correctly', async () => {
      const store = useReleaseStore()
      const releaseService = useReleaseService()
      let resolvePromise: (value: ReleaseNote[] | null) => void
      const promise = new Promise<ReleaseNote[] | null>((resolve) => {
        resolvePromise = resolve
      })

      vi.mocked(releaseService.getReleases).mockReturnValue(promise)

      const initPromise = store.initialize()
      expect(store.isLoading).toBe(true)

      resolvePromise!([mockRelease])
      await initPromise

      expect(store.isLoading).toBe(false)
    })

    it('should fetch system stats if not available', async () => {
      const store = useReleaseStore()
      const releaseService = useReleaseService()
      const systemStatsStore = useSystemStatsStore()
      systemStatsStore.systemStats = null
      systemStatsStore.isInitialized = false
      vi.mocked(releaseService.getReleases).mockResolvedValue([mockRelease])

      await store.initialize()

      expect(vi.mocked(until)).toHaveBeenCalled()
      expect(releaseService.getReleases).toHaveBeenCalled()
    })

    it('should not set loading state when notifications disabled', async () => {
      const store = useReleaseStore()
      const settingStore = useSettingStore()
      vi.mocked(settingStore.get).mockImplementation((key: string) => {
        if (key === 'Comfy.Notification.ShowVersionUpdates') return false
        return null
      })

      await store.initialize()

      expect(store.isLoading).toBe(false)
    })
  })

  describe('--disable-api-nodes argument handling', () => {
    it('should skip fetchReleases when --disable-api-nodes is present', async () => {
      const store = useReleaseStore()
      const releaseService = useReleaseService()
      const systemStatsStore = useSystemStatsStore()
      systemStatsStore.systemStats!.system.argv = ['--disable-api-nodes']

      await store.fetchReleases()

      expect(releaseService.getReleases).not.toHaveBeenCalled()
      expect(store.isLoading).toBe(false)
    })

    it('should skip fetchReleases when --disable-api-nodes is among other args', async () => {
      const store = useReleaseStore()
      const releaseService = useReleaseService()
      const systemStatsStore = useSystemStatsStore()
      systemStatsStore.systemStats!.system.argv = [
        '--port',
        '8080',
        '--disable-api-nodes',
        '--verbose'
      ]

      await store.fetchReleases()

      expect(releaseService.getReleases).not.toHaveBeenCalled()
      expect(store.isLoading).toBe(false)
    })

    it('should proceed with fetchReleases when --disable-api-nodes is not present', async () => {
      const store = useReleaseStore()
      const releaseService = useReleaseService()
      const systemStatsStore = useSystemStatsStore()
      systemStatsStore.systemStats!.system.argv = [
        '--port',
        '8080',
        '--verbose'
      ]
      vi.mocked(releaseService.getReleases).mockResolvedValue([mockRelease])

      await store.fetchReleases()

      expect(releaseService.getReleases).toHaveBeenCalled()
    })

    it('should proceed with fetchReleases when argv is undefined', async () => {
      const store = useReleaseStore()
      const releaseService = useReleaseService()
      const systemStatsStore = useSystemStatsStore()
      delete (systemStatsStore.systemStats!.system as { argv?: string[] }).argv
      vi.mocked(releaseService.getReleases).mockResolvedValue([mockRelease])

      await store.fetchReleases()

      expect(releaseService.getReleases).toHaveBeenCalled()
    })

    it('should proceed with fetchReleases when system stats are not available', async () => {
      const store = useReleaseStore()
      const releaseService = useReleaseService()
      const systemStatsStore = useSystemStatsStore()
      systemStatsStore.systemStats = null
      systemStatsStore.isInitialized = false
      vi.mocked(releaseService.getReleases).mockResolvedValue([mockRelease])

      await store.fetchReleases()

      expect(until).toHaveBeenCalled()
      expect(releaseService.getReleases).toHaveBeenCalled()
    })
  })

  describe('action handlers', () => {
    it('should handle skip release', async () => {
      const store = useReleaseStore()
      store.releases = [mockRelease]
      const settingStore = useSettingStore()
      await store.handleSkipRelease('1.2.0')

      expect(settingStore.setMany).toHaveBeenCalledWith({
        'Comfy.Release.Version': '1.2.0',
        'Comfy.Release.Status': 'skipped',
        'Comfy.Release.Timestamp': expect.any(Number)
      })
    })

    it('should handle show changelog', async () => {
      const store = useReleaseStore()
      store.releases = [mockRelease]
      const settingStore = useSettingStore()
      await store.handleShowChangelog('1.2.0')

      expect(settingStore.setMany).toHaveBeenCalledWith({
        'Comfy.Release.Version': '1.2.0',
        'Comfy.Release.Status': 'changelog seen',
        'Comfy.Release.Timestamp': expect.any(Number)
      })
    })

    it('should handle whats new seen', async () => {
      const store = useReleaseStore()
      store.releases = [mockRelease]
      const settingStore = useSettingStore()
      await store.handleWhatsNewSeen('1.2.0')

      expect(settingStore.setMany).toHaveBeenCalledWith({
        'Comfy.Release.Version': '1.2.0',
        'Comfy.Release.Status': "what's new seen",
        'Comfy.Release.Timestamp': expect.any(Number)
      })
    })
  })

  describe('popup visibility', () => {
    it('should show toast for medium/high attention releases', () => {
      const store = useReleaseStore()
      const settingStore = useSettingStore()
      vi.mocked(settingStore.get).mockImplementation((key: string) => {
        if (key === 'Comfy.Release.Version') return null
        if (key === 'Comfy.Release.Status') return null
        if (key === 'Comfy.Notification.ShowVersionUpdates') return true
        return null
      })

      vi.mocked(compare).mockReturnValue(1)

      store.releases = [mockRelease]

      expect(store.shouldShowToast).toBe(true)
    })

    it('should show red dot for new versions', () => {
      const store = useReleaseStore()
      const settingStore = useSettingStore()
      vi.mocked(compare).mockReturnValue(1)
      vi.mocked(settingStore.get).mockImplementation((key: string) => {
        if (key === 'Comfy.Notification.ShowVersionUpdates') return true
        return null
      })

      store.releases = [mockRelease]

      expect(store.shouldShowRedDot).toBe(true)
    })

    it('should show popup for latest version', () => {
      const store = useReleaseStore()
      const systemStatsStore = useSystemStatsStore()
      const settingStore = useSettingStore()
      systemStatsStore.systemStats!.system.comfyui_version = '1.2.0' // Same as release
      vi.mocked(settingStore.get).mockImplementation((key: string) => {
        if (key === 'Comfy.Notification.ShowVersionUpdates') return true
        return null
      })

      vi.mocked(compare).mockReturnValue(0) // versions are equal (latest version)

      store.releases = [mockRelease]

      expect(store.shouldShowPopup).toBe(true)
    })
  })

  describe('edge cases', () => {
    it('should handle missing system stats gracefully', async () => {
      const store = useReleaseStore()
      const systemStatsStore = useSystemStatsStore()
      const settingStore = useSettingStore()
      systemStatsStore.systemStats = null
      vi.mocked(settingStore.get).mockImplementation((key: string) => {
        if (key === 'Comfy.Notification.ShowVersionUpdates') return false
        return null
      })

      await store.initialize()

      // Should not fetch system stats when notifications disabled
      expect(systemStatsStore.refetchSystemStats).not.toHaveBeenCalled()
    })

    it('should handle concurrent fetchReleases calls', async () => {
      const store = useReleaseStore()
      const releaseService = useReleaseService()
      vi.mocked(releaseService.getReleases).mockImplementation(
        () =>
          new Promise((resolve) =>
            setTimeout(() => resolve([mockRelease]), 100)
          )
      )

      // Start two concurrent calls
      const promise1 = store.fetchReleases()
      const promise2 = store.fetchReleases()

      await Promise.all([promise1, promise2])

      // Should only call API once due to loading check
      expect(releaseService.getReleases).toHaveBeenCalledTimes(1)
    })
  })

  describe('isDesktop environment checks', () => {
    describe('when running on desktop', () => {
      beforeEach(() => {
        mockData.isDesktop = true
      })

      it('should show toast when conditions are met', () => {
        const store = useReleaseStore()
        store.releases = [mockRelease]
        vi.mocked(compare).mockReturnValue(1)

        expect(store.shouldShowToast).toBe(true)
      })

      it('should show red dot when new version available', () => {
        const store = useReleaseStore()
        store.releases = [mockRelease]
        vi.mocked(compare).mockReturnValue(1)

        expect(store.shouldShowRedDot).toBe(true)
      })

      it('should show popup for latest version', () => {
        const store = useReleaseStore()
        store.releases = [mockRelease]
        const systemStatsStore = useSystemStatsStore()
        systemStatsStore.systemStats!.system.comfyui_version = '1.2.0'

        vi.mocked(compare).mockReturnValue(0)

        expect(store.shouldShowPopup).toBe(true)
      })
    })

    describe('when NOT running on desktop (web)', () => {
      beforeEach(() => {
        mockData.isDesktop = false
      })

      it('should NOT show toast even when all other conditions are met', () => {
        const store = useReleaseStore()
        vi.mocked(compare).mockReturnValue(1)

        // Set up all conditions that would normally show toast
        store.releases = [mockRelease]

        expect(store.shouldShowToast).toBe(false)
      })

      it('should NOT show red dot even when new version available', () => {
        const store = useReleaseStore()
        store.releases = [mockRelease]
        vi.mocked(compare).mockReturnValue(1)

        expect(store.shouldShowRedDot).toBe(false)
      })

      it('should NOT show toast regardless of attention level', () => {
        const store = useReleaseStore()
        vi.mocked(compare).mockReturnValue(1)

        // Test with high attention releases
        const highRelease = {
          ...mockRelease,
          id: 2,
          attention: 'high' as const
        }
        const mediumRelease = {
          ...mockRelease,
          id: 3,
          attention: 'medium' as const
        }
        store.releases = [highRelease, mediumRelease]

        expect(store.shouldShowToast).toBe(false)
      })

      it('should NOT show red dot even with high attention release', () => {
        const store = useReleaseStore()
        vi.mocked(compare).mockReturnValue(1)

        store.releases = [{ ...mockRelease, attention: 'high' as const }]

        expect(store.shouldShowRedDot).toBe(false)
      })

      it('should NOT show popup even for latest version', () => {
        const store = useReleaseStore()
        store.releases = [mockRelease]
        const systemStatsStore = useSystemStatsStore()
        systemStatsStore.systemStats!.system.comfyui_version = '1.2.0'

        vi.mocked(compare).mockReturnValue(0)

        expect(store.shouldShowPopup).toBe(false)
      })
    })
  })
})
