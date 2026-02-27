import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { SystemStats } from '@/schemas/apiSchema'
import { api } from '@/scripts/api'
import { useSystemStatsStore } from '@/stores/systemStatsStore'

const mockData = vi.hoisted(() => ({ isDesktop: false }))

// Mock the API
vi.mock('@/scripts/api', () => ({
  api: {
    getSystemStats: vi.fn()
  }
}))

vi.mock('@/platform/distribution/types', () => ({
  get isDesktop() {
    return mockData.isDesktop
  },
  isCloud: false
}))

describe('useSystemStatsStore', () => {
  let store: ReturnType<typeof useSystemStatsStore>

  beforeEach(() => {
    // Mock API to prevent automatic fetch on store creation
    vi.mocked(api.getSystemStats).mockResolvedValue(null!)
    setActivePinia(createTestingPinia({ stubActions: false }))
    store = useSystemStatsStore()
    vi.clearAllMocks()
  })

  it('should initialize and start fetching immediately', async () => {
    // useAsyncState with immediate: true starts loading right away
    // In test environment, the mock resolves immediately so loading might be false already
    expect(store.systemStats).toBeNull() // Initial value is null
    expect(store.error).toBeUndefined()

    // Wait for initial fetch to complete
    await new Promise((resolve) => setTimeout(resolve, 0))
    expect(store.isInitialized).toBe(true) // Should be initialized after fetch
  })

  describe('refetchSystemStats', () => {
    it('should fetch system stats successfully', async () => {
      const mockStats = {
        system: {
          os: 'Windows',
          python_version: '3.10.0',
          embedded_python: false,
          comfyui_version: '1.0.0',
          pytorch_version: '2.0.0',
          required_frontend_version: '1.24.0',
          argv: [],
          ram_total: 16000000000,
          ram_free: 8000000000
        },
        devices: []
      }

      vi.mocked(api.getSystemStats).mockResolvedValue(mockStats)

      await store.refetchSystemStats()

      expect(store.systemStats).toEqual(mockStats)
      expect(store.isLoading).toBe(false)
      expect(store.error).toBeUndefined() // useAsyncState uses undefined for no error
      expect(store.isInitialized).toBe(true)
      expect(api.getSystemStats).toHaveBeenCalled()
    })

    it('should handle API errors', async () => {
      const error = new Error('API Error')
      vi.mocked(api.getSystemStats).mockRejectedValue(error)

      await store.refetchSystemStats()

      expect(store.systemStats).toBeNull() // Initial value stays null on error
      expect(store.isLoading).toBe(false)
      expect(store.error).toEqual(error) // useAsyncState stores the actual error object
    })

    it('should handle non-Error objects', async () => {
      vi.mocked(api.getSystemStats).mockRejectedValue('String error')

      await store.refetchSystemStats()

      expect(store.error).toBe('String error') // useAsyncState stores the actual error
    })

    it('should set loading state correctly', async () => {
      let resolvePromise: (value: SystemStats) => void = () => {}
      const promise = new Promise<SystemStats>((resolve) => {
        resolvePromise = resolve
      })
      vi.mocked(api.getSystemStats).mockReturnValue(promise)

      const fetchPromise = store.refetchSystemStats()
      expect(store.isLoading).toBe(true)

      resolvePromise({} as SystemStats)
      await fetchPromise

      expect(store.isLoading).toBe(false)
    })

    it('should handle system stats updates', async () => {
      const updatedStats = {
        system: {
          os: 'Windows',
          python_version: '3.11.0',
          embedded_python: false,
          comfyui_version: '1.1.0',
          pytorch_version: '2.1.0',
          required_frontend_version: '1.25.0',
          argv: [],
          ram_total: 16000000000,
          ram_free: 7000000000
        },
        devices: []
      }

      vi.mocked(api.getSystemStats).mockResolvedValue(updatedStats)

      await store.refetchSystemStats()

      expect(store.systemStats).toEqual(updatedStats)
      expect(store.isLoading).toBe(false)
      expect(store.error).toBeUndefined()
      expect(store.isInitialized).toBe(true)
      expect(api.getSystemStats).toHaveBeenCalled()
    })
  })

  describe('getFormFactor', () => {
    beforeEach(() => {
      // Reset systemStats for each test
      store.systemStats = null
    })

    it('should return "other" when systemStats is null', () => {
      expect(store.getFormFactor()).toBe('other')
    })

    it('should return "other" when os is not available', () => {
      store.systemStats = {
        system: {
          python_version: '3.10.0',
          embedded_python: false,
          comfyui_version: '1.0.0',
          pytorch_version: '2.0.0',
          argv: [],
          ram_total: 16000000000,
          ram_free: 8000000000
        } as Partial<SystemStats['system']> as SystemStats['system'],
        devices: []
      }

      expect(store.getFormFactor()).toBe('other')
    })

    describe('desktop environment', () => {
      beforeEach(() => {
        mockData.isDesktop = true
      })

      it('should return "desktop-windows" for Windows desktop', () => {
        store.systemStats = {
          system: {
            os: 'Windows 11',
            python_version: '3.10.0',
            embedded_python: false,
            comfyui_version: '1.0.0',
            pytorch_version: '2.0.0',
            argv: [],
            ram_total: 16000000000,
            ram_free: 8000000000
          },
          devices: []
        }

        expect(store.getFormFactor()).toBe('desktop-windows')
      })

      it('should return "desktop-mac" for macOS desktop', () => {
        store.systemStats = {
          system: {
            os: 'Darwin 22.0.0',
            python_version: '3.10.0',
            embedded_python: false,
            comfyui_version: '1.0.0',
            pytorch_version: '2.0.0',
            argv: [],
            ram_total: 16000000000,
            ram_free: 8000000000
          },
          devices: []
        }

        expect(store.getFormFactor()).toBe('desktop-mac')
      })

      it('should return "desktop-mac" for Mac desktop', () => {
        store.systemStats = {
          system: {
            os: 'Mac OS X 13.0',
            python_version: '3.10.0',
            embedded_python: false,
            comfyui_version: '1.0.0',
            pytorch_version: '2.0.0',
            argv: [],
            ram_total: 16000000000,
            ram_free: 8000000000
          },
          devices: []
        }

        expect(store.getFormFactor()).toBe('desktop-mac')
      })

      it('should return "other" for unknown desktop OS', () => {
        store.systemStats = {
          system: {
            os: 'Linux',
            python_version: '3.10.0',
            embedded_python: false,
            comfyui_version: '1.0.0',
            pytorch_version: '2.0.0',
            argv: [],
            ram_total: 16000000000,
            ram_free: 8000000000
          },
          devices: []
        }

        expect(store.getFormFactor()).toBe('other')
      })
    })

    describe('git environment (non-desktop)', () => {
      beforeEach(() => {
        mockData.isDesktop = false
      })

      it('should return "git-windows" for Windows git', () => {
        store.systemStats = {
          system: {
            os: 'Windows 11',
            python_version: '3.10.0',
            embedded_python: false,
            comfyui_version: '1.0.0',
            pytorch_version: '2.0.0',
            argv: [],
            ram_total: 16000000000,
            ram_free: 8000000000
          },
          devices: []
        }

        expect(store.getFormFactor()).toBe('git-windows')
      })

      it('should return "git-mac" for macOS git', () => {
        store.systemStats = {
          system: {
            os: 'Darwin 22.0.0',
            python_version: '3.10.0',
            embedded_python: false,
            comfyui_version: '1.0.0',
            pytorch_version: '2.0.0',
            argv: [],
            ram_total: 16000000000,
            ram_free: 8000000000
          },
          devices: []
        }

        expect(store.getFormFactor()).toBe('git-mac')
      })

      it('should return "git-linux" for Linux git', () => {
        store.systemStats = {
          system: {
            os: 'linux Ubuntu 22.04',
            python_version: '3.10.0',
            embedded_python: false,
            comfyui_version: '1.0.0',
            pytorch_version: '2.0.0',
            argv: [],
            ram_total: 16000000000,
            ram_free: 8000000000
          },
          devices: []
        }

        expect(store.getFormFactor()).toBe('git-linux')
      })

      it('should return "other" for unknown git OS', () => {
        store.systemStats = {
          system: {
            os: 'FreeBSD',
            python_version: '3.10.0',
            embedded_python: false,
            comfyui_version: '1.0.0',
            pytorch_version: '2.0.0',
            argv: [],
            ram_total: 16000000000,
            ram_free: 8000000000
          },
          devices: []
        }

        expect(store.getFormFactor()).toBe('other')
      })
    })

    describe('case insensitive OS detection', () => {
      beforeEach(() => {
        mockData.isDesktop = false
      })

      it('should handle uppercase OS names', () => {
        store.systemStats = {
          system: {
            os: 'WINDOWS',
            python_version: '3.10.0',
            embedded_python: false,
            comfyui_version: '1.0.0',
            pytorch_version: '2.0.0',
            argv: [],
            ram_total: 16000000000,
            ram_free: 8000000000
          },
          devices: []
        }

        expect(store.getFormFactor()).toBe('git-windows')
      })

      it('should handle mixed case OS names', () => {
        store.systemStats = {
          system: {
            os: 'LiNuX',
            python_version: '3.10.0',
            embedded_python: false,
            comfyui_version: '1.0.0',
            pytorch_version: '2.0.0',
            argv: [],
            ram_total: 16000000000,
            ram_free: 8000000000
          },
          devices: []
        }

        expect(store.getFormFactor()).toBe('git-linux')
      })
    })
  })
})
