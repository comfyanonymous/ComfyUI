import { beforeEach, describe, expect, it, vi } from 'vitest'

import { api } from '@/scripts/api'

import { refreshRemoteConfig } from './refreshRemoteConfig'
import { remoteConfig } from './remoteConfig'

vi.mock('@/scripts/api', () => ({
  api: {
    fetchApi: vi.fn()
  }
}))

vi.stubGlobal('fetch', vi.fn())

describe('refreshRemoteConfig', () => {
  const mockConfig = { feature1: true, feature2: 'value' }

  function mockSuccessResponse(config = mockConfig) {
    return {
      ok: true,
      json: async () => config
    } as Response
  }

  function mockErrorResponse(status: number, statusText: string) {
    return {
      ok: false,
      status,
      statusText
    } as Response
  }

  beforeEach(() => {
    vi.clearAllMocks()
    remoteConfig.value = {}
    window.__CONFIG__ = {}
  })

  describe('with auth (default)', () => {
    it('uses api.fetchApi when useAuth is true', async () => {
      vi.mocked(api.fetchApi).mockResolvedValue(mockSuccessResponse())

      await refreshRemoteConfig({ useAuth: true })

      expect(api.fetchApi).toHaveBeenCalledWith('/features', {
        cache: 'no-store'
      })
      expect(global.fetch).not.toHaveBeenCalled()
      expect(remoteConfig.value).toEqual(mockConfig)
      expect(window.__CONFIG__).toEqual(mockConfig)
    })

    it('uses api.fetchApi by default', async () => {
      vi.mocked(api.fetchApi).mockResolvedValue(mockSuccessResponse())

      await refreshRemoteConfig()

      expect(api.fetchApi).toHaveBeenCalled()
      expect(global.fetch).not.toHaveBeenCalled()
    })
  })

  describe('without auth', () => {
    it('uses raw fetch when useAuth is false', async () => {
      vi.mocked(global.fetch).mockResolvedValue(mockSuccessResponse())

      await refreshRemoteConfig({ useAuth: false })

      expect(global.fetch).toHaveBeenCalledWith('/api/features', {
        cache: 'no-store'
      })
      expect(api.fetchApi).not.toHaveBeenCalled()
      expect(remoteConfig.value).toEqual(mockConfig)
      expect(window.__CONFIG__).toEqual(mockConfig)
    })
  })

  describe('error handling', () => {
    it('clears config on 401 response', async () => {
      vi.mocked(api.fetchApi).mockResolvedValue(
        mockErrorResponse(401, 'Unauthorized')
      )

      await refreshRemoteConfig()

      expect(remoteConfig.value).toEqual({})
      expect(window.__CONFIG__).toEqual({})
    })

    it('clears config on 403 response', async () => {
      vi.mocked(api.fetchApi).mockResolvedValue(
        mockErrorResponse(403, 'Forbidden')
      )

      await refreshRemoteConfig()

      expect(remoteConfig.value).toEqual({})
      expect(window.__CONFIG__).toEqual({})
    })

    it('clears config on fetch error', async () => {
      vi.mocked(api.fetchApi).mockRejectedValue(new Error('Network error'))

      await refreshRemoteConfig()

      expect(remoteConfig.value).toEqual({})
      expect(window.__CONFIG__).toEqual({})
    })

    it('preserves config on 500 response', async () => {
      const existingConfig = { subscription_required: true }
      remoteConfig.value = existingConfig
      window.__CONFIG__ = existingConfig

      vi.mocked(api.fetchApi).mockResolvedValue(
        mockErrorResponse(500, 'Internal Server Error')
      )

      await refreshRemoteConfig()

      expect(remoteConfig.value).toEqual(existingConfig)
      expect(window.__CONFIG__).toEqual(existingConfig)
    })
  })
})
