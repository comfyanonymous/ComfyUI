import { beforeEach, describe, expect, it, vi } from 'vitest'

import { api } from '@/scripts/api'

// Mock global fetch
vi.stubGlobal('fetch', vi.fn())

describe('api.fetchApi', () => {
  beforeEach(() => {
    vi.resetAllMocks()

    // Reset api state
    api.user = 'test-user'
  })

  describe('header handling', () => {
    it('should add Comfy-User header with plain object headers', async () => {
      const mockFetch = vi
        .mocked(global.fetch)
        .mockResolvedValue(new Response())

      await api.fetchApi('/test', {
        headers: {}
      })

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/test'),
        expect.objectContaining({
          headers: {
            'Comfy-User': 'test-user'
          }
        })
      )
    })

    it('should add Comfy-User header with Headers instance', async () => {
      const mockFetch = vi
        .mocked(global.fetch)
        .mockResolvedValue(new Response())
      const headers = new Headers()

      await api.fetchApi('/test', { headers })

      expect(mockFetch).toHaveBeenCalled()
      const callHeaders = mockFetch.mock.calls[0][1]?.headers
      expect(callHeaders).toEqual(headers)
    })

    it('should add Comfy-User header with array headers', async () => {
      const mockFetch = vi
        .mocked(global.fetch)
        .mockResolvedValue(new Response())
      const headers: [string, string][] = []

      await api.fetchApi('/test', { headers })

      expect(mockFetch).toHaveBeenCalled()
      const callHeaders = mockFetch.mock.calls[0][1]?.headers
      expect(callHeaders).toContainEqual(['Comfy-User', 'test-user'])
    })

    it('should preserve existing headers when adding Comfy-User', async () => {
      const mockFetch = vi
        .mocked(global.fetch)
        .mockResolvedValue(new Response())

      await api.fetchApi('/test', {
        headers: {
          'Content-Type': 'application/json',
          'X-Custom': 'value'
        }
      })

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/test'),
        expect.objectContaining({
          headers: {
            'Content-Type': 'application/json',
            'X-Custom': 'value',
            'Comfy-User': 'test-user'
          }
        })
      )
    })

    it('should not allow developer-specified headers to be overridden by options', async () => {
      const mockFetch = vi
        .mocked(global.fetch)
        .mockResolvedValue(new Response())

      await api.fetchApi('/test', {
        headers: {
          'Comfy-User': 'fennec-girl'
        }
      })

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/test'),
        expect.objectContaining({
          headers: {
            'Comfy-User': 'test-user'
          }
        })
      )
    })
  })

  describe('default options', () => {
    it('should set cache to no-cache by default', async () => {
      const mockFetch = vi
        .mocked(global.fetch)
        .mockResolvedValue(new Response())

      await api.fetchApi('/test')

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          cache: 'no-cache'
        })
      )
    })

    it('should include required headers even when no headers option is provided', async () => {
      const mockFetch = vi
        .mocked(global.fetch)
        .mockResolvedValue(new Response())

      await api.fetchApi('/test')

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            'Comfy-User': 'test-user'
          })
        })
      )
    })

    it('should not override existing cache option', async () => {
      const mockFetch = vi
        .mocked(global.fetch)
        .mockResolvedValue(new Response())

      await api.fetchApi('/test', { cache: 'force-cache' })

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          cache: 'force-cache'
        })
      )
    })
  })

  describe('URL construction', () => {
    it('should use apiURL for route construction', async () => {
      const mockFetch = vi
        .mocked(global.fetch)
        .mockResolvedValue(new Response())

      await api.fetchApi('/test/route')

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/test/route'),
        expect.any(Object)
      )
    })
  })
})
