import axios from 'axios'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useReleaseService } from '@/platform/updates/common/releaseService'

// Hoist the mock to avoid hoisting issues
const mockAxiosInstance = vi.hoisted(() => ({
  get: vi.fn()
}))

vi.mock('axios', () => ({
  default: {
    create: vi.fn(() => mockAxiosInstance),
    isAxiosError: vi.fn()
  }
}))

describe('useReleaseService', () => {
  let service: ReturnType<typeof useReleaseService>

  const mockReleases = [
    {
      id: 1,
      project: 'comfyui' as const,
      version: '1.2.0',
      attention: 'high' as const,
      content: 'New features and improvements',
      published_at: '2023-12-01T00:00:00Z'
    }
  ]

  beforeEach(() => {
    vi.clearAllMocks()
    service = useReleaseService()
  })

  it('should initialize with default state', () => {
    expect(service.isLoading.value).toBe(false)
    expect(service.error.value).toBeNull()
  })

  describe('getReleases', () => {
    it('should fetch releases successfully', async () => {
      mockAxiosInstance.get.mockResolvedValue({ data: mockReleases })

      const result = await service.getReleases({
        project: 'comfyui',
        current_version: '1.0.0'
      })

      expect(mockAxiosInstance.get).toHaveBeenCalledWith('/releases', {
        params: {
          project: 'comfyui',
          current_version: '1.0.0'
        },
        signal: undefined
      })

      expect(result).toEqual(mockReleases)
      expect(service.isLoading.value).toBe(false)
      expect(service.error.value).toBeNull()
    })

    it('should fetch releases with form_factor parameter', async () => {
      mockAxiosInstance.get.mockResolvedValue({ data: mockReleases })

      const result = await service.getReleases({
        project: 'comfyui',
        current_version: '1.0.0',
        form_factor: 'desktop-windows'
      })

      expect(mockAxiosInstance.get).toHaveBeenCalledWith('/releases', {
        params: {
          project: 'comfyui',
          current_version: '1.0.0',
          form_factor: 'desktop-windows'
        },
        signal: undefined
      })

      expect(result).toEqual(mockReleases)
    })

    it('should pass abort signal when provided', async () => {
      const abortController = new AbortController()
      mockAxiosInstance.get.mockResolvedValue({ data: mockReleases })

      await service.getReleases({ project: 'comfyui' }, abortController.signal)

      expect(mockAxiosInstance.get).toHaveBeenCalledWith('/releases', {
        params: { project: 'comfyui' },
        signal: abortController.signal
      })
    })

    it('should handle API errors with response', async () => {
      const errorResponse = {
        response: {
          status: 400,
          data: { message: 'Invalid parameters' }
        }
      }
      mockAxiosInstance.get.mockRejectedValue(errorResponse)
      vi.mocked(axios.isAxiosError).mockReturnValue(true)

      const result = await service.getReleases({ project: 'comfyui' })

      expect(result).toBeNull()
      expect(service.error.value).toBe('Invalid project or version parameter')
      expect(service.isLoading.value).toBe(false)
    })

    it('should handle 401 errors', async () => {
      const errorResponse = {
        response: {
          status: 401,
          data: { message: 'Unauthorized' }
        }
      }
      mockAxiosInstance.get.mockRejectedValue(errorResponse)

      const result = await service.getReleases({ project: 'comfyui' })

      expect(result).toBeNull()
      expect(service.error.value).toBe('Unauthorized: Authentication required')
    })

    it('should handle 404 errors', async () => {
      const errorResponse = {
        response: {
          status: 404,
          data: { message: 'Not found' }
        }
      }
      mockAxiosInstance.get.mockRejectedValue(errorResponse)

      const result = await service.getReleases({ project: 'comfyui' })

      expect(result).toBeNull()
      expect(service.error.value).toBe('Not found: Not found')
    })

    it('should handle 500 errors', async () => {
      const errorResponse = {
        response: {
          status: 500,
          data: { message: 'Server error' }
        }
      }
      mockAxiosInstance.get.mockRejectedValue(errorResponse)

      const result = await service.getReleases({ project: 'comfyui' })

      expect(result).toBeNull()
      expect(service.error.value).toBe('Server error: Server error')
    })

    it('should handle network errors', async () => {
      const networkError = new Error('Network Error')
      mockAxiosInstance.get.mockRejectedValue(networkError)

      const result = await service.getReleases({ project: 'comfyui' })

      expect(result).toBeNull()
      expect(service.error.value).toBe('Failed to get releases: Network Error')
    })

    it('should handle abort errors gracefully', async () => {
      const abortError = {
        name: 'AbortError',
        message: 'Request aborted'
      }
      mockAxiosInstance.get.mockRejectedValue(abortError)

      const result = await service.getReleases({ project: 'comfyui' })

      expect(result).toBeNull()
      expect(service.error.value).toContain('Request aborted') // Abort errors are handled
    })

    it('should handle non-Error objects', async () => {
      const stringError = 'String error'
      mockAxiosInstance.get.mockRejectedValue(stringError)

      const result = await service.getReleases({ project: 'comfyui' })

      expect(result).toBeNull()
      expect(service.error.value).toBe('Failed to get releases: undefined')
    })

    it('should set loading state correctly', async () => {
      let resolvePromise: (value: unknown) => void
      const promise = new Promise((resolve) => {
        resolvePromise = resolve
      })
      mockAxiosInstance.get.mockReturnValue(promise)

      const fetchPromise = service.getReleases({ project: 'comfyui' })
      expect(service.isLoading.value).toBe(true)

      resolvePromise!({ data: mockReleases })
      await fetchPromise

      expect(service.isLoading.value).toBe(false)
    })

    it('should reset error state on new request', async () => {
      // First request fails
      mockAxiosInstance.get.mockRejectedValueOnce(new Error('First error'))
      await service.getReleases({ project: 'comfyui' })
      expect(service.error.value).toBe('Failed to get releases: First error')

      // Second request succeeds
      mockAxiosInstance.get.mockResolvedValueOnce({ data: mockReleases })
      await service.getReleases({ project: 'comfyui' })
      expect(service.error.value).toBeNull()
    })
  })
})
