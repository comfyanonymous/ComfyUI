import { describe, expect, it, vi } from 'vitest'

import { useMediaCache } from './mediaCacheService'

// Mock fetch
global.fetch = vi.fn()
global.URL = {
  createObjectURL: vi.fn(() => 'blob:mock-url'),
  revokeObjectURL: vi.fn()
} as Partial<typeof URL> as typeof URL

describe('mediaCacheService', () => {
  describe('URL reference counting', () => {
    it('should handle URL acquisition for non-existent cache entry', () => {
      const { acquireUrl } = useMediaCache()

      const url = acquireUrl('non-existent.jpg')
      expect(url).toBeUndefined()
    })

    it('should handle URL release for non-existent cache entry', () => {
      const { releaseUrl } = useMediaCache()

      // Should not throw error
      expect(() => releaseUrl('non-existent.jpg')).not.toThrow()
    })

    it('should provide acquireUrl and releaseUrl methods', () => {
      const cache = useMediaCache()

      expect(typeof cache.acquireUrl).toBe('function')
      expect(typeof cache.releaseUrl).toBe('function')
    })
  })
})
