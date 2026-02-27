import { reactive } from 'vue'

interface CachedMedia {
  src: string
  blob?: Blob
  objectUrl?: string
  error?: boolean
  isLoading: boolean
  lastAccessed: number
}

interface MediaCacheOptions {
  maxSize?: number
  maxAge?: number // in milliseconds
  preloadDistance?: number // pixels from viewport
}

class MediaCacheService {
  public cache = reactive(new Map<string, CachedMedia>())
  private readonly maxSize: number
  private readonly maxAge: number
  private cleanupInterval: number | null = null
  private urlRefCount = new Map<string, number>()

  constructor(options: MediaCacheOptions = {}) {
    this.maxSize = options.maxSize ?? 100
    this.maxAge = options.maxAge ?? 30 * 60 * 1000 // 30 minutes

    // Start cleanup interval
    this.startCleanupInterval()
  }

  private startCleanupInterval() {
    // Clean up every 5 minutes
    this.cleanupInterval = window.setInterval(
      () => {
        this.cleanup()
      },
      5 * 60 * 1000
    )
  }

  private cleanup() {
    const now = Date.now()
    const keysToDelete: string[] = []

    // Find expired entries
    for (const [key, entry] of Array.from(this.cache.entries())) {
      if (now - entry.lastAccessed > this.maxAge) {
        // Only revoke object URL if no components are using it
        if (entry.objectUrl) {
          const refCount = this.urlRefCount.get(entry.objectUrl) || 0
          if (refCount === 0) {
            URL.revokeObjectURL(entry.objectUrl)
            this.urlRefCount.delete(entry.objectUrl)
            keysToDelete.push(key)
          }
          // Don't delete cache entry if URL is still in use
        } else {
          keysToDelete.push(key)
        }
      }
    }

    // Remove expired entries
    keysToDelete.forEach((key) => this.cache.delete(key))

    // If still over size limit, remove oldest entries that aren't in use
    if (this.cache.size > this.maxSize) {
      const entries = Array.from(this.cache.entries())
      entries.sort((a, b) => a[1].lastAccessed - b[1].lastAccessed)

      let removedCount = 0
      const targetRemoveCount = this.cache.size - this.maxSize

      for (const [key, entry] of entries) {
        if (removedCount >= targetRemoveCount) break

        if (entry.objectUrl) {
          const refCount = this.urlRefCount.get(entry.objectUrl) || 0
          if (refCount === 0) {
            URL.revokeObjectURL(entry.objectUrl)
            this.urlRefCount.delete(entry.objectUrl)
            this.cache.delete(key)
            removedCount++
          }
        } else {
          this.cache.delete(key)
          removedCount++
        }
      }
    }
  }

  async getCachedMedia(src: string): Promise<CachedMedia> {
    let entry = this.cache.get(src)

    if (entry) {
      // Update last accessed time
      entry.lastAccessed = Date.now()
      return entry
    }

    // Create new entry
    entry = {
      src,
      isLoading: true,
      lastAccessed: Date.now()
    }

    // Update cache with loading entry
    this.cache.set(src, entry)

    try {
      // Fetch the media
      const response = await fetch(src, { cache: 'force-cache' })
      if (!response.ok) {
        throw new Error(`Failed to fetch: ${response.status}`)
      }

      const blob = await response.blob()
      const objectUrl = URL.createObjectURL(blob)

      // Update entry with successful result
      const updatedEntry: CachedMedia = {
        src,
        blob,
        objectUrl,
        isLoading: false,
        lastAccessed: Date.now()
      }

      this.cache.set(src, updatedEntry)
      return updatedEntry
    } catch (error) {
      console.warn('Failed to cache media:', src, error)

      // Update entry with error
      const errorEntry: CachedMedia = {
        src,
        error: true,
        isLoading: false,
        lastAccessed: Date.now()
      }

      this.cache.set(src, errorEntry)
      return errorEntry
    }
  }

  acquireUrl(src: string): string | undefined {
    const entry = this.cache.get(src)
    if (entry?.objectUrl) {
      const currentCount = this.urlRefCount.get(entry.objectUrl) || 0
      this.urlRefCount.set(entry.objectUrl, currentCount + 1)
      return entry.objectUrl
    }
    return undefined
  }

  releaseUrl(src: string): void {
    const entry = this.cache.get(src)
    if (entry?.objectUrl) {
      const count = (this.urlRefCount.get(entry.objectUrl) || 1) - 1
      if (count <= 0) {
        URL.revokeObjectURL(entry.objectUrl)
        this.urlRefCount.delete(entry.objectUrl)
        // Remove from cache as well
        this.cache.delete(src)
      } else {
        this.urlRefCount.set(entry.objectUrl, count)
      }
    }
  }

  clearCache() {
    // Revoke all object URLs
    for (const entry of Array.from(this.cache.values())) {
      if (entry.objectUrl) {
        URL.revokeObjectURL(entry.objectUrl)
      }
    }
    this.cache.clear()
    this.urlRefCount.clear()
  }

  destroy() {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval)
      this.cleanupInterval = null
    }
    this.clearCache()
  }
}

// Global instance
let mediaCacheInstance: MediaCacheService | null = null

export function useMediaCache(options?: MediaCacheOptions) {
  if (!mediaCacheInstance) {
    mediaCacheInstance = new MediaCacheService(options)
  }

  const getCachedMedia = (src: string) =>
    mediaCacheInstance!.getCachedMedia(src)
  const clearCache = () => mediaCacheInstance!.clearCache()
  const acquireUrl = (src: string) => mediaCacheInstance!.acquireUrl(src)
  const releaseUrl = (src: string) => mediaCacheInstance!.releaseUrl(src)

  return {
    getCachedMedia,
    clearCache,
    acquireUrl,
    releaseUrl,
    cache: mediaCacheInstance.cache
  }
}

// Cleanup on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    if (mediaCacheInstance) {
      mediaCacheInstance.destroy()
    }
  })
}
