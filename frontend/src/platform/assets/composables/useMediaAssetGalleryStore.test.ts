import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { ResultItemImpl } from '@/stores/queueStore'

import type { AssetMeta } from '../schemas/mediaAssetSchema'
import { useMediaAssetGalleryStore } from './useMediaAssetGalleryStore'

vi.mock('@/stores/queueStore', () => ({
  ResultItemImpl: vi
    .fn<typeof ResultItemImpl>()
    .mockImplementation(function (data) {
      Object.assign(this, {
        ...data,
        url: ''
      })
    })
}))

describe('useMediaAssetGalleryStore', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  describe('openSingle', () => {
    it('should convert AssetMeta to ResultItemImpl format', () => {
      const store = useMediaAssetGalleryStore()
      const mockAsset: AssetMeta = {
        id: 'test-1',
        name: 'test-image.png',
        kind: 'image',
        src: 'https://example.com/image.png',
        size: 1024,
        tags: [],
        created_at: '2025-01-01'
      }

      store.openSingle(mockAsset)

      expect(ResultItemImpl).toHaveBeenCalledWith({
        filename: 'test-image.png',
        subfolder: '',
        type: 'output',
        nodeId: '0',
        mediaType: 'images'
      })
      expect(store.items).toHaveLength(1)
      expect(store.activeIndex).toBe(0)
    })

    it('should set correct mediaType for video assets', () => {
      const store = useMediaAssetGalleryStore()
      const mockVideoAsset: AssetMeta = {
        id: 'test-2',
        name: 'test-video.mp4',
        kind: 'video',
        src: 'https://example.com/video.mp4',
        size: 2048,
        tags: [],
        created_at: '2025-01-01'
      }

      store.openSingle(mockVideoAsset)

      expect(ResultItemImpl).toHaveBeenCalledWith(
        expect.objectContaining({
          filename: 'test-video.mp4',
          mediaType: 'video'
        })
      )
    })

    it('should set correct mediaType for audio assets', () => {
      const store = useMediaAssetGalleryStore()
      const mockAudioAsset: AssetMeta = {
        id: 'test-3',
        name: 'test-audio.mp3',
        kind: 'audio',
        src: 'https://example.com/audio.mp3',
        size: 512,
        tags: [],
        created_at: '2025-01-01'
      }

      store.openSingle(mockAudioAsset)

      expect(ResultItemImpl).toHaveBeenCalledWith(
        expect.objectContaining({
          filename: 'test-audio.mp3',
          mediaType: 'audio'
        })
      )
    })

    it('should override url getter with asset.src', () => {
      const store = useMediaAssetGalleryStore()
      const mockAsset: AssetMeta = {
        id: 'test-4',
        name: 'test.png',
        kind: 'image',
        src: 'https://example.com/custom-url.png',
        size: 1024,
        tags: [],
        created_at: '2025-01-01'
      }

      store.openSingle(mockAsset)

      const resultItem = store.items[0]
      expect(resultItem.url).toBe('https://example.com/custom-url.png')
    })

    it('should handle assets without src gracefully', () => {
      const store = useMediaAssetGalleryStore()
      const mockAsset: AssetMeta = {
        id: 'test-5',
        name: 'no-src.png',
        kind: 'image',
        src: '',
        size: 1024,
        tags: [],
        created_at: '2025-01-01'
      }

      store.openSingle(mockAsset)

      const resultItem = store.items[0]
      expect(resultItem.url).toBe('')
    })

    it('should update activeIndex and items when called multiple times', () => {
      const store = useMediaAssetGalleryStore()
      const asset1: AssetMeta = {
        id: '1',
        name: 'first.png',
        kind: 'image',
        src: 'url1',
        size: 100,
        tags: [],
        created_at: '2025-01-01'
      }
      const asset2: AssetMeta = {
        id: '2',
        name: 'second.png',
        kind: 'image',
        src: 'url2',
        size: 200,
        tags: [],
        created_at: '2025-01-01'
      }

      store.openSingle(asset1)
      expect(store.items).toHaveLength(1)
      expect(store.items[0].filename).toBe('first.png')

      store.openSingle(asset2)
      expect(store.items).toHaveLength(1)
      expect(store.items[0].filename).toBe('second.png')
      expect(store.activeIndex).toBe(0)
    })
  })

  describe('close', () => {
    it('should reset activeIndex to -1', () => {
      const store = useMediaAssetGalleryStore()
      const mockAsset: AssetMeta = {
        id: 'test',
        name: 'test.png',
        kind: 'image',
        src: 'test-url',
        size: 1024,
        tags: [],
        created_at: '2025-01-01'
      }

      store.openSingle(mockAsset)
      expect(store.activeIndex).toBe(0)

      store.close()
      expect(store.activeIndex).toBe(-1)
    })
  })
})
