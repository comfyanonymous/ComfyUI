import { defineStore } from 'pinia'
import { ref, shallowRef } from 'vue'

import { ResultItemImpl } from '@/stores/queueStore'

import type { AssetMeta } from '../schemas/mediaAssetSchema'

export const useMediaAssetGalleryStore = defineStore(
  'mediaAssetGallery',
  () => {
    const activeIndex = ref(-1)
    const items = shallowRef<ResultItemImpl[]>([])

    const close = () => {
      activeIndex.value = -1
    }

    const openSingle = (asset: AssetMeta) => {
      // Convert AssetMeta to ResultItemImpl format
      const resultItem = new ResultItemImpl({
        filename: asset.name,
        subfolder: '',
        type: 'output',
        nodeId: '0',
        mediaType: asset.kind === 'image' ? 'images' : asset.kind
      })

      // Override the url getter to use asset.src
      Object.defineProperty(resultItem, 'url', {
        get() {
          return asset.src || ''
        },
        configurable: true
      })

      items.value = [resultItem]
      activeIndex.value = 0
    }

    return {
      activeIndex,
      items,
      close,
      openSingle
    }
  }
)
