import { mount } from '@vue/test-utils'
import { defineComponent } from 'vue'
import { describe, expect, it, vi } from 'vitest'

import type { OutputStackListItem } from '@/platform/assets/composables/useOutputStacks'
import type { AssetItem } from '@/platform/assets/schemas/assetSchema'

import AssetsSidebarListView from './AssetsSidebarListView.vue'

vi.mock('vue-i18n', () => ({
  useI18n: () => ({
    t: (key: string) => key
  })
}))

vi.mock('@/stores/assetsStore', () => ({
  useAssetsStore: () => ({
    isAssetDeleting: () => false
  })
}))

const VirtualGridStub = defineComponent({
  name: 'VirtualGrid',
  props: {
    items: {
      type: Array,
      default: () => []
    }
  },
  template:
    '<div><slot v-for="item in items" :key="item.key" name="item" :item="item" /></div>'
})

const buildAsset = (id: string, name: string): AssetItem =>
  ({
    id,
    name,
    tags: []
  }) satisfies AssetItem

const buildOutputItem = (asset: AssetItem): OutputStackListItem => ({
  key: `asset-${asset.id}`,
  asset
})

const mountListView = (assetItems: OutputStackListItem[] = []) =>
  mount(AssetsSidebarListView, {
    props: {
      assetItems,
      selectableAssets: [],
      isSelected: () => false,
      isStackExpanded: () => false,
      toggleStack: async () => {},
      assetType: 'output'
    },
    global: {
      stubs: {
        VirtualGrid: VirtualGridStub
      }
    }
  })

describe('AssetsSidebarListView', () => {
  it('shows generated assets header when there are assets', () => {
    const wrapper = mountListView([buildOutputItem(buildAsset('a1', 'x.png'))])

    expect(wrapper.text()).toContain('sideToolbar.generatedAssetsHeader')
  })

  it('does not show assets header when there are no assets', () => {
    const wrapper = mountListView([])

    expect(wrapper.text()).not.toContain('sideToolbar.generatedAssetsHeader')
  })

  it('marks mp4 assets as video previews', () => {
    const videoAsset = {
      ...buildAsset('video-asset', 'clip.mp4'),
      preview_url: '/api/view/clip.mp4',
      user_metadata: {}
    } satisfies AssetItem

    const wrapper = mountListView([buildOutputItem(videoAsset)])

    const listItems = wrapper.findAllComponents({ name: 'AssetsListItem' })
    const assetListItem = listItems.at(-1)

    expect(assetListItem).toBeDefined()
    expect(assetListItem?.props('previewUrl')).toBe('/api/view/clip.mp4')
    expect(assetListItem?.props('isVideoPreview')).toBe(true)
  })

  it('uses icon fallback for text assets even when preview_url exists', () => {
    const textAsset = {
      ...buildAsset('text-asset', 'notes.txt'),
      preview_url: '/api/view/notes.txt',
      user_metadata: {}
    } satisfies AssetItem

    const wrapper = mountListView([buildOutputItem(textAsset)])

    const listItems = wrapper.findAllComponents({ name: 'AssetsListItem' })
    const assetListItem = listItems.at(-1)

    expect(assetListItem).toBeDefined()
    expect(assetListItem?.props('previewUrl')).toBe('')
    expect(assetListItem?.props('isVideoPreview')).toBe(false)
  })

  it('emits preview-asset when item preview is clicked', async () => {
    const imageAsset = {
      ...buildAsset('image-asset', 'image.png'),
      preview_url: '/api/view/image.png',
      user_metadata: {}
    } satisfies AssetItem

    const wrapper = mountListView([buildOutputItem(imageAsset)])
    const listItems = wrapper.findAllComponents({ name: 'AssetsListItem' })
    const assetListItem = listItems.at(-1)

    expect(assetListItem).toBeDefined()

    assetListItem!.vm.$emit('preview-click')
    await wrapper.vm.$nextTick()

    expect(wrapper.emitted('preview-asset')).toEqual([[imageAsset]])
  })

  it('emits preview-asset when item is double-clicked', async () => {
    const imageAsset = {
      ...buildAsset('image-asset-dbl', 'image.png'),
      preview_url: '/api/view/image.png',
      user_metadata: {}
    } satisfies AssetItem

    const wrapper = mountListView([buildOutputItem(imageAsset)])
    const listItems = wrapper.findAllComponents({ name: 'AssetsListItem' })
    const assetListItem = listItems.at(-1)

    expect(assetListItem).toBeDefined()

    await assetListItem!.trigger('dblclick')
    await wrapper.vm.$nextTick()

    expect(wrapper.emitted('preview-asset')).toEqual([[imageAsset]])
  })
})
