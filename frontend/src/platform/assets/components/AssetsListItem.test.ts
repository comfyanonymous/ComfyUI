import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import AssetsListItem from './AssetsListItem.vue'

describe('AssetsListItem', () => {
  it('renders video element with play overlay for video previews', () => {
    const wrapper = mount(AssetsListItem, {
      props: {
        previewUrl: 'https://example.com/preview.mp4',
        previewAlt: 'clip.mp4',
        isVideoPreview: true
      }
    })

    const video = wrapper.find('video')
    expect(video.exists()).toBe(true)
    expect(video.attributes('src')).toBe('https://example.com/preview.mp4')
    expect(video.attributes('preload')).toBe('metadata')
    expect(wrapper.find('img').exists()).toBe(false)
    expect(wrapper.find('.bg-black\\/15').exists()).toBe(true)
    expect(wrapper.find('.icon-\\[lucide--play\\]').exists()).toBe(true)
  })

  it('does not show play overlay for non-video previews', () => {
    const wrapper = mount(AssetsListItem, {
      props: {
        previewUrl: 'https://example.com/preview.jpg',
        previewAlt: 'image.png',
        isVideoPreview: false
      }
    })

    expect(wrapper.find('img').exists()).toBe(true)
    expect(wrapper.find('video').exists()).toBe(false)
    expect(wrapper.find('.icon-\\[lucide--play\\]').exists()).toBe(false)
  })

  it('emits preview-click when preview is clicked', async () => {
    const wrapper = mount(AssetsListItem, {
      props: {
        previewUrl: 'https://example.com/preview.jpg',
        previewAlt: 'image.png'
      }
    })

    await wrapper.find('img').trigger('click')

    expect(wrapper.emitted('preview-click')).toHaveLength(1)
  })

  it('emits preview-click when fallback icon is clicked', async () => {
    const wrapper = mount(AssetsListItem, {
      props: {
        iconName: 'icon-[lucide--box]'
      }
    })

    await wrapper.find('i').trigger('click')

    expect(wrapper.emitted('preview-click')).toHaveLength(1)
  })
})
