import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

import type { AssetMeta } from '../schemas/mediaAssetSchema'
import MediaVideoTop from './MediaVideoTop.vue'

function createVideoAsset(
  src: string,
  mimeType: AssetMeta['mime_type'] = 'video/mp4'
): AssetMeta {
  return {
    id: 'video-1',
    name: 'clip.mp4',
    asset_hash: null,
    mime_type: mimeType,
    tags: [],
    kind: 'video',
    src
  }
}

describe('MediaVideoTop', () => {
  it('renders playable video with darkened paused overlay and play icon', () => {
    const wrapper = mount(MediaVideoTop, {
      props: {
        asset: createVideoAsset('https://example.com/thumb.jpg')
      }
    })

    const video = wrapper.find('video')
    const videoElement = video.element as HTMLVideoElement
    expect(video.exists()).toBe(true)
    expect(videoElement.controls).toBe(false)
    expect(wrapper.find('source').attributes('src')).toBe(
      'https://example.com/thumb.jpg'
    )
    expect(wrapper.find('source').attributes('type')).toBe('video/mp4')
    expect(wrapper.find('.bg-black\\/15').exists()).toBe(true)
    expect(wrapper.find('.icon-\\[lucide--play\\]').exists()).toBe(true)
  })

  it('does not render source element when src is empty', () => {
    const wrapper = mount(MediaVideoTop, {
      props: {
        asset: createVideoAsset('')
      }
    })

    expect(wrapper.find('video').exists()).toBe(true)
    expect(wrapper.find('source').exists()).toBe(false)
  })
  it('emits playback events and hides paused overlay while playing', async () => {
    const wrapper = mount(MediaVideoTop, {
      props: {
        asset: createVideoAsset('https://example.com/thumb.jpg')
      }
    })

    const video = wrapper.find('video')
    const videoElement = video.element as HTMLVideoElement
    expect(video.exists()).toBe(true)

    await video.trigger('play')
    expect(wrapper.emitted('videoPlayingStateChanged')?.at(-1)).toEqual([true])
    expect(wrapper.find('.bg-black\\/15').exists()).toBe(false)

    await wrapper.trigger('mouseenter')
    expect(videoElement.controls).toBe(true)

    await wrapper.trigger('mouseleave')
    expect(videoElement.controls).toBe(false)

    await video.trigger('pause')
    expect(wrapper.emitted('videoPlayingStateChanged')?.at(-1)).toEqual([false])
    expect(wrapper.find('.bg-black\\/15').exists()).toBe(true)
    expect(videoElement.controls).toBe(false)
  })

  it('starts playback from click when controls are hidden', async () => {
    const wrapper = mount(MediaVideoTop, {
      props: {
        asset: createVideoAsset('https://example.com/thumb.jpg')
      }
    })

    const video = wrapper.find('video')
    const videoElement = video.element as HTMLVideoElement
    const playSpy = vi
      .spyOn(videoElement, 'play')
      .mockImplementation(() => Promise.resolve())

    Object.defineProperty(videoElement, 'paused', {
      value: true,
      configurable: true
    })

    await video.trigger('click')

    expect(playSpy).toHaveBeenCalledTimes(1)
  })
})
