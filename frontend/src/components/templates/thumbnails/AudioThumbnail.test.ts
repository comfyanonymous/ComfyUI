import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'

import AudioThumbnail from '@/components/templates/thumbnails/AudioThumbnail.vue'

vi.mock('@/components/templates/thumbnails/BaseThumbnail.vue', () => ({
  default: {
    name: 'BaseThumbnail',
    template: '<div class="base-thumbnail"><slot /></div>'
  }
}))

describe('AudioThumbnail', () => {
  const mountThumbnail = (props = {}) => {
    return mount(AudioThumbnail, {
      props: {
        src: '/test-audio.mp3',
        ...props
      }
    })
  }

  it('renders an audio element with correct src', () => {
    const wrapper = mountThumbnail()
    const audio = wrapper.find('audio')
    expect(audio.exists()).toBe(true)
    expect(audio.attributes('src')).toBe('/test-audio.mp3')
  })

  it('uses BaseThumbnail as container', () => {
    const wrapper = mountThumbnail()
    const baseThumbnail = wrapper.findComponent({ name: 'BaseThumbnail' })
    expect(baseThumbnail.exists()).toBe(true)
  })
})
