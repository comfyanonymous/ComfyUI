import { mount } from '@vue/test-utils'
import { describe, expect, it, vi } from 'vitest'
import { createI18n } from 'vue-i18n'

import Button from '@/components/ui/button/Button.vue'
import AudioPreviewPlayer from '@/renderer/extensions/vueNodes/widgets/components/audio/AudioPreviewPlayer.vue'

const mockToastAdd = vi.fn()

vi.mock('primevue/usetoast', () => ({
  useToast: () => ({ add: mockToastAdd })
}))

vi.mock('@/base/common/downloadUtil', () => ({
  downloadFile: vi.fn()
}))

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: { en: {} }
})

function mountPlayer(modelValue?: string) {
  return mount(AudioPreviewPlayer, {
    props: {
      modelValue,
      hideWhenEmpty: false
    },
    global: {
      plugins: [i18n],
      components: { Button },
      stubs: {
        TieredMenu: true,
        Slider: true
      }
    }
  })
}

function findDownloadButton(wrapper: ReturnType<typeof mountPlayer>) {
  return wrapper.find('[aria-label="g.downloadAudio"]')
}

describe('AudioPreviewPlayer', () => {
  describe('download button', () => {
    it('shows download button when audio is loaded', () => {
      const wrapper = mountPlayer('http://example.com/audio.mp3')

      expect(findDownloadButton(wrapper).exists()).toBe(true)
    })

    it('hides download button when no audio is loaded', () => {
      const wrapper = mountPlayer()

      expect(findDownloadButton(wrapper).exists()).toBe(false)
    })

    it('calls downloadFile when download button is clicked', async () => {
      const { downloadFile } = await import('@/base/common/downloadUtil')

      const wrapper = mountPlayer('http://example.com/audio.mp3')
      await findDownloadButton(wrapper).trigger('click')

      expect(downloadFile).toHaveBeenCalledWith('http://example.com/audio.mp3')
    })

    it('shows toast on download failure', async () => {
      const { downloadFile } = await import('@/base/common/downloadUtil')
      vi.mocked(downloadFile).mockImplementation(() => {
        throw new Error('download failed')
      })

      const wrapper = mountPlayer('http://example.com/audio.mp3')
      await findDownloadButton(wrapper).trigger('click')

      expect(mockToastAdd).toHaveBeenCalledWith(
        expect.objectContaining({
          severity: 'error'
        })
      )

      vi.mocked(downloadFile).mockReset()
    })
  })
})
