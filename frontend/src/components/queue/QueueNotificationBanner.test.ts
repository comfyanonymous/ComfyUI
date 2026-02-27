import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'
import { createI18n } from 'vue-i18n'

import QueueNotificationBanner from '@/components/queue/QueueNotificationBanner.vue'
import type { QueueNotificationBanner as QueueNotificationBannerItem } from '@/composables/queue/useQueueNotificationBanners'

const i18n = createI18n({
  legacy: false,
  locale: 'en',
  messages: {
    en: {
      queue: {
        jobAddedToQueue: 'Job added to queue',
        jobQueueing: 'Job queueing'
      },
      sideToolbar: {
        queueProgressOverlay: {
          preview: 'Preview',
          jobCompleted: 'Job completed',
          jobFailed: 'Job failed',
          jobsAddedToQueue:
            '{count} job added to queue | {count} jobs added to queue',
          jobsCompleted: '{count} job completed | {count} jobs completed',
          jobsFailed: '{count} job failed | {count} jobs failed'
        }
      }
    }
  }
})

const mountComponent = (notification: QueueNotificationBannerItem) =>
  mount(QueueNotificationBanner, {
    props: { notification },
    global: {
      plugins: [i18n]
    }
  })

describe(QueueNotificationBanner, () => {
  it('renders singular queued message without count prefix', () => {
    const wrapper = mountComponent({
      type: 'queued',
      count: 1
    })

    expect(wrapper.text()).toContain('Job added to queue')
    expect(wrapper.text()).not.toContain('1 job')
  })

  it('renders queued message with pluralization', () => {
    const wrapper = mountComponent({
      type: 'queued',
      count: 2
    })

    expect(wrapper.text()).toContain('2 jobs added to queue')
    expect(wrapper.html()).toContain('icon-[lucide--check]')
  })

  it('renders queued pending message with spinner icon', () => {
    const wrapper = mountComponent({
      type: 'queuedPending',
      count: 1
    })

    expect(wrapper.text()).toContain('Job queueing')
    expect(wrapper.html()).toContain('icon-[lucide--loader-circle]')
    expect(wrapper.html()).toContain('animate-spin')
  })

  it('renders failed message and alert icon', () => {
    const wrapper = mountComponent({
      type: 'failed',
      count: 1
    })

    expect(wrapper.text()).toContain('Job failed')
    expect(wrapper.html()).toContain('icon-[lucide--circle-alert]')
  })

  it('renders completed message with thumbnail preview when provided', () => {
    const wrapper = mountComponent({
      type: 'completed',
      count: 3,
      thumbnailUrls: ['https://example.com/preview.png']
    })

    expect(wrapper.text()).toContain('3 jobs completed')
    const image = wrapper.get('img')
    expect(image.attributes('src')).toBe('https://example.com/preview.png')
    expect(image.attributes('alt')).toBe('Preview')
  })

  it('renders two completion thumbnail previews', () => {
    const wrapper = mountComponent({
      type: 'completed',
      count: 4,
      thumbnailUrls: [
        'https://example.com/preview-1.png',
        'https://example.com/preview-2.png'
      ]
    })

    const images = wrapper.findAll('img')
    expect(images.length).toBe(2)
    expect(images[0].attributes('src')).toBe(
      'https://example.com/preview-1.png'
    )
    expect(images[1].attributes('src')).toBe(
      'https://example.com/preview-2.png'
    )
  })

  it('caps completion thumbnail previews at two', () => {
    const wrapper = mountComponent({
      type: 'completed',
      count: 4,
      thumbnailUrls: [
        'https://example.com/preview-1.png',
        'https://example.com/preview-2.png',
        'https://example.com/preview-3.png',
        'https://example.com/preview-4.png'
      ]
    })

    const images = wrapper.findAll('img')
    expect(images.length).toBe(2)
    expect(images[0].attributes('src')).toBe(
      'https://example.com/preview-1.png'
    )
    expect(images[1].attributes('src')).toBe(
      'https://example.com/preview-2.png'
    )
  })
})
