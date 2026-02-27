import type { Meta, StoryObj } from '@storybook/vue3-vite'

import type { QueueNotificationBanner as QueueNotificationBannerItem } from '@/composables/queue/useQueueNotificationBanners'

import QueueNotificationBanner from './QueueNotificationBanner.vue'

const meta: Meta<typeof QueueNotificationBanner> = {
  title: 'Queue/QueueNotificationBanner',
  component: QueueNotificationBanner,
  parameters: {
    layout: 'padded'
  }
}

export default meta
type Story = StoryObj<typeof meta>

const thumbnail = (hex: string) =>
  `data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='256' height='256'><rect width='256' height='256' fill='%23${hex}'/></svg>`

const args = (notification: QueueNotificationBannerItem) => ({ notification })

export const Queueing: Story = {
  args: args({
    type: 'queuedPending',
    count: 1
  })
}

export const QueueingMultiple: Story = {
  args: args({
    type: 'queuedPending',
    count: 3
  })
}

export const Queued: Story = {
  args: args({
    type: 'queued',
    count: 1
  })
}

export const QueuedMultiple: Story = {
  args: args({
    type: 'queued',
    count: 4
  })
}

export const Completed: Story = {
  args: args({
    type: 'completed',
    count: 1,
    thumbnailUrls: [thumbnail('4dabf7')]
  })
}

export const CompletedMultiple: Story = {
  args: args({
    type: 'completed',
    count: 4
  })
}

export const CompletedMultipleWithThumbnail: Story = {
  args: args({
    type: 'completed',
    count: 4,
    thumbnailUrls: [
      thumbnail('ff6b6b'),
      thumbnail('4dabf7'),
      thumbnail('51cf66')
    ]
  })
}

export const Failed: Story = {
  args: args({
    type: 'failed',
    count: 1
  })
}

export const Gallery: Story = {
  render: () => ({
    components: { QueueNotificationBanner },
    setup() {
      const queueing = args({
        type: 'queuedPending',
        count: 1
      })
      const queued = args({
        type: 'queued',
        count: 2
      })
      const completed = args({
        type: 'completed',
        count: 1,
        thumbnailUrls: [thumbnail('ff6b6b')]
      })
      const completedMultiple = args({
        type: 'completed',
        count: 4
      })
      const completedMultipleWithThumbnail = args({
        type: 'completed',
        count: 4,
        thumbnailUrls: [
          thumbnail('51cf66'),
          thumbnail('ffd43b'),
          thumbnail('ff922b')
        ]
      })
      const failed = args({
        type: 'failed',
        count: 2
      })

      return {
        queueing,
        queued,
        completed,
        completedMultiple,
        completedMultipleWithThumbnail,
        failed
      }
    },
    template: `
      <div class="flex flex-col gap-2">
        <QueueNotificationBanner v-bind="queueing" />
        <QueueNotificationBanner v-bind="queued" />
        <QueueNotificationBanner v-bind="completed" />
        <QueueNotificationBanner v-bind="completedMultiple" />
        <QueueNotificationBanner v-bind="completedMultipleWithThumbnail" />
        <QueueNotificationBanner v-bind="failed" />
      </div>
    `
  })
}
