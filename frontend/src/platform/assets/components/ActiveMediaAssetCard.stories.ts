import type { Meta, StoryObj } from '@storybook/vue3-vite'

import type { JobListItem } from '@/composables/queue/useJobList'

import ActiveMediaAssetCard from './ActiveMediaAssetCard.vue'

const meta: Meta<typeof ActiveMediaAssetCard> = {
  title: 'Platform/Assets/ActiveMediaAssetCard',
  component: ActiveMediaAssetCard
}

export default meta
type Story = StoryObj<typeof meta>

const SAMPLE_PREVIEW =
  'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/images/BigBuckBunny.jpg'

function createJob(overrides: Partial<JobListItem> = {}): JobListItem {
  return {
    id: 'job-1',
    title: 'Running...',
    meta: 'Step 5/10',
    state: 'running',
    progressTotalPercent: 50,
    progressCurrentPercent: 75,
    ...overrides
  }
}

export const Running: Story = {
  decorators: [
    () => ({
      template: '<div style="max-width: 220px;"><story /></div>'
    })
  ],
  args: {
    job: createJob({
      state: 'running',
      progressTotalPercent: 65,
      iconImageUrl: SAMPLE_PREVIEW
    })
  }
}

export const RunningWithoutPreview: Story = {
  decorators: [
    () => ({
      template: '<div style="max-width: 220px;"><story /></div>'
    })
  ],
  args: {
    job: createJob({
      state: 'running',
      progressTotalPercent: 30
    })
  }
}

export const Pending: Story = {
  decorators: [
    () => ({
      template: '<div style="max-width: 220px;"><story /></div>'
    })
  ],
  args: {
    job: createJob({
      state: 'pending',
      title: 'In queue...',
      progressTotalPercent: undefined
    })
  }
}

export const Initialization: Story = {
  decorators: [
    () => ({
      template: '<div style="max-width: 220px;"><story /></div>'
    })
  ],
  args: {
    job: createJob({
      state: 'initialization',
      title: 'Initializing...',
      progressTotalPercent: undefined
    })
  }
}

export const Failed: Story = {
  decorators: [
    () => ({
      template: '<div style="max-width: 220px;"><story /></div>'
    })
  ],
  args: {
    job: createJob({
      state: 'failed',
      title: 'Failed'
    })
  }
}

export const GridLayout: Story = {
  render: () => ({
    components: { ActiveMediaAssetCard },
    setup() {
      const jobs: JobListItem[] = [
        createJob({
          id: 'job-1',
          state: 'running',
          progressTotalPercent: 75,
          iconImageUrl: SAMPLE_PREVIEW
        }),
        createJob({
          id: 'job-2',
          state: 'running',
          progressTotalPercent: 45
        }),
        createJob({
          id: 'job-3',
          state: 'pending',
          title: 'In queue...',
          progressTotalPercent: undefined
        }),
        createJob({
          id: 'job-4',
          state: 'failed',
          title: 'Failed'
        })
      ]
      return { jobs }
    },
    template: `
      <div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 8px; padding: 8px;">
        <ActiveMediaAssetCard
          v-for="job in jobs"
          :key="job.id"
          :job="job"
        />
      </div>
    `
  })
}
