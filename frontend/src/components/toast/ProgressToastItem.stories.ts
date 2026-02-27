import type { Meta, StoryObj } from '@storybook/vue3-vite'

import type { AssetDownload } from '@/stores/assetDownloadStore'

import ProgressToastItem from './ProgressToastItem.vue'

const meta: Meta<typeof ProgressToastItem> = {
  title: 'Toast/ProgressToastItem',
  component: ProgressToastItem,
  parameters: {
    layout: 'padded'
  },
  decorators: [
    () => ({
      template: '<div class="w-[400px] bg-base-background p-4"><story /></div>'
    })
  ]
}

export default meta
type Story = StoryObj<typeof meta>

function createMockJob(overrides: Partial<AssetDownload> = {}): AssetDownload {
  return {
    taskId: 'task-1',
    assetId: 'asset-1',
    assetName: 'model-v1.safetensors',
    bytesTotal: 1000000,
    bytesDownloaded: 0,
    progress: 0,
    status: 'created',
    lastUpdate: Date.now(),
    ...overrides
  }
}

export const Pending: Story = {
  args: {
    job: createMockJob({
      status: 'created',
      assetName: 'sd-xl-base-1.0.safetensors'
    })
  }
}

export const Running: Story = {
  args: {
    job: createMockJob({
      status: 'running',
      progress: 0.45,
      assetName: 'lora-detail-enhancer.safetensors'
    })
  }
}

export const RunningAlmostComplete: Story = {
  args: {
    job: createMockJob({
      status: 'running',
      progress: 0.92,
      assetName: 'vae-ft-mse-840000.safetensors'
    })
  }
}

export const Completed: Story = {
  args: {
    job: createMockJob({
      status: 'completed',
      progress: 1,
      assetName: 'controlnet-canny.safetensors'
    })
  }
}

export const Failed: Story = {
  args: {
    job: createMockJob({
      status: 'failed',
      progress: 0.23,
      assetName: 'unreachable-model.safetensors'
    })
  }
}

export const LongFileName: Story = {
  args: {
    job: createMockJob({
      status: 'running',
      progress: 0.67,
      assetName:
        'very-long-model-name-with-lots-of-descriptive-text-v2.1-final-release.safetensors'
    })
  }
}
