import type { Meta, StoryObj } from '@storybook/vue3-vite'
import { toRef } from 'vue'

import type { JobAction } from '@/composables/queue/useJobActions'
import type { JobListItem } from '@/composables/queue/useJobList'
import { useOutputStacks } from '@/platform/assets/composables/useOutputStacks'
import type { AssetItem } from '@/platform/assets/schemas/assetSchema'
import { setMockJobActions } from '@/storybook/mocks/useJobActions'
import { setMockJobItems } from '@/storybook/mocks/useJobList'
import { iconForJobState } from '@/utils/queueDisplay'

import AssetsSidebarListView from './AssetsSidebarListView.vue'

type StoryArgs = {
  assets: AssetItem[]
  jobs: JobListItem[]
  selectedAssetIds?: string[]
  actionsByJobId?: Record<string, JobAction[]>
}

function baseDecorator() {
  return {
    template: `
        <div class="bg-base-background p-6">
          <story />
        </div>
      `
  }
}

const meta: Meta<StoryArgs> = {
  title: 'Components/Sidebar/AssetsSidebarListView',
  component: AssetsSidebarListView,
  parameters: {
    layout: 'centered'
  },
  decorators: [baseDecorator]
}

export default meta
type Story = StoryObj<typeof meta>

const baseTimestamp = '2024-01-15T10:00:00Z'

const sampleJobs: JobListItem[] = [
  {
    id: 'job-pending-1',
    title: 'In queue',
    meta: '8:59:30pm',
    state: 'pending',
    iconName: iconForJobState('pending'),
    showClear: true
  },
  {
    id: 'job-init-1',
    title: 'Initializing...',
    meta: '8:59:35pm',
    state: 'initialization',
    iconName: iconForJobState('initialization'),
    showClear: true
  },
  {
    id: 'job-running-1',
    title: 'Total: 30%',
    meta: 'KSampler: 70%',
    state: 'running',
    iconName: iconForJobState('running'),
    showClear: true,
    progressTotalPercent: 30,
    progressCurrentPercent: 70
  }
]

const sampleAssets: AssetItem[] = [
  {
    id: 'asset-image-1',
    name: 'image-032.png',
    created_at: baseTimestamp,
    preview_url: '/assets/images/comfy-logo-single.svg',
    size: 1887437,
    tags: [],
    user_metadata: {
      jobId: 'job-running-1',
      nodeId: 12,
      executionTimeInSeconds: 1.84
    }
  },
  {
    id: 'asset-video-1',
    name: 'clip-01.mp4',
    created_at: baseTimestamp,
    preview_url: '/assets/images/default-template.png',
    size: 8394820,
    tags: [],
    user_metadata: {
      duration: 132000
    }
  },
  {
    id: 'asset-audio-1',
    name: 'soundtrack-01.mp3',
    created_at: baseTimestamp,
    size: 5242880,
    tags: [],
    user_metadata: {
      duration: 200000
    }
  },
  {
    id: 'asset-3d-1',
    name: 'scene-01.glb',
    created_at: baseTimestamp,
    size: 134217728,
    tags: []
  },
  {
    id: 'asset-text-1',
    name: 'generation-notes.txt',
    created_at: baseTimestamp,
    preview_url: '/assets/images/default-template.png',
    size: 2048,
    tags: []
  },
  {
    id: 'asset-other-1',
    name: 'workflow-payload.bin',
    created_at: baseTimestamp,
    preview_url: '/assets/images/default-template.png',
    size: 4096,
    tags: []
  }
]

const cancelAction: JobAction = {
  icon: 'icon-[lucide--x]',
  label: 'Cancel',
  variant: 'destructive'
}

export const RunningAndGenerated: Story = {
  args: {
    assets: sampleAssets,
    jobs: sampleJobs,
    actionsByJobId: {
      'job-pending-1': [cancelAction],
      'job-init-1': [cancelAction],
      'job-running-1': [cancelAction]
    }
  },
  render: renderAssetsSidebarListView
}

export const TextAndMiscGeneratedAssets: Story = {
  args: {
    assets: sampleAssets.filter((asset) =>
      ['.txt', '.bin'].some((suffix) => asset.name.endsWith(suffix))
    ),
    jobs: []
  },
  render: renderAssetsSidebarListView
}

function renderAssetsSidebarListView(args: StoryArgs) {
  return {
    components: { AssetsSidebarListView },
    setup() {
      setMockJobItems(args.jobs)
      setMockJobActions(args.actionsByJobId ?? {})
      const { assetItems, selectableAssets, isStackExpanded, toggleStack } =
        useOutputStacks({
          assets: toRef(args, 'assets')
        })
      const selectedIds = new Set(args.selectedAssetIds ?? [])
      function isSelected(assetId: string) {
        return selectedIds.has(assetId)
      }

      return {
        args,
        assetItems,
        selectableAssets,
        isSelected,
        isStackExpanded,
        toggleStack
      }
    },
    template: `
      <div class="h-[520px] w-[320px] overflow-hidden rounded-lg border border-panel-border">
        <AssetsSidebarListView
          :asset-items="assetItems"
          :selectable-assets="selectableAssets"
          :is-selected="isSelected"
          :is-stack-expanded="isStackExpanded"
          :toggle-stack="toggleStack"
        />
      </div>
    `
  }
}
