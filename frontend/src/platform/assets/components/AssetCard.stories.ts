import type { Meta, StoryObj } from '@storybook/vue3-vite'

import AssetCard from '@/platform/assets/components/AssetCard.vue'
import type { AssetDisplayItem } from '@/platform/assets/composables/useAssetBrowser'
import { mockAssets } from '@/platform/assets/fixtures/ui-mock-assets'

// Use the first mock asset as base and transform it to display format
const baseAsset = mockAssets[0]
const createAssetData = (
  overrides: Partial<AssetDisplayItem> = {}
): AssetDisplayItem => ({
  ...baseAsset,
  secondaryText:
    'High-quality realistic images with perfect detail and natural lighting effects for professional photography',
  badges: [
    { label: 'checkpoints', type: 'type' },
    { label: '2.1 GB', type: 'size' }
  ],
  stats: {
    formattedDate: '3/15/25',
    downloadCount: '1.8k',
    stars: '4.2k'
  },
  ...overrides
})

const meta: Meta<typeof AssetCard> = {
  title: 'Platform/Assets/AssetCard',
  component: AssetCard,
  parameters: {
    layout: 'centered'
  },
  decorators: [
    () => ({
      template: '<div class="p-8 bg-base-background"><story /></div>'
    })
  ]
}

export default meta
type Story = StoryObj<typeof meta>

export const Interactive: Story = {
  args: {
    asset: createAssetData(),
    interactive: true
  },
  decorators: [
    () => ({
      template: '<div class="p-8 bg-base-background max-w-96"><story /></div>'
    })
  ],
  parameters: {
    docs: {
      description: {
        story:
          'Default AssetCard with complete data including badges and all stats.'
      }
    }
  }
}

export const NonInteractive: Story = {
  args: {
    asset: createAssetData(),
    interactive: false
  },
  decorators: [
    () => ({
      template: '<div class="p-8 bg-base-background max-w-96"><story /></div>'
    })
  ],
  parameters: {
    docs: {
      description: {
        story:
          'AssetCard in non-interactive mode - renders as div without button semantics.'
      }
    }
  }
}

export const WithPreviewImage: Story = {
  args: {
    asset: createAssetData({
      preview_url: '/assets/images/comfy-logo-single.svg'
    }),
    interactive: true
  },
  decorators: [
    () => ({
      template: '<div class="p-8 bg-base-background max-w-96"><story /></div>'
    })
  ],
  parameters: {
    docs: {
      description: {
        story: 'AssetCard with a preview image displayed.'
      }
    }
  }
}

export const FallbackGradient: Story = {
  args: {
    asset: createAssetData({
      preview_url: undefined
    }),
    interactive: true
  },
  decorators: [
    () => ({
      template: '<div class="p-8 bg-base-background max-w-96"><story /></div>'
    })
  ],
  parameters: {
    docs: {
      description: {
        story:
          'AssetCard showing fallback gradient when no preview image is available.'
      }
    }
  }
}

export const EdgeCases: Story = {
  render: () => ({
    components: { AssetCard },
    setup() {
      const edgeCases = [
        // Default case for comparison
        createAssetData({
          name: 'Complete Data',
          secondaryText: 'Asset with all data present for comparison'
        }),
        // No badges
        createAssetData({
          id: 'no-badges',
          name: 'No Badges',
          secondaryText:
            'Testing graceful handling when badges are not provided',
          badges: []
        }),
        // No stars
        createAssetData({
          id: 'no-stars',
          name: 'No Stars',
          secondaryText: 'Testing missing stars data gracefully',
          stats: {
            downloadCount: '1.8k',
            formattedDate: '3/15/25'
          }
        }),
        // No downloads
        createAssetData({
          id: 'no-downloads',
          name: 'No Downloads',
          secondaryText: 'Testing missing downloads data gracefully',
          stats: {
            stars: '4.2k',
            formattedDate: '3/15/25'
          }
        }),
        // No date
        createAssetData({
          id: 'no-date',
          name: 'No Date',
          secondaryText: 'Testing missing date data gracefully',
          stats: {
            stars: '4.2k',
            downloadCount: '1.8k'
          }
        }),
        // No stats at all
        createAssetData({
          id: 'no-stats',
          name: 'No Stats',
          secondaryText: 'Testing when all stats are missing',
          stats: {}
        }),
        // Long secondaryText
        createAssetData({
          id: 'long-desc',
          name: 'Long Description',
          secondaryText:
            'This is a very long description that should demonstrate how the component handles text overflow and truncation with ellipsis. The description continues with even more content to ensure we test the 2-line clamp behavior properly and see how it renders when there is significantly more text than can fit in the allocated space.'
        }),
        // Minimal data
        createAssetData({
          id: 'minimal',
          name: 'Minimal',
          secondaryText: 'Basic model',
          tags: ['models'],
          badges: [],
          stats: {}
        })
      ]

      return { edgeCases }
    },
    template: `
      <div class="grid grid-cols-4 gap-6 p-8 bg-base-background">
        <AssetCard
          v-for="asset in edgeCases"
          :key="asset.id"
          :asset="asset"
          :interactive="true"
          @select="(asset) => console.log('Selected:', asset)"
        />
      </div>
    `
  }),
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        story:
          'All AssetCard edge cases in a grid layout to test graceful handling of missing data, badges, stats, and long descriptions.'
      }
    }
  }
}
