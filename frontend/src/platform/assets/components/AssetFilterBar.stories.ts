import type { Meta, StoryObj } from '@storybook/vue3-vite'
import { ref } from 'vue'

import {
  createAssetWithSpecificBaseModel,
  createAssetWithSpecificExtension,
  createAssetWithoutBaseModel,
  createAssetWithoutExtension
} from '@/platform/assets/fixtures/ui-mock-assets'
import type { AssetItem } from '@/platform/assets/schemas/assetSchema'

import AssetFilterBar from './AssetFilterBar.vue'

const meta: Meta<typeof AssetFilterBar> = {
  title: 'Platform/Assets/AssetFilterBar',
  component: AssetFilterBar,
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component:
          'Filter bar for asset browser that dynamically shows/hides filters based on available options.'
      }
    }
  },
  decorators: [
    () => ({
      template: `
        <div class="min-h-screen bg-base-background">
          <div class="bg-base-background border-b border-border-default">
            <story />
          </div>
          <div class="p-6 text-sm text-muted-foreground">
            <p>Filter bar with proper chrome styling showing contextual background and borders.</p>
          </div>
        </div>
      `
    })
  ],
  argTypes: {
    assets: {
      description: 'Array of assets to generate filter options from'
    }
  }
}

export default meta
type Story = StoryObj<typeof meta>

export const BothFiltersVisible: Story = {
  render: () => ({
    components: { AssetFilterBar },
    setup() {
      const assets = ref<AssetItem[]>([
        createAssetWithSpecificExtension('safetensors'),
        createAssetWithSpecificExtension('ckpt'),
        createAssetWithSpecificBaseModel('sd15'),
        createAssetWithSpecificBaseModel('sdxl')
      ])
      return { assets }
    },
    template: '<AssetFilterBar :assets="assets" />'
  }),
  parameters: {
    docs: {
      description: {
        story:
          'Shows both file format and base model filters when assets contain both types of options.'
      }
    }
  }
}

export const OnlyFileFormatFilter: Story = {
  render: () => ({
    components: { AssetFilterBar },
    setup() {
      const assets = ref<AssetItem[]>([
        {
          ...createAssetWithSpecificExtension('safetensors'),
          user_metadata: undefined
        },
        {
          ...createAssetWithSpecificExtension('ckpt'),
          user_metadata: undefined
        }
      ])
      return { assets }
    },
    template: '<AssetFilterBar :assets="assets" />'
  }),
  parameters: {
    docs: {
      description: {
        story:
          'Shows only file format filter when assets have file extensions but no base model metadata.'
      }
    }
  }
}

export const OnlyBaseModelFilter: Story = {
  render: () => ({
    components: { AssetFilterBar },
    setup() {
      const assets = ref<AssetItem[]>([
        {
          ...createAssetWithSpecificBaseModel('sd15'),
          name: 'model_without_extension'
        },
        { ...createAssetWithSpecificBaseModel('sdxl'), name: 'another_model' }
      ])
      return { assets }
    },
    template: '<AssetFilterBar :assets="assets" />'
  }),
  parameters: {
    docs: {
      description: {
        story:
          'Shows only base model filter when assets have base model metadata but no recognizable file extensions.'
      }
    }
  }
}

export const NoFiltersVisible: Story = {
  render: () => ({
    components: { AssetFilterBar },
    setup() {
      const assets = ref<AssetItem[]>([])
      return { assets }
    },
    template: '<AssetFilterBar :assets="assets" />'
  }),
  parameters: {
    docs: {
      description: {
        story:
          'Shows no filters when no assets are provided or assets contain no filterable options.'
      }
    }
  }
}

export const NoFiltersFromAssetsWithoutOptions: Story = {
  render: () => ({
    components: { AssetFilterBar },
    setup() {
      const assets = ref<AssetItem[]>([
        createAssetWithoutExtension(),
        createAssetWithoutBaseModel()
      ])
      return { assets }
    },
    template: '<AssetFilterBar :assets="assets" />'
  }),
  parameters: {
    docs: {
      description: {
        story:
          'Shows no filters when assets are provided but contain no filterable options (no extensions or base models).'
      }
    }
  }
}

export const CategorySwitchingReactivity: Story = {
  render: () => ({
    components: { AssetFilterBar },
    setup() {
      const selectedCategory = ref('all')

      const checkpointAssets: AssetItem[] = [
        {
          ...createAssetWithSpecificExtension('safetensors'),
          tags: ['models', 'checkpoints'],
          user_metadata: { base_model: 'sd15' }
        },
        {
          ...createAssetWithSpecificExtension('safetensors'),
          tags: ['models', 'checkpoints'],
          user_metadata: { base_model: 'sdxl' }
        }
      ]

      const loraAssets: AssetItem[] = [
        {
          ...createAssetWithSpecificExtension('pt'),
          tags: ['models', 'loras'],
          user_metadata: { base_model: 'sd15' }
        },
        {
          ...createAssetWithSpecificExtension('pt'),
          tags: ['models', 'loras'],
          user_metadata: undefined
        }
      ]

      const allAssets = [...checkpointAssets, ...loraAssets]

      const categoryFilteredAssets = ref<AssetItem[]>(allAssets)

      const switchCategory = (category: string) => {
        selectedCategory.value = category
        categoryFilteredAssets.value =
          category === 'all'
            ? allAssets
            : category === 'checkpoints'
              ? checkpointAssets
              : loraAssets
      }

      return { categoryFilteredAssets, selectedCategory, switchCategory }
    },
    template: `
      <div class="space-y-4 p-4">
        <div class="flex gap-2">
          <button
            @click="switchCategory('all')"
            :class="[
              'px-4 py-2 rounded border',
              selectedCategory === 'all'
                ? 'bg-blue-500 text-white border-blue-600'
                : 'bg-secondary-background border-border-default'
            ]"
          >
            All (.safetensors + .pt, sd15 + sdxl)
          </button>
          <button
            @click="switchCategory('checkpoints')"
            :class="[
              'px-4 py-2 rounded border',
              selectedCategory === 'checkpoints'
                ? 'bg-blue-500 text-white border-blue-600'
                : 'bg-secondary-background border-border-default'
            ]"
          >
            Checkpoints (.safetensors, sd15 + sdxl)
          </button>
          <button
            @click="switchCategory('loras')"
            :class="[
              'px-4 py-2 rounded border',
              selectedCategory === 'loras'
                ? 'bg-blue-500 text-white border-blue-600'
                : 'bg-secondary-background border-border-default'
            ]"
          >
            LoRAs (.pt, sd15 only)
          </button>
        </div>
        <AssetFilterBar :assets="categoryFilteredAssets" />
      </div>
    `
  }),
  parameters: {
    docs: {
      description: {
        story:
          'Interactive demo showing filter options updating reactively when category changes. Click buttons to see filters adapt to the selected category.'
      }
    }
  }
}
