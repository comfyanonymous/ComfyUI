import type { Meta, StoryObj } from '@storybook/vue3-vite'

import AssetBrowserModal from '@/platform/assets/components/AssetBrowserModal.vue'
import type { AssetDisplayItem } from '@/platform/assets/composables/useAssetBrowser'
import {
  createMockAssets,
  mockAssets
} from '@/platform/assets/fixtures/ui-mock-assets'

// Story arguments interface
interface StoryArgs {
  nodeType: string
  inputName: string
  currentValue: string
  showLeftPanel?: boolean
}

const meta: Meta<StoryArgs> = {
  title: 'Platform/Assets/AssetBrowserModal',
  component: AssetBrowserModal,
  parameters: {
    layout: 'fullscreen'
  },
  argTypes: {
    nodeType: {
      control: 'select',
      options: ['CheckpointLoaderSimple', 'VAELoader', 'ControlNetLoader'],
      description: 'ComfyUI node type for context'
    },
    inputName: {
      control: 'select',
      options: ['ckpt_name', 'vae_name', 'control_net_name'],
      description: 'Widget input name'
    },
    currentValue: {
      control: 'text',
      description: 'Current selected asset value'
    },
    showLeftPanel: {
      control: 'boolean',
      description: 'Whether to show the left panel with categories'
    }
  }
}

export default meta
type Story = StoryObj<typeof meta>

// Modal Layout Stories
export const Default: Story = {
  args: {
    nodeType: 'CheckpointLoaderSimple',
    inputName: 'ckpt_name',
    currentValue: '',
    showLeftPanel: true
  },
  render: (args) => ({
    components: { AssetBrowserModal },
    setup() {
      const onAssetSelect = (_asset: AssetDisplayItem) => {
        // Asset selection handler for story
      }
      const onClose = () => {}

      return {
        ...args,
        onAssetSelect,
        onClose,
        assets: mockAssets
      }
    },
    template: `
      <div class="flex items-center justify-center min-h-screen bg-ash-800 p-4">
        <AssetBrowserModal
          :node-type="nodeType"
          :input-name="inputName"
          :show-left-panel="showLeftPanel"
          :assets="assets"
          @asset-select="onAssetSelect"
          @close="onClose"
        />
      </div>
    `
  })
}

// Story demonstrating single asset type (auto-hides left panel)
export const SingleAssetType: Story = {
  args: {
    nodeType: 'CheckpointLoaderSimple',
    inputName: 'ckpt_name',
    currentValue: '',
    showLeftPanel: false
  },
  render: (args) => ({
    components: { AssetBrowserModal },
    setup() {
      const onAssetSelect = (_asset: AssetDisplayItem) => {
        // Asset selection handler for story
      }
      const onClose = () => {
        // Modal close handler for story
      }

      // Create assets with only one type (checkpoints)
      const singleTypeAssets = createMockAssets(15).map((asset) => ({
        ...asset,
        type: 'checkpoint'
      }))

      return { ...args, onAssetSelect, onClose, assets: singleTypeAssets }
    },
    template: `
      <div class="flex items-center justify-center min-h-screen bg-ash-800 p-4">
        <AssetBrowserModal
          :node-type="nodeType"
          :input-name="inputName"
          :show-left-panel="showLeftPanel"
          :assets="assets"
          @asset-select="onAssetSelect"
          @close="onClose"
        />
      </div>
    `
  }),
  parameters: {
    docs: {
      description: {
        story:
          'Modal with assets of only one type (checkpoint) - left panel auto-hidden.'
      }
    }
  }
}

// Story with left panel explicitly hidden
export const NoLeftPanel: Story = {
  args: {
    nodeType: 'CheckpointLoaderSimple',
    inputName: 'ckpt_name',
    currentValue: '',
    showLeftPanel: false
  },
  render: (args) => ({
    components: { AssetBrowserModal },
    setup() {
      const onAssetSelect = (_asset: AssetDisplayItem) => {
        // Asset selection handler for story
      }
      const onClose = () => {
        // Modal close handler for story
      }

      return { ...args, onAssetSelect, onClose, assets: mockAssets }
    },
    template: `
      <div class="flex items-center justify-center min-h-screen bg-ash-800 p-4">
        <AssetBrowserModal
          :node-type="nodeType"
          :input-name="inputName"
          :show-left-panel="showLeftPanel"
          :assets="assets"
          @asset-select="onAssetSelect"
          @close="onClose"
        />
      </div>
    `
  }),
  parameters: {
    docs: {
      description: {
        story:
          'Modal with left panel explicitly disabled via showLeftPanel=false.'
      }
    }
  }
}
