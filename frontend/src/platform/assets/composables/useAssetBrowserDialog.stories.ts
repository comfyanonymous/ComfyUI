import type { Meta, StoryObj } from '@storybook/vue3-vite'
import { ref } from 'vue'

import AssetBrowserModal from '@/platform/assets/components/AssetBrowserModal.vue'
import { mockAssets } from '@/platform/assets/fixtures/ui-mock-assets'

// Component that simulates the useAssetBrowserDialog functionality with working close
const DialogDemoComponent = {
  components: { AssetBrowserModal },
  setup() {
    const isDialogOpen = ref(false)
    const currentNodeType = ref('CheckpointLoaderSimple')
    const currentInputName = ref('ckpt_name')
    const currentValue = ref('')

    const handleOpenDialog = (
      nodeType: string,
      inputName: string,
      value = ''
    ) => {
      currentNodeType.value = nodeType
      currentInputName.value = inputName
      currentValue.value = value
      isDialogOpen.value = true
    }

    const handleCloseDialog = () => {
      isDialogOpen.value = false
    }

    const handleAssetSelected = (assetPath: string) => {
      alert(`Selected asset: ${assetPath}`)
      isDialogOpen.value = false // Auto-close like the real composable
    }

    const handleOpenWithCurrentValue = () => {
      handleOpenDialog(
        'CheckpointLoaderSimple',
        'ckpt_name',
        'realistic_vision_v5.safetensors'
      )
    }

    return {
      isDialogOpen,
      currentNodeType,
      currentInputName,
      currentValue,
      handleOpenDialog,
      handleOpenWithCurrentValue,
      handleCloseDialog,
      handleAssetSelected,
      mockAssets
    }
  },
  template: `
    <div class="relative">
      <div class="p-8 space-y-4">
        <h2 class="text-2xl font-bold mb-6">Asset Browser Dialog Demo</h2>

        <div class="space-y-4">
          <div>
            <h3 class="text-lg font-semibold mb-2">Different Node Types</h3>
            <div class="flex gap-3 flex-wrap">
              <button
                @click="handleOpenDialog('CheckpointLoaderSimple', 'ckpt_name')"
                class="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
              >
                Browse Checkpoints
              </button>
              <button
                @click="handleOpenDialog('VAELoader', 'vae_name')"
                class="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700"
              >
                Browse VAE
              </button>
              <button
                @click="handleOpenDialog('ControlNetLoader', 'control_net_name')"
                class="px-4 py-2 bg-purple-600 text-white rounded hover:bg-purple-700"
              >
                Browse ControlNet
              </button>
            </div>
          </div>

          <div>
            <h3 class="text-lg font-semibold mb-2">With Current Value</h3>
            <button
              @click="handleOpenWithCurrentValue"
              class="px-4 py-2 bg-orange-600 text-white rounded hover:bg-orange-700"
            >
              Change Current Model
            </button>
            <p class="text-sm text-smoke-600 mt-1">
              Opens with "realistic_vision_v5.safetensors" as current value
            </p>
          </div>

          <div class="mt-8 p-4 bg-smoke-100 rounded">
            <h4 class="font-semibold mb-2">Instructions:</h4>
            <ul class="text-sm space-y-1">
              <li>• Click any button to open the Asset Browser dialog</li>
              <li>• Select an asset to see the callback in action</li>
              <li>• Check the browser console for logged events</li>
              <li>• Try toggling the left panel with different asset types</li>
              <li>• Close button will work properly in this demo</li>
            </ul>
          </div>
        </div>
      </div>

      <!-- Dialog Modal Overlay -->
      <div
        v-if="isDialogOpen"
        class="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50"
        @click.self="handleCloseDialog"
      >
        <div class="w-[80vw] h-[80vh] max-w-[80vw] max-h-[80vh] rounded-2xl overflow-hidden">
          <AssetBrowserModal
            :assets="mockAssets"
            :node-type="currentNodeType"
            :input-name="currentInputName"
            :current-value="currentValue"
            @asset-select="handleAssetSelected"
            @close="handleCloseDialog"
          />
        </div>
      </div>
    </div>
  `
}

const meta: Meta = {
  title: 'Platform/Assets/useAssetBrowserDialog',
  parameters: {
    layout: 'fullscreen',
    docs: {
      description: {
        component:
          'Demonstrates the AssetBrowserModal functionality as used by the useAssetBrowserDialog composable.'
      }
    }
  }
}

export default meta
type Story = StoryObj<typeof meta>

export const Demo: Story = {
  render: () => ({
    components: { DialogDemoComponent },
    template: `
      <div>
        <DialogDemoComponent />

        <!-- Code Example Section -->
        <div class="p-8 border-t border-smoke-200 bg-gray-50">
          <h2 class="text-2xl font-bold mb-4">Code Example</h2>
          <p class="text-smoke-600 mb-4">
            This is how you would use the composable in your component:
          </p>
          <div class="bg-white p-4 rounded-lg border shadow-sm">
            <pre><code class="text-sm">import { useAssetBrowserDialog } from '@/platform/assets/composables/useAssetBrowserDialog'

export default {
  setup() {
    const assetBrowserDialog = useAssetBrowserDialog()

    const openBrowser = () => {
      assetBrowserDialog.show({
        nodeType: 'CheckpointLoaderSimple',
        inputName: 'ckpt_name',
        currentValue: '',
        onAssetSelected: (assetPath) => {
          console.log('Selected:', assetPath)
          // Update your component state
        }
      })
    }

    return { openBrowser }
  }
}</code></pre>
          </div>
          <div class="mt-4 p-3 bg-blue-50 border border-azure-400 rounded">
            <p class="text-sm text-blue-800">
              <strong>💡 Try it:</strong> Use the interactive buttons above to see this code in action!
            </p>
          </div>
        </div>
      </div>
    `
  }),
  parameters: {
    docs: {
      description: {
        story:
          'Complete demo showing both interactive functionality and code examples for using useAssetBrowserDialog to open the Asset Browser modal programmatically.'
      }
    }
  }
}
