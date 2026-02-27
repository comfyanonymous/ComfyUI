// eslint-disable-next-line storybook/no-renderer-packages
import type { Meta, StoryObj } from '@storybook/vue3'
import { ref } from 'vue'

import InstallLocationPicker from './InstallLocationPicker.vue'

const meta: Meta<typeof InstallLocationPicker> = {
  title: 'Desktop/Components/InstallLocationPicker',
  component: InstallLocationPicker,
  parameters: {
    layout: 'padded',
    backgrounds: {
      default: 'dark',
      values: [
        { name: 'dark', value: '#0a0a0a' },
        { name: 'neutral-900', value: '#171717' },
        { name: 'neutral-950', value: '#0a0a0a' }
      ]
    }
  },
  decorators: [
    () => {
      // Mock electron API
      ;(window as any).electronAPI = {
        getSystemPaths: () =>
          Promise.resolve({
            defaultInstallPath: '/Users/username/ComfyUI'
          }),
        validateInstallPath: () =>
          Promise.resolve({
            isValid: true,
            exists: false,
            canWrite: true,
            freeSpace: 100000000000,
            requiredSpace: 10000000000,
            isNonDefaultDrive: false
          }),
        validateComfyUISource: () =>
          Promise.resolve({
            isValid: true
          }),
        showDirectoryPicker: () => Promise.resolve('/Users/username/ComfyUI')
      }
      return { template: '<story />' }
    }
  ]
}

export default meta
type Story = StoryObj<typeof meta>

// Default story with accordion expanded
export const Default: Story = {
  render: (args) => ({
    components: { InstallLocationPicker },
    setup() {
      const installPath = ref('/Users/username/ComfyUI')
      const pathError = ref('')
      const migrationSourcePath = ref('/Users/username/ComfyUI-old')
      const migrationItemIds = ref<string[]>(['models', 'custom_nodes'])

      return {
        args,
        installPath,
        pathError,
        migrationSourcePath,
        migrationItemIds
      }
    },
    template: `
      <div class="min-h-screen bg-neutral-950 p-8">
        <InstallLocationPicker
          v-model:installPath="installPath"
          v-model:pathError="pathError"
          v-model:migrationSourcePath="migrationSourcePath"
          v-model:migrationItemIds="migrationItemIds"
        />
      </div>
    `
  })
}

// Story with different background to test transparency
export const OnNeutral900: Story = {
  render: (args) => ({
    components: { InstallLocationPicker },
    setup() {
      const installPath = ref('/Users/username/ComfyUI')
      const pathError = ref('')
      const migrationSourcePath = ref('/Users/username/ComfyUI-old')
      const migrationItemIds = ref<string[]>(['models', 'custom_nodes'])

      return {
        args,
        installPath,
        pathError,
        migrationSourcePath,
        migrationItemIds
      }
    },
    template: `
      <div class="min-h-screen bg-neutral-900 p-8">
        <InstallLocationPicker
          v-model:installPath="installPath"
          v-model:pathError="pathError"
          v-model:migrationSourcePath="migrationSourcePath"
          v-model:migrationItemIds="migrationItemIds"
        />
      </div>
    `
  })
}

// Story with debug overlay showing background colors
export const DebugBackgrounds: Story = {
  render: (args) => ({
    components: { InstallLocationPicker },
    setup() {
      const installPath = ref('/Users/username/ComfyUI')
      const pathError = ref('')
      const migrationSourcePath = ref('/Users/username/ComfyUI-old')
      const migrationItemIds = ref<string[]>(['models', 'custom_nodes'])

      return {
        args,
        installPath,
        pathError,
        migrationSourcePath,
        migrationItemIds
      }
    },
    template: `
      <div class="min-h-screen bg-neutral-950 p-8 relative">
        <div class="absolute top-4 right-4 text-white text-xs space-y-2 z-50">
          <div>Parent bg: neutral-950 (#0a0a0a)</div>
          <div>Accordion content: bg-transparent</div>
          <div>Migration options: bg-transparent + p-4 rounded-lg</div>
        </div>
        <InstallLocationPicker
          v-model:installPath="installPath"
          v-model:pathError="pathError"
          v-model:migrationSourcePath="migrationSourcePath"
          v-model:migrationItemIds="migrationItemIds"
        />
      </div>
    `
  })
}
