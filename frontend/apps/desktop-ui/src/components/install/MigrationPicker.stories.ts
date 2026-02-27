// eslint-disable-next-line storybook/no-renderer-packages
import type { Meta, StoryObj } from '@storybook/vue3'
import { ref } from 'vue'

import MigrationPicker from './MigrationPicker.vue'

const meta: Meta<typeof MigrationPicker> = {
  title: 'Desktop/Components/MigrationPicker',
  component: MigrationPicker,
  parameters: {
    backgrounds: {
      default: 'dark',
      values: [
        { name: 'dark', value: '#0a0a0a' },
        { name: 'neutral-900', value: '#171717' }
      ]
    }
  },
  decorators: [
    () => {
      ;(window as any).electronAPI = {
        validateComfyUISource: () => Promise.resolve({ isValid: true }),
        showDirectoryPicker: () => Promise.resolve('/Users/username/ComfyUI')
      }

      return { template: '<story />' }
    }
  ]
}

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: () => ({
    components: { MigrationPicker },
    setup() {
      const sourcePath = ref('')
      const migrationItemIds = ref<string[]>([])
      return { sourcePath, migrationItemIds }
    },
    template:
      '<MigrationPicker v-model:sourcePath="sourcePath" v-model:migrationItemIds="migrationItemIds" />'
  })
}
