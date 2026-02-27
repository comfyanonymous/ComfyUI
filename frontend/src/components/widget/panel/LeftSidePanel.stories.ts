import type { Meta, StoryObj } from '@storybook/vue3-vite'
import { ref } from 'vue'

import LeftSidePanel from './LeftSidePanel.vue'

const meta: Meta<typeof LeftSidePanel> = {
  title: 'Components/Widget/Panel/LeftSidePanel',
  component: LeftSidePanel,
  argTypes: {
    'onUpdate:modelValue': {
      table: { disable: true }
    }
  }
}

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    modelValue: 'installed',
    navItems: [
      {
        id: 'installed',
        label: 'Installed',
        icon: 'icon-[lucide--download]'
      },
      {
        id: 'models',
        label: 'Models',
        icon: 'icon-[lucide--layers]'
      },
      {
        id: 'nodes',
        label: 'Nodes',
        icon: 'icon-[lucide--grid-3x3]'
      }
    ]
  },
  render: (args) => ({
    components: { LeftSidePanel },
    setup() {
      const selectedItem = ref(args.modelValue)
      return { args, selectedItem }
    },
    template: `
      <div style="height: 500px; width: 256px;">
        <LeftSidePanel v-model="selectedItem" :nav-items="args.navItems" />
      </div>
    `
  })
}

export const WithGroups: Story = {
  args: {
    modelValue: 'tag-sd15',
    navItems: [
      {
        id: 'installed',
        label: 'Installed',
        icon: 'icon-[lucide--download]'
      },
      {
        title: 'TAGS',
        items: [
          {
            id: 'tag-sd15',
            label: 'SD 1.5',
            icon: 'icon-[lucide--tag]'
          },
          {
            id: 'tag-sdxl',
            label: 'SDXL',
            icon: 'icon-[lucide--tag]'
          },
          {
            id: 'tag-utility',
            label: 'Utility',
            icon: 'icon-[lucide--tag]'
          }
        ]
      },
      {
        title: 'CATEGORIES',
        items: [
          {
            id: 'cat-models',
            label: 'Models',
            icon: 'icon-[lucide--layers]'
          },
          {
            id: 'cat-nodes',
            label: 'Nodes',
            icon: 'icon-[lucide--grid-3x3]'
          }
        ]
      }
    ]
  },
  render: (args) => ({
    components: { LeftSidePanel },
    setup() {
      const selectedItem = ref(args.modelValue)
      return { args, selectedItem }
    },
    template: `
      <div style="height: 500px; width: 256px;">
        <LeftSidePanel v-model="selectedItem" :nav-items="args.navItems" />
        <div class="mt-4 p-2 text-sm">
          Selected: {{ selectedItem }}
        </div>
      </div>
    `
  })
}

export const DefaultIcons: Story = {
  args: {
    modelValue: 'home',
    navItems: [
      {
        id: 'home',
        label: 'Home',
        icon: 'icon-[lucide--folder]'
      },
      {
        id: 'documents',
        label: 'Documents',
        icon: 'icon-[lucide--folder]'
      },
      {
        id: 'downloads',
        label: 'Downloads',
        icon: 'icon-[lucide--folder]'
      },
      {
        id: 'desktop',
        label: 'Desktop',
        icon: 'icon-[lucide--folder]'
      }
    ]
  },
  render: (args) => ({
    components: { LeftSidePanel },
    setup() {
      const selectedItem = ref(args.modelValue)
      return { args, selectedItem }
    },
    template: `
      <div style="height: 400px; width: 256px;">
        <LeftSidePanel v-model="selectedItem" :nav-items="args.navItems" />
      </div>
    `
  })
}

export const LongLabels: Story = {
  args: {
    modelValue: 'general',
    navItems: [
      {
        id: 'general',
        label: 'General Settings',
        icon: 'icon-[lucide--wrench]'
      },
      {
        id: 'appearance',
        label: 'Appearance & Themes Configuration',
        icon: 'icon-[lucide--wrench]'
      },
      {
        title: 'ADVANCED OPTIONS',
        items: [
          {
            id: 'performance',
            label: 'Performance & Optimization Settings',
            icon: 'icon-[lucide--zap]'
          },
          {
            id: 'experimental',
            label: 'Experimental Features (Beta)',
            icon: 'icon-[lucide--puzzle]'
          }
        ]
      }
    ]
  },
  render: (args) => ({
    components: { LeftSidePanel },
    setup() {
      const selectedItem = ref(args.modelValue)
      return { args, selectedItem }
    },
    template: `
      <div style="height: 500px; width: 256px;">
        <LeftSidePanel v-model="selectedItem" :nav-items="args.navItems" />
      </div>
    `
  })
}
