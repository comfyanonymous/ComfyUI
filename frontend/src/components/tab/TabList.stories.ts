import type { Meta, StoryObj } from '@storybook/vue3-vite'
import { ref } from 'vue'

import Tab from './Tab.vue'
import TabList from './TabList.vue'
import type { ComponentExposed } from 'vue-component-type-helpers'

interface GenericMeta<C> extends Omit<Meta<C>, 'component'> {
  component: ComponentExposed<C>
}

const meta: GenericMeta<typeof TabList> = {
  title: 'Components/Tab/TabList',
  component: TabList,
  tags: ['autodocs'],
  argTypes: {
    modelValue: {
      control: 'text',
      description: 'The currently selected tab value'
    },
    'onUpdate:modelValue': { action: 'update:modelValue' }
  }
}

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: (args) => ({
    components: { TabList, Tab },
    setup() {
      const activeTab = ref(args.modelValue || 'tab1')
      return { activeTab }
    },
    template: `
      <TabList v-model="activeTab">
        <Tab value="tab1">Tab 1</Tab>
        <Tab value="tab2">Tab 2</Tab>
        <Tab value="tab3">Tab 3</Tab>
      </TabList>
      <div class="mt-4 p-4 border rounded">
        Selected tab: {{ activeTab }}
      </div>
    `
  }),
  args: {
    modelValue: 'tab1'
  }
}

export const ManyTabs: Story = {
  render: () => ({
    components: { TabList, Tab },
    setup() {
      const activeTab = ref('tab1')
      return { activeTab }
    },
    template: `
      <TabList v-model="activeTab">
        <Tab value="tab1">Dashboard</Tab>
        <Tab value="tab2">Analytics</Tab>
        <Tab value="tab3">Reports</Tab>
        <Tab value="tab4">Settings</Tab>
        <Tab value="tab5">Profile</Tab>
      </TabList>
      <div class="mt-4 p-4 border rounded">
        Selected tab: {{ activeTab }}
      </div>
    `
  })
}

export const WithIcons: Story = {
  render: () => ({
    components: { TabList, Tab },
    setup() {
      const activeTab = ref('home')
      return { activeTab }
    },
    template: `
      <TabList v-model="activeTab">
        <Tab value="home">
          <i class="pi pi-home mr-2"></i>
          Home
        </Tab>
        <Tab value="users">
          <i class="pi pi-users mr-2"></i>
          Users
        </Tab>
        <Tab value="settings">
          <i class="pi pi-cog mr-2"></i>
          Settings
        </Tab>
      </TabList>
      <div class="mt-4 p-4 border rounded">
        Selected tab: {{ activeTab }}
      </div>
    `
  })
}

export const LongLabels: Story = {
  render: () => ({
    components: { TabList, Tab },
    setup() {
      const activeTab = ref('overview')
      return { activeTab }
    },
    template: `
      <TabList v-model="activeTab">
        <Tab value="overview">Project Overview</Tab>
        <Tab value="documentation">Documentation & Guides</Tab>
        <Tab value="deployment">Deployment Settings</Tab>
        <Tab value="monitoring">Monitoring & Analytics</Tab>
      </TabList>
      <div class="mt-4 p-4 border rounded">
        Selected tab: {{ activeTab }}
      </div>
    `
  })
}

export const Interactive: Story = {
  render: () => ({
    components: { TabList, Tab },
    setup() {
      const activeTab = ref('input')
      const handleTabChange = (value: string) => {
        console.log('Tab changed to:', value)
      }
      return { activeTab, handleTabChange }
    },
    template: `
      <div class="space-y-4">
        <div>
          <h3 class="text-sm font-semibold mb-2">Example: Media Assets</h3>
          <TabList v-model="activeTab" @update:model-value="handleTabChange">
            <Tab value="input">Imported</Tab>
            <Tab value="output">Generated</Tab>
          </TabList>
        </div>
        
        <div class="p-4 bg-gray-50 dark:bg-gray-800 rounded">
          <div v-if="activeTab === 'input'">
            <p>Showing imported assets...</p>
          </div>
          <div v-else-if="activeTab === 'output'">
            <p>Showing generated assets...</p>
          </div>
        </div>
        
        <div class="text-sm text-gray-600">
          Current tab value: <code>{{ activeTab }}</code>
        </div>
      </div>
    `
  })
}
