import type { Meta, StoryObj } from '@storybook/vue3-vite'
import type { MultiSelectProps } from 'primevue/multiselect'
import { ref } from 'vue'

import MultiSelect from './MultiSelect.vue'
import type { SelectOption } from './types'

// Combine our component props with PrimeVue MultiSelect props
// Since we use v-bind="$attrs", all PrimeVue props are available
interface ExtendedProps extends Partial<MultiSelectProps> {
  // Our custom props
  label?: string
  showSearchBox?: boolean
  showSelectedCount?: boolean
  showClearButton?: boolean
  searchPlaceholder?: string
  listMaxHeight?: string
  popoverMinWidth?: string
  popoverMaxWidth?: string
  // Override modelValue type to match our Option type
  modelValue?: SelectOption[]
}

const meta: Meta<ExtendedProps> = {
  title: 'Components/Input/MultiSelect',
  component: MultiSelect,
  tags: ['autodocs'],
  argTypes: {
    label: {
      control: 'text'
    },
    options: {
      control: 'object'
    },
    showSearchBox: {
      control: 'boolean',
      description: 'Toggle searchBar visibility'
    },
    showSelectedCount: {
      control: 'boolean',
      description: 'Toggle selected count visibility'
    },
    showClearButton: {
      control: 'boolean',
      description: 'Toggle clear button visibility'
    },
    searchPlaceholder: {
      control: 'text'
    },
    listMaxHeight: {
      control: 'text',
      description: 'Maximum height of the dropdown list'
    },
    popoverMinWidth: {
      control: 'text',
      description: 'Minimum width of the popover'
    },
    popoverMaxWidth: {
      control: 'text',
      description: 'Maximum width of the popover'
    }
  },
  args: {
    label: 'Select',
    options: [
      { name: 'Vue', value: 'vue' },
      { name: 'React', value: 'react' },
      { name: 'Angular', value: 'angular' },
      { name: 'Svelte', value: 'svelte' }
    ],
    showSearchBox: false,
    showSelectedCount: false,
    showClearButton: false,
    searchPlaceholder: 'Search...'
  }
}

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: (args) => ({
    components: { MultiSelect },
    setup() {
      const selected = ref([])
      const options = args.options || [
        { name: 'Vue', value: 'vue' },
        { name: 'React', value: 'react' },
        { name: 'Angular', value: 'angular' },
        { name: 'Svelte', value: 'svelte' }
      ]
      return { selected, options, args }
    },
    template: `
      <div>
        <MultiSelect 
          v-model="selected" 
          :options="options"
          :label="args.label"
          :showSearchBox="args.showSearchBox"
          :showSelectedCount="args.showSelectedCount"
          :showClearButton="args.showClearButton"
          :searchPlaceholder="args.searchPlaceholder"
        />
        <div class="mt-4 p-3 bg-base-background rounded">
          <p class="text-sm">Selected: {{ selected.length > 0 ? selected.map(s => s.name).join(', ') : 'None' }}</p>
        </div>
      </div>
    `
  })
}

export const WithPreselectedValues: Story = {
  render: (args) => ({
    components: { MultiSelect },
    setup() {
      const options = args.options || [
        { name: 'JavaScript', value: 'js' },
        { name: 'TypeScript', value: 'ts' },
        { name: 'Python', value: 'python' },
        { name: 'Go', value: 'go' },
        { name: 'Rust', value: 'rust' }
      ]
      const selected = ref([options[0], options[1]])
      return { selected, options, args }
    },
    template: `
      <div>
        <MultiSelect 
          v-model="selected" 
          :options="options"
          :label="args.label"
          :showSearchBox="args.showSearchBox"
          :showSelectedCount="args.showSelectedCount"
          :showClearButton="args.showClearButton"
          :searchPlaceholder="args.searchPlaceholder"
        />
        <div class="mt-4 p-3 bg-base-background rounded">
          <p class="text-sm">Selected: {{ selected.map(s => s.name).join(', ') }}</p>
        </div>
      </div>
    `
  }),
  args: {
    label: 'Select Languages',
    options: [
      { name: 'JavaScript', value: 'js' },
      { name: 'TypeScript', value: 'ts' },
      { name: 'Python', value: 'python' },
      { name: 'Go', value: 'go' },
      { name: 'Rust', value: 'rust' }
    ],
    showSearchBox: false,
    showSelectedCount: false,
    showClearButton: false,
    searchPlaceholder: 'Search...'
  }
}

export const MultipleSelectors: Story = {
  render: (args) => ({
    components: { MultiSelect },
    setup() {
      const frameworkOptions = ref([
        { name: 'Vue', value: 'vue' },
        { name: 'React', value: 'react' },
        { name: 'Angular', value: 'angular' },
        { name: 'Svelte', value: 'svelte' }
      ])

      const projectOptions = ref([
        { name: 'Project A', value: 'proj-a' },
        { name: 'Project B', value: 'proj-b' },
        { name: 'Project C', value: 'proj-c' },
        { name: 'Project D', value: 'proj-d' }
      ])

      const tagOptions = ref([
        { name: 'Frontend', value: 'frontend' },
        { name: 'Backend', value: 'backend' },
        { name: 'Database', value: 'database' },
        { name: 'DevOps', value: 'devops' },
        { name: 'Testing', value: 'testing' }
      ])

      const selectedFrameworks = ref([])
      const selectedProjects = ref([])
      const selectedTags = ref([])

      return {
        frameworkOptions,
        projectOptions,
        tagOptions,
        selectedFrameworks,
        selectedProjects,
        selectedTags,
        args
      }
    },
    template: `
      <div class="space-y-4">
        <div class="flex gap-2">
          <MultiSelect 
            v-model="selectedFrameworks" 
            :options="frameworkOptions"
            label="Select Frameworks"
            :showSearchBox="args.showSearchBox"
            :showSelectedCount="args.showSelectedCount"
            :showClearButton="args.showClearButton"
            :searchPlaceholder="args.searchPlaceholder"
          />
          <MultiSelect 
            v-model="selectedProjects" 
            :options="projectOptions"
            label="Select Projects"
            :showSearchBox="args.showSearchBox"
            :showSelectedCount="args.showSelectedCount"
            :showClearButton="args.showClearButton"
            :searchPlaceholder="args.searchPlaceholder"
          />
          <MultiSelect 
            v-model="selectedTags" 
            :options="tagOptions"
            label="Select Tags"
            :showSearchBox="args.showSearchBox"
            :showSelectedCount="args.showSelectedCount"
            :showClearButton="args.showClearButton"
            :searchPlaceholder="args.searchPlaceholder"
          />
        </div>
        
        <div class="p-4 bg-base-background rounded">
          <h4 class="font-medium mt-0">Current Selection:</h4>
          <div class="flex flex-col text-sm">
            <p>Frameworks: {{ selectedFrameworks.length > 0 ? selectedFrameworks.map(s => s.name).join(', ') : 'None' }}</p>
            <p>Projects: {{ selectedProjects.length > 0 ? selectedProjects.map(s => s.name).join(', ') : 'None' }}</p>
            <p>Tags: {{ selectedTags.length > 0 ? selectedTags.map(s => s.name).join(', ') : 'None' }}</p>
          </div>
        </div>
      </div>
    `
  }),
  args: {
    showSearchBox: false,
    showSelectedCount: false,
    showClearButton: false,
    searchPlaceholder: 'Search...'
  }
}

export const WithSearchBox: Story = {
  ...Default,
  args: {
    ...Default.args,
    showSearchBox: true
  }
}

export const WithSelectedCount: Story = {
  ...Default,
  args: {
    ...Default.args,
    showSelectedCount: true
  }
}

export const WithClearButton: Story = {
  ...Default,
  args: {
    ...Default.args,
    showClearButton: true
  }
}

export const AllHeaderFeatures: Story = {
  ...Default,
  args: {
    ...Default.args,
    showSearchBox: true,
    showSelectedCount: true,
    showClearButton: true
  }
}

export const CustomSearchPlaceholder: Story = {
  ...Default,
  args: {
    ...Default.args,
    showSearchBox: true,
    searchPlaceholder: 'Filter packages...'
  }
}

export const CustomMaxHeight: Story = {
  render: () => ({
    components: { MultiSelect },
    setup() {
      const selected1 = ref([])
      const selected2 = ref([])
      const selected3 = ref([])
      const manyOptions = Array.from({ length: 20 }, (_, i) => ({
        name: `Option ${i + 1}`,
        value: `option${i + 1}`
      }))
      return { selected1, selected2, selected3, manyOptions }
    },
    template: `
      <div class="flex gap-4">
        <div>
          <h3 class="text-sm font-semibold mb-2">Small Height (10rem)</h3>
          <MultiSelect 
            v-model="selected1" 
            :options="manyOptions" 
            label="Small Dropdown" 
            list-max-height="10rem"
            show-selected-count 
          />
        </div>
        <div>
          <h3 class="text-sm font-semibold mb-2">Default Height (28rem)</h3>
          <MultiSelect 
            v-model="selected2" 
            :options="manyOptions" 
            label="Default Dropdown"
            list-max-height="28rem"
            show-selected-count 
          />
        </div>
        <div>
          <h3 class="text-sm font-semibold mb-2">Large Height (32rem)</h3>
          <MultiSelect 
            v-model="selected3" 
            :options="manyOptions" 
            label="Large Dropdown" 
            list-max-height="32rem"
            show-selected-count 
          />
        </div>
      </div>
    `
  }),
  parameters: {
    controls: { disable: true },
    actions: { disable: true },
    slot: { disable: true }
  }
}

export const CustomMinWidth: Story = {
  render: () => ({
    components: { MultiSelect },
    setup() {
      const selected1 = ref([])
      const selected2 = ref([])
      const selected3 = ref([])
      const options = [
        { name: 'A', value: 'a' },
        { name: 'B', value: 'b' },
        { name: 'Very Long Option Name Here', value: 'long' }
      ]
      return { selected1, selected2, selected3, options }
    },
    template: `
      <div class="flex gap-4">
        <div>
          <h3 class="text-sm font-semibold mb-2">Auto Width</h3>
          <MultiSelect v-model="selected1" :options="options" label="Auto" />
        </div>
        <div>
          <h3 class="text-sm font-semibold mb-2">Min Width 18rem</h3>
          <MultiSelect v-model="selected2" :options="options" label="Min 18rem" popover-min-width="18rem" />
        </div>
        <div>
          <h3 class="text-sm font-semibold mb-2">Min Width 28rem</h3>
          <MultiSelect v-model="selected3" :options="options" label="Min 28rem" popover-min-width="28rem" />
        </div>
      </div>
    `
  }),
  parameters: {
    controls: { disable: true },
    actions: { disable: true },
    slot: { disable: true }
  }
}

export const CustomMaxWidth: Story = {
  render: () => ({
    components: { MultiSelect },
    setup() {
      const selected1 = ref([])
      const selected2 = ref([])
      const selected3 = ref([])
      const longOptions = [
        { name: 'Short', value: 'short' },
        {
          name: 'This is a very long option name that would normally expand the dropdown',
          value: 'long1'
        },
        {
          name: 'Another extremely long option that demonstrates max-width constraint',
          value: 'long2'
        }
      ]
      return { selected1, selected2, selected3, longOptions }
    },
    template: `
      <div class="flex gap-4">
        <div>
          <h3 class="text-sm font-semibold mb-2">Auto Width</h3>
          <MultiSelect v-model="selected1" :options="longOptions" label="Auto" />
        </div>
        <div>
          <h3 class="text-sm font-semibold mb-2">Max Width 18rem</h3>
          <MultiSelect v-model="selected2" :options="longOptions" label="Max 18rem" popover-max-width="18rem" />
        </div>
        <div>
          <h3 class="text-sm font-semibold mb-2">Min 12rem Max 22rem</h3>
          <MultiSelect v-model="selected3" :options="longOptions" label="Min & Max" popover-min-width="12rem" popover-max-width="22rem" />
        </div>
      </div>
    `
  }),
  parameters: {
    controls: { disable: true },
    actions: { disable: true },
    slot: { disable: true }
  }
}
