import type { Meta, StoryObj } from '@storybook/vue3-vite'
import { ref } from 'vue'

import SingleSelect from './SingleSelect.vue'

// SingleSelect already includes options prop, so no need to extend
const meta: Meta<typeof SingleSelect> = {
  title: 'Components/Input/SingleSelect',
  component: SingleSelect,
  tags: ['autodocs'],
  argTypes: {
    label: { control: 'text' },
    options: { control: 'object' },
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
    label: 'Sorting Type',
    options: [
      { name: 'Popular', value: 'popular' },
      { name: 'Newest', value: 'newest' },
      { name: 'Oldest', value: 'oldest' },
      { name: 'A → Z', value: 'az' },
      { name: 'Z → A', value: 'za' }
    ]
  }
}

export default meta
export type Story = StoryObj<typeof meta>

const sampleOptions = [
  { name: 'Popular', value: 'popular' },
  { name: 'Newest', value: 'newest' },
  { name: 'Oldest', value: 'oldest' },
  { name: 'A → Z', value: 'az' },
  { name: 'Z → A', value: 'za' }
]

export const Default: Story = {
  render: (args) => ({
    components: { SingleSelect },
    setup() {
      const selected = ref<string | null>(null)
      const options = args.options || sampleOptions
      return { selected, options, args }
    },
    template: `
      <div>
        <SingleSelect v-model="selected" :options="options" :label="args.label" />
        <div class="mt-4 p-3 bg-base-background rounded">
          <p class="text-sm">Selected: {{ selected ?? 'None' }}</p>
        </div>
      </div>
    `
  })
}

export const WithIcon: Story = {
  render: () => ({
    components: { SingleSelect },
    setup() {
      const selected = ref<string | null>('popular')
      const options = sampleOptions
      return { selected, options }
    },
    template: `
      <div>
        <SingleSelect v-model="selected" :options="options" label="Sorting Type">
          <template #icon>
            <i class="icon-[lucide--arrow-up-down] w-3.5 h-3.5" />
          </template>
        </SingleSelect>
        <div class="mt-4 p-3 bg-base-background rounded">
          <p class="text-sm">Selected: {{ selected }}</p>
        </div>
      </div>
    `
  })
}

export const Preselected: Story = {
  render: () => ({
    components: { SingleSelect },
    setup() {
      const selected = ref<string | null>('newest')
      const options = sampleOptions
      return { selected, options }
    },
    template: `
      <SingleSelect v-model="selected" :options="options" label="Sorting Type" />
    `
  })
}

export const AllVariants: Story = {
  render: () => ({
    components: { SingleSelect },
    setup() {
      const options = sampleOptions
      const a = ref<string | null>(null)
      const b = ref<string | null>('popular')
      const c = ref<string | null>('az')
      return { options, a, b, c }
    },
    template: `
      <div class="flex flex-col gap-4">
        <div class="flex items-center gap-3">
          <SingleSelect v-model="a" :options="options" label="No Icon" />
        </div>
        <div class="flex items-center gap-3">
          <SingleSelect v-model="b" :options="options" label="With Icon">
            <template #icon>
              <i class="icon-[lucide--arrow-up-down] w-3.5 h-3.5" />
            </template>
          </SingleSelect>
        </div>
        <div class="flex items-center gap-3">
          <SingleSelect v-model="c" :options="options" label="Preselected (A→Z)" />
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

export const CustomMaxHeight: Story = {
  render: () => ({
    components: { SingleSelect },
    setup() {
      const selected = ref<string | null>(null)
      const manyOptions = Array.from({ length: 20 }, (_, i) => ({
        name: `Option ${i + 1}`,
        value: `option${i + 1}`
      }))
      return { selected, manyOptions }
    },
    template: `
      <div class="flex gap-4">
        <div>
          <h3 class="text-sm font-semibold mb-2">Small Height (10rem)</h3>
          <SingleSelect v-model="selected" :options="manyOptions" label="Small Dropdown" list-max-height="10rem" />
        </div>
        <div>
          <h3 class="text-sm font-semibold mb-2">Default Height (28rem)</h3>
          <SingleSelect v-model="selected" :options="manyOptions" label="Default Dropdown" />
        </div>
        <div>
          <h3 class="text-sm font-semibold mb-2">Large Height (32rem)</h3>
          <SingleSelect v-model="selected" :options="manyOptions" label="Large Dropdown" list-max-height="32rem" />
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
    components: { SingleSelect },
    setup() {
      const selected1 = ref<string | null>(null)
      const selected2 = ref<string | null>(null)
      const selected3 = ref<string | null>(null)
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
          <SingleSelect v-model="selected1" :options="options" label="Auto" />
        </div>
        <div>
          <h3 class="text-sm font-semibold mb-2">Min Width 15rem</h3>
          <SingleSelect v-model="selected2" :options="options" label="Min 15rem" popover-min-width="15rem" />
        </div>
        <div>
          <h3 class="text-sm font-semibold mb-2">Min Width 25rem</h3>
          <SingleSelect v-model="selected3" :options="options" label="Min 25rem" popover-min-width="25rem" />
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
    components: { SingleSelect },
    setup() {
      const selected1 = ref<string | null>(null)
      const selected2 = ref<string | null>(null)
      const selected3 = ref<string | null>(null)
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
          <SingleSelect v-model="selected1" :options="longOptions" label="Auto" />
        </div>
        <div>
          <h3 class="text-sm font-semibold mb-2">Max Width 15rem</h3>
          <SingleSelect v-model="selected2" :options="longOptions" label="Max 15rem" popover-max-width="15rem" />
        </div>
        <div>
          <h3 class="text-sm font-semibold mb-2">Min 10rem Max 20rem</h3>
          <SingleSelect v-model="selected3" :options="longOptions" label="Min & Max" popover-min-width="10rem" popover-max-width="20rem" />
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
