import type { Meta, StoryObj } from '@storybook/vue3-vite'
import { ref } from 'vue'

import ToggleGroup from './ToggleGroup.vue'
import ToggleGroupItem from './ToggleGroupItem.vue'

const meta = {
  title: 'Components/ToggleGroup',
  component: ToggleGroup,
  tags: ['autodocs'],
  argTypes: {
    type: {
      control: 'select',
      options: ['single', 'multiple'],
      description: 'Single or multiple selection'
    },
    variant: {
      control: 'select',
      options: ['default', 'outline'],
      description: 'Visual style variant'
    },
    disabled: {
      control: 'boolean',
      description: 'When true, disables the toggle group'
    },
    'onUpdate:modelValue': { action: 'update:modelValue' }
  }
} satisfies Meta<typeof ToggleGroup>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: (args) => ({
    components: { ToggleGroup, ToggleGroupItem },
    setup() {
      const value = ref('center')
      return { value, args }
    },
    template: `
      <ToggleGroup
        v-model="value"
        type="single"
        :variant="args.variant"
        :disabled="args.disabled"
        class="border border-border-default rounded-lg p-1"
      >
        <ToggleGroupItem value="left">Left</ToggleGroupItem>
        <ToggleGroupItem value="center">Center</ToggleGroupItem>
        <ToggleGroupItem value="right">Right</ToggleGroupItem>
      </ToggleGroup>
      <div class="mt-4 text-sm text-muted-foreground">
        Selected: {{ value || 'None' }}
      </div>
    `
  }),
  args: {
    variant: 'default',
    disabled: false
  }
}

export const Disabled: Story = {
  render: (args) => ({
    components: { ToggleGroup, ToggleGroupItem },
    setup() {
      const value = ref('on')
      return { value, args }
    },
    template: `
      <ToggleGroup v-model="value" type="single" disabled class="border border-border-default rounded-lg p-1">
        <ToggleGroupItem value="off">Off</ToggleGroupItem>
        <ToggleGroupItem value="on">On</ToggleGroupItem>
      </ToggleGroup>
    `
  }),
  args: {}
}

export const OutlineVariant: Story = {
  render: (args) => ({
    components: { ToggleGroup, ToggleGroupItem },
    setup() {
      const value = ref('medium')
      return { value, args }
    },
    template: `
      <ToggleGroup v-model="value" type="single" variant="outline" class="gap-2">
        <ToggleGroupItem value="small" size="sm">Small</ToggleGroupItem>
        <ToggleGroupItem value="medium">Medium</ToggleGroupItem>
        <ToggleGroupItem value="large" size="lg">Large</ToggleGroupItem>
      </ToggleGroup>
    `
  }),
  args: {}
}

export const BooleanToggle: Story = {
  render: (args) => ({
    components: { ToggleGroup, ToggleGroupItem },
    setup() {
      const value = ref('off')
      return { value, args }
    },
    template: `
      <div class="space-y-4">
        <p class="text-sm text-muted-foreground">Boolean toggle with custom labels:</p>
        <ToggleGroup
          v-model="value"
          type="single"
          class="w-48 border border-border-default rounded-lg p-1"
        >
          <ToggleGroupItem value="off" size="sm">Outside</ToggleGroupItem>
          <ToggleGroupItem value="on" size="sm">Inside</ToggleGroupItem>
        </ToggleGroup>
        <div class="text-sm">Value: {{ value === 'on' ? true : false }}</div>
      </div>
    `
  }),
  args: {}
}

export const LongLabels: Story = {
  render: (args) => ({
    components: { ToggleGroup, ToggleGroupItem },
    setup() {
      const value = ref('option1')
      return { value, args }
    },
    template: `
      <div class="w-64">
        <p class="text-sm text-muted-foreground mb-2">Labels truncate with ellipsis:</p>
        <ToggleGroup
          v-model="value"
          type="single"
          class="border border-border-default rounded-lg p-1"
        >
          <ToggleGroupItem value="option1" size="sm">Very Long Label One</ToggleGroupItem>
          <ToggleGroupItem value="option2" size="sm">Another Long Label</ToggleGroupItem>
        </ToggleGroup>
      </div>
    `
  }),
  args: {}
}

export const Sizes: Story = {
  render: () => ({
    components: { ToggleGroup, ToggleGroupItem },
    setup() {
      const sm = ref('a')
      const md = ref('a')
      const lg = ref('a')
      return { sm, md, lg }
    },
    template: `
      <div class="space-y-4">
        <div>
          <p class="text-sm text-muted-foreground mb-2">Small:</p>
          <ToggleGroup v-model="sm" type="single" class="border border-border-default rounded p-1">
            <ToggleGroupItem value="a" size="sm">A</ToggleGroupItem>
            <ToggleGroupItem value="b" size="sm">B</ToggleGroupItem>
          </ToggleGroup>
        </div>
        <div>
          <p class="text-sm text-muted-foreground mb-2">Default:</p>
          <ToggleGroup v-model="md" type="single" class="border border-border-default rounded-lg p-1">
            <ToggleGroupItem value="a">A</ToggleGroupItem>
            <ToggleGroupItem value="b">B</ToggleGroupItem>
          </ToggleGroup>
        </div>
        <div>
          <p class="text-sm text-muted-foreground mb-2">Large:</p>
          <ToggleGroup v-model="lg" type="single" class="border border-border-default rounded-lg p-1">
            <ToggleGroupItem value="a" size="lg">A</ToggleGroupItem>
            <ToggleGroupItem value="b" size="lg">B</ToggleGroupItem>
          </ToggleGroup>
        </div>
      </div>
    `
  })
}
