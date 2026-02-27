import type { Meta, StoryObj } from '@storybook/vue3-vite'
import { ref } from 'vue'

import Select from './Select.vue'
import SelectContent from './SelectContent.vue'
import SelectGroup from './SelectGroup.vue'
import SelectItem from './SelectItem.vue'
import SelectLabel from './SelectLabel.vue'
import SelectSeparator from './SelectSeparator.vue'
import SelectTrigger from './SelectTrigger.vue'
import SelectValue from './SelectValue.vue'

const meta = {
  title: 'Components/Select',
  component: Select,
  tags: ['autodocs'],
  argTypes: {
    modelValue: {
      control: 'text',
      description: 'Selected value'
    },
    disabled: {
      control: 'boolean',
      description: 'When true, disables the select'
    },
    'onUpdate:modelValue': { action: 'update:modelValue' }
  }
} satisfies Meta<typeof Select>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: (args) => ({
    components: {
      Select,
      SelectContent,
      SelectItem,
      SelectTrigger,
      SelectValue
    },
    setup() {
      const value = ref(args.modelValue || '')
      return { value, args }
    },
    template: `
      <Select v-model="value" :disabled="args.disabled">
        <SelectTrigger class="w-56">
          <SelectValue placeholder="Select a fruit" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="apple">Apple</SelectItem>
          <SelectItem value="banana">Banana</SelectItem>
          <SelectItem value="cherry">Cherry</SelectItem>
          <SelectItem value="grape">Grape</SelectItem>
          <SelectItem value="orange">Orange</SelectItem>
        </SelectContent>
      </Select>
      <div class="mt-4 text-sm text-muted-foreground">
        Selected: {{ value || 'None' }}
      </div>
    `
  }),
  args: {
    disabled: false
  }
}

export const WithPlaceholder: Story = {
  render: (args) => ({
    components: {
      Select,
      SelectContent,
      SelectItem,
      SelectTrigger,
      SelectValue
    },
    setup() {
      const value = ref('')
      return { value, args }
    },
    template: `
      <Select v-model="value" :disabled="args.disabled">
        <SelectTrigger class="w-56">
          <SelectValue placeholder="Choose an option..." />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="option1">Option 1</SelectItem>
          <SelectItem value="option2">Option 2</SelectItem>
          <SelectItem value="option3">Option 3</SelectItem>
        </SelectContent>
      </Select>
    `
  }),
  args: {
    disabled: false
  }
}

export const Disabled: Story = {
  render: (args) => ({
    components: {
      Select,
      SelectContent,
      SelectItem,
      SelectTrigger,
      SelectValue
    },
    setup() {
      const value = ref('apple')
      return { value, args }
    },
    template: `
      <Select v-model="value" disabled>
        <SelectTrigger class="w-56">
          <SelectValue placeholder="Select a fruit" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="apple">Apple</SelectItem>
          <SelectItem value="banana">Banana</SelectItem>
          <SelectItem value="cherry">Cherry</SelectItem>
        </SelectContent>
      </Select>
    `
  })
}

export const WithGroups: Story = {
  render: (args) => ({
    components: {
      Select,
      SelectContent,
      SelectGroup,
      SelectItem,
      SelectLabel,
      SelectSeparator,
      SelectTrigger,
      SelectValue
    },
    setup() {
      const value = ref('')
      return { value, args }
    },
    template: `
      <Select v-model="value" :disabled="args.disabled">
        <SelectTrigger class="w-56">
          <SelectValue placeholder="Select a model type" />
        </SelectTrigger>
        <SelectContent>
          <SelectGroup>
            <SelectLabel>Checkpoints</SelectLabel>
            <SelectItem value="sd15">SD 1.5</SelectItem>
            <SelectItem value="sdxl">SDXL</SelectItem>
            <SelectItem value="flux">Flux</SelectItem>
          </SelectGroup>
          <SelectSeparator />
          <SelectGroup>
            <SelectLabel>LoRAs</SelectLabel>
            <SelectItem value="lora-style">Style LoRA</SelectItem>
            <SelectItem value="lora-character">Character LoRA</SelectItem>
          </SelectGroup>
          <SelectSeparator />
          <SelectGroup>
            <SelectLabel>Other</SelectLabel>
            <SelectItem value="vae">VAE</SelectItem>
            <SelectItem value="embedding">Embedding</SelectItem>
          </SelectGroup>
        </SelectContent>
      </Select>
      <div class="mt-4 text-sm text-muted-foreground">
        Selected: {{ value || 'None' }}
      </div>
    `
  }),
  args: {
    disabled: false
  }
}

export const Scrollable: Story = {
  render: (args) => ({
    components: {
      Select,
      SelectContent,
      SelectItem,
      SelectTrigger,
      SelectValue
    },
    setup() {
      const value = ref('')
      const items = Array.from({ length: 20 }, (_, i) => ({
        value: `item-${i + 1}`,
        label: `Option ${i + 1}`
      }))
      return { value, items, args }
    },
    template: `
      <Select v-model="value" :disabled="args.disabled">
        <SelectTrigger class="w-56">
          <SelectValue placeholder="Select an option" />
        </SelectTrigger>
        <SelectContent>
          <SelectItem
            v-for="item in items"
            :key="item.value"
            :value="item.value"
          >
            {{ item.label }}
          </SelectItem>
        </SelectContent>
      </Select>
    `
  }),
  args: {
    disabled: false
  }
}

export const CustomWidth: Story = {
  render: (args) => ({
    components: {
      Select,
      SelectContent,
      SelectItem,
      SelectTrigger,
      SelectValue
    },
    setup() {
      const value = ref('')
      return { value, args }
    },
    template: `
      <div class="space-y-4">
        <Select v-model="value" :disabled="args.disabled">
          <SelectTrigger class="w-32">
            <SelectValue placeholder="Small" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="a">A</SelectItem>
            <SelectItem value="b">B</SelectItem>
            <SelectItem value="c">C</SelectItem>
          </SelectContent>
        </Select>

        <Select v-model="value" :disabled="args.disabled">
          <SelectTrigger class="w-full">
            <SelectValue placeholder="Full width select" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="option1">Option 1</SelectItem>
            <SelectItem value="option2">Option 2</SelectItem>
            <SelectItem value="option3">Option 3</SelectItem>
          </SelectContent>
        </Select>
      </div>
    `
  }),
  args: {
    disabled: false
  }
}
