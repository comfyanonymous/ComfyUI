import type { Meta, StoryObj } from '@storybook/vue3-vite'
import { ref } from 'vue'

import SearchBox from '@/components/common/SearchBox.vue'
import type { ComponentExposed } from 'vue-component-type-helpers'
interface GenericMeta<C> extends Omit<Meta<C>, 'component'> {
  component: Omit<ComponentExposed<C>, 'focus'>
}

const meta: GenericMeta<typeof SearchBox> = {
  title: 'Components/Input/SearchBox',
  component: SearchBox,
  tags: ['autodocs'],
  argTypes: {
    modelValue: {
      control: 'text'
    },
    placeholder: {
      control: 'text'
    },
    showBorder: {
      control: 'boolean',
      description: 'Toggle border prop'
    },
    size: {
      control: 'select',
      options: ['md', 'lg'],
      description: 'Size variant of the search box'
    },
    'onUpdate:modelValue': { action: 'update:modelValue' },
    onSearch: { action: 'search' }
  },
  args: {
    modelValue: '',
    placeholder: 'Search...',
    showBorder: false,
    size: 'md'
  }
}

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: (args) => ({
    components: { SearchBox },
    setup() {
      const searchText = ref('')
      return { searchText, args }
    },
    template: `
      <div style="max-width: 320px;">
        <SearchBox v-bind="args" v-model="searchText" />
      </div>
    `
  })
}

export const WithBorder: Story = {
  ...Default,
  args: {
    showBorder: true
  }
}

export const NoBorder: Story = {
  ...Default,
  args: {
    showBorder: false
  }
}

export const MediumSize: Story = {
  ...Default,
  args: {
    size: 'md',
    showBorder: false
  }
}

export const LargeSize: Story = {
  ...Default,
  args: {
    size: 'lg',
    showBorder: false
  }
}

export const LargeSizeWithBorder: Story = {
  ...Default,
  args: {
    size: 'lg',
    showBorder: true
  }
}
