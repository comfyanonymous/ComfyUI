import type { Meta, StoryObj } from '@storybook/vue3-vite'

import SquareChip from './SquareChip.vue'

const meta: Meta<typeof SquareChip> = {
  title: 'Components/SquareChip',
  component: SquareChip,
  tags: ['autodocs'],
  argTypes: {
    label: {
      control: 'text',
      defaultValue: 'Tag'
    }
  }
}

export default meta
type Story = StoryObj<typeof meta>

export const TagList: Story = {
  render: () => ({
    components: { SquareChip },
    template: `
      <div class="flex flex-wrap gap-2">
        <SquareChip label="JavaScript" />
        <SquareChip label="TypeScript" />
        <SquareChip label="Vue.js" />
        <SquareChip label="React" />
        <SquareChip label="Node.js" />
        <SquareChip label="Python" />
        <SquareChip label="Docker" />
        <SquareChip label="Kubernetes" />
      </div>
    `
  })
}
