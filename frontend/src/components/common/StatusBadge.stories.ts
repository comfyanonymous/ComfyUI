import type { Meta, StoryObj } from '@storybook/vue3-vite'

import StatusBadge from './StatusBadge.vue'

const meta = {
  title: 'Common/StatusBadge',
  component: StatusBadge,
  tags: ['autodocs'],
  argTypes: {
    label: { control: 'text' },
    severity: {
      control: 'select',
      options: ['default', 'secondary', 'warn', 'danger', 'contrast']
    },
    variant: {
      control: 'select',
      options: ['label', 'dot', 'circle']
    }
  },
  args: {
    label: 'Status',
    severity: 'default'
  }
} satisfies Meta<typeof StatusBadge>

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {}

export const Failed: Story = {
  args: {
    label: 'Failed',
    severity: 'danger'
  }
}

export const Finished: Story = {
  args: {
    label: 'Finished',
    severity: 'contrast'
  }
}

export const Dot: Story = {
  args: {
    label: undefined,
    variant: 'dot',
    severity: 'danger'
  }
}

export const Circle: Story = {
  args: {
    label: '3',
    variant: 'circle'
  }
}

export const AllSeverities: Story = {
  render: () => ({
    components: { StatusBadge },
    template: `
      <div class="flex items-center gap-2">
        <StatusBadge label="Default" severity="default" />
        <StatusBadge label="Secondary" severity="secondary" />
        <StatusBadge label="Warn" severity="warn" />
        <StatusBadge label="Danger" severity="danger" />
        <StatusBadge label="Contrast" severity="contrast" />
      </div>
    `
  })
}

export const AllVariants: Story = {
  render: () => ({
    components: { StatusBadge },
    template: `
      <div class="flex items-center gap-4">
        <div class="flex flex-col items-center gap-1">
          <StatusBadge label="Label" variant="label" />
          <span class="text-xs text-muted">label</span>
        </div>
        <div class="flex flex-col items-center gap-1">
          <StatusBadge variant="dot" severity="danger" />
          <span class="text-xs text-muted">dot</span>
        </div>
        <div class="flex flex-col items-center gap-1">
          <StatusBadge label="5" variant="circle" />
          <span class="text-xs text-muted">circle</span>
        </div>
      </div>
    `
  })
}
