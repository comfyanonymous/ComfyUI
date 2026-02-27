// eslint-disable-next-line storybook/no-renderer-packages
import type { Meta, StoryObj } from '@storybook/vue3'

import HardwareOption from './HardwareOption.vue'

const meta: Meta<typeof HardwareOption> = {
  title: 'Desktop/Components/HardwareOption',
  component: HardwareOption,
  parameters: {
    layout: 'centered',
    backgrounds: {
      default: 'dark',
      values: [{ name: 'dark', value: '#1a1a1a' }]
    }
  },
  argTypes: {
    selected: { control: 'boolean' },
    imagePath: { control: 'text' },
    placeholderText: { control: 'text' },
    subtitle: { control: 'text' }
  }
}

export default meta
type Story = StoryObj<typeof meta>

export const AppleMetalSelected: Story = {
  args: {
    imagePath: '/assets/images/apple-mps-logo.png',
    placeholderText: 'Apple Metal',
    subtitle: 'Apple Metal',
    selected: true
  }
}

export const AppleMetalUnselected: Story = {
  args: {
    imagePath: '/assets/images/apple-mps-logo.png',
    placeholderText: 'Apple Metal',
    subtitle: 'Apple Metal',
    selected: false
  }
}

export const CPUOption: Story = {
  args: {
    placeholderText: 'CPU',
    subtitle: 'Subtitle',
    selected: false
  }
}

export const ManualInstall: Story = {
  args: {
    placeholderText: 'Manual Install',
    subtitle: 'Subtitle',
    selected: false
  }
}

export const NvidiaSelected: Story = {
  args: {
    imagePath: '/assets/images/nvidia-logo-square.jpg',
    placeholderText: 'NVIDIA',
    subtitle: 'NVIDIA',
    selected: true
  }
}
