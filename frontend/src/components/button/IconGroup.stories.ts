import type { Meta, StoryObj } from '@storybook/vue3-vite'

import Button from '@/components/ui/button/Button.vue'
import IconGroup from './IconGroup.vue'

const meta: Meta<typeof IconGroup> = {
  title: 'Components/Button/IconGroup',
  component: IconGroup,
  parameters: {
    layout: 'centered'
  }
}
export default meta

type Story = StoryObj<typeof IconGroup>

export const Basic: Story = {
  render: () => ({
    components: { IconGroup, Button },
    template: `
      <IconGroup>
        <Button size="icon" @click="console.log('Hello World!!')">
          <i class="icon-[lucide--heart] size-4" />
        </Button>
        <Button size="icon" @click="console.log('Hello World!!')">
          <i class="icon-[lucide--download] size-4" />
        </Button>
        <Button size="icon" @click="console.log('Hello World!!')">
          <i class="icon-[lucide--external-link] size-4" />
        </Button>
      </IconGroup>
    `
  })
}
