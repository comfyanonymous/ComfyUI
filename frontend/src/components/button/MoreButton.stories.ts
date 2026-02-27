import type { Meta, StoryObj } from '@storybook/vue3-vite'

import Button from '@/components/ui/button/Button.vue'
import MoreButton from './MoreButton.vue'

const meta: Meta<typeof MoreButton> = {
  title: 'Components/Button/MoreButton',
  component: MoreButton,
  parameters: {
    layout: 'centered'
  },
  argTypes: {}
}
export default meta

type Story = StoryObj<typeof MoreButton>

export const Basic: Story = {
  render: () => ({
    components: { MoreButton, Button },
    template: `
      <div style="height: 200px; display: flex; align-items: center; justify-content: center;">
        <MoreButton>
          <template #default="{ close }">
            <Button
              variant="textonly"
              @click="() => { close() }"
            >
              <i class="icon-[lucide--download] size-4" />
              <span>Settings</span>
            </Button>

            <Button
              variant="textonly"
              @click="() => { close() }"
            >
              <i class="icon-[lucide--scroll-text] size-4" />
              <span>Profile</span>
            </Button>
          </template>
        </MoreButton>
      </div>
    `
  })
}
