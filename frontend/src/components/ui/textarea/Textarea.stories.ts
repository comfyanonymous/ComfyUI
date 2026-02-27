import type { Meta, StoryObj } from '@storybook/vue3-vite'
import { ref } from 'vue'

import Textarea from './Textarea.vue'

const meta: Meta<typeof Textarea> = {
  title: 'UI/Textarea',
  component: Textarea,
  tags: ['autodocs']
}

export default meta
type Story = StoryObj<typeof Textarea>

export const Default: Story = {
  render: () => ({
    components: { Textarea },
    setup() {
      const value = ref('Hello world')
      return { value }
    },
    template:
      '<Textarea v-model="value" placeholder="Type something..." class="max-w-sm" />'
  })
}

export const Disabled: Story = {
  render: () => ({
    components: { Textarea },
    template:
      '<Textarea model-value="Disabled textarea" disabled class="max-w-sm" />'
  })
}

export const WithLabel: Story = {
  render: () => ({
    components: { Textarea },
    setup() {
      const value = ref('Content that sits below the label')
      return { value }
    },
    template: `
      <div class="relative max-w-sm rounded-lg bg-component-node-widget-background">
        <label class="pointer-events-none absolute left-3 top-1.5 text-xxs text-muted-foreground z-10">
          Prompt
        </label>
        <Textarea
          v-model="value"
          class="size-full resize-none border-none bg-transparent pt-5 text-xs"
        />
      </div>
    `
  })
}
