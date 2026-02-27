import type {
  Meta,
  StoryObj,
  ComponentPropsAndSlots
} from '@storybook/vue3-vite'

import Button from './Button.vue'
import { FOR_STORIES } from '@/components/ui/button/button.variants'

interface ButtonPropsAndStoryArgs extends ComponentPropsAndSlots<
  typeof Button
> {
  icon?: 'left' | 'right'
}

const { variants, sizes } = FOR_STORIES
const meta: Meta<ButtonPropsAndStoryArgs> = {
  title: 'Components/Button/Button',
  component: Button,
  tags: ['autodocs'],
  argTypes: {
    size: {
      control: { type: 'select' },
      options: sizes,
      defaultValue: 'md'
    },
    variant: {
      control: { type: 'select' },
      options: variants,
      defaultValue: 'primary'
    },
    as: { defaultValue: 'button' },
    asChild: { defaultValue: false },
    default: {
      control: { type: 'text' },
      defaultValue: 'Button'
    },
    icon: {
      control: { type: 'select' },
      options: [undefined, 'left', 'right']
    }
  },
  args: {
    variant: 'secondary',
    size: 'md',
    default: 'Button',
    icon: undefined
  }
}

export default meta
type Story = StoryObj<typeof meta>

export const SingleButton: Story = {
  render: (args) => ({
    components: { Button },
    setup() {
      return { args }
    },
    template: `
    <Button v-bind="args">
      <i v-if="args.icon === 'left'" class="icon-[lucide--settings]" />
      {{args.default}}
      <i v-if="args.icon === 'right'" class="icon-[lucide--settings]" />
    </Button>`
  })
}

function generateVariants() {
  const variantButtons: string[] = []
  for (const variant of variants) {
    for (const size of sizes) {
      variantButtons.push(
        `<Button 
            variant="${variant}"
            size="${size}">${
              size.startsWith('icon')
                ? `<i class="icon-[lucide--settings]" />`
                : variant
            }</Button>`
      )
    }
  }
  return variantButtons
}

// Note: Keep the number of columns here aligned with the number of sizes above.
export const AllVariants: Story = {
  render: () => ({
    components: { Button },
    template: `
      <div class="grid grid-cols-5 gap-4 place-items-center-safe">
      ${generateVariants().join('\n')}
        
      </div>
    `
  })
}
