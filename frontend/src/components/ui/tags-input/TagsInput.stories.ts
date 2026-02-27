import type { Meta, StoryObj } from '@storybook/vue3-vite'
import type { ComponentExposed } from 'vue-component-type-helpers'
import { ref } from 'vue'

import TagsInput from './TagsInput.vue'
import TagsInputInput from './TagsInputInput.vue'
import TagsInputItem from './TagsInputItem.vue'
import TagsInputItemDelete from './TagsInputItemDelete.vue'
import TagsInputItemText from './TagsInputItemText.vue'

interface GenericMeta<C> extends Omit<Meta<C>, 'component'> {
  component: ComponentExposed<C>
}

const meta: GenericMeta<typeof TagsInput> = {
  title: 'Components/TagsInput',
  component: TagsInput,
  tags: ['autodocs'],
  argTypes: {
    modelValue: {
      control: 'object',
      description: 'Array of tag values'
    },
    disabled: {
      control: 'boolean',
      description:
        'When true, completely disables the component. When false (default), shows read-only state with edit icon until clicked.'
    },
    'onUpdate:modelValue': { action: 'update:modelValue' }
  }
}

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  render: (args) => ({
    components: {
      TagsInput,
      TagsInputInput,
      TagsInputItem,
      TagsInputItemDelete,
      TagsInputItemText
    },
    setup() {
      const tags = ref(args.modelValue || ['tag1', 'tag2'])
      return { tags, args }
    },
    template: `
      <TagsInput v-model="tags" :disabled="args.disabled" class="w-80" v-slot="{ isEmpty }">
        <TagsInputItem v-for="tag in tags" :key="tag" :value="tag">
          <TagsInputItemText />
          <TagsInputItemDelete />
        </TagsInputItem>
        <TagsInputInput :is-empty="isEmpty" placeholder="Add tag..." />
      </TagsInput>
      <div class="mt-4 text-sm text-muted-foreground">
        Tags: {{ tags.join(', ') }}
      </div>
    `
  }),
  args: {
    modelValue: ['Vue', 'TypeScript'],
    disabled: false
  }
}

export const Empty: Story = {
  args: {
    disabled: false
  },
  render: (args) => ({
    components: {
      TagsInput,
      TagsInputInput,
      TagsInputItem,
      TagsInputItemDelete,
      TagsInputItemText
    },
    setup() {
      const tags = ref<string[]>([])
      return { tags, args }
    },
    template: `
      <TagsInput v-model="tags" :disabled="args.disabled" class="w-80" v-slot="{ isEmpty }">
        <TagsInputItem v-for="tag in tags" :key="tag" :value="tag">
          <TagsInputItemText />
          <TagsInputItemDelete />
        </TagsInputItem>
        <TagsInputInput :is-empty="isEmpty" placeholder="Start typing to add tags..." />
      </TagsInput>
    `
  })
}

export const ManyTags: Story = {
  render: (args) => ({
    components: {
      TagsInput,
      TagsInputInput,
      TagsInputItem,
      TagsInputItemDelete,
      TagsInputItemText
    },
    setup() {
      const tags = ref([
        'JavaScript',
        'TypeScript',
        'Vue',
        'React',
        'Svelte',
        'Node.js',
        'Python',
        'Rust'
      ])
      return { tags, args }
    },
    template: `
      <TagsInput v-model="tags" :disabled="args.disabled" class="w-96" v-slot="{ isEmpty }">
        <TagsInputItem v-for="tag in tags" :key="tag" :value="tag">
          <TagsInputItemText />
          <TagsInputItemDelete />
        </TagsInputItem>
        <TagsInputInput :is-empty="isEmpty" placeholder="Add technology..." />
      </TagsInput>
    `
  })
}

export const Disabled: Story = {
  args: {
    disabled: true
  },

  render: (args) => ({
    components: {
      TagsInput,
      TagsInputInput,
      TagsInputItem,
      TagsInputItemDelete,
      TagsInputItemText
    },
    setup() {
      const tags = ref(['Read', 'Only', 'Tags'])
      return { tags, args }
    },
    template: `
      <TagsInput v-model="tags" :disabled="args.disabled" class="w-80" v-slot="{ isEmpty }">
        <TagsInputItem v-for="tag in tags" :key="tag" :value="tag">
          <TagsInputItemText />
          <TagsInputItemDelete />
        </TagsInputItem>
        <TagsInputInput :is-empty="isEmpty" placeholder="Cannot add tags..." />
      </TagsInput>
    `
  })
}

export const CustomWidth: Story = {
  render: (args) => ({
    components: {
      TagsInput,
      TagsInputInput,
      TagsInputItem,
      TagsInputItemDelete,
      TagsInputItemText
    },
    setup() {
      const tags = ref(['Full', 'Width'])
      return { tags, args }
    },
    template: `
      <TagsInput v-model="tags" :disabled="args.disabled" class="w-full" v-slot="{ isEmpty }">
        <TagsInputItem v-for="tag in tags" :key="tag" :value="tag">
          <TagsInputItemText />
          <TagsInputItemDelete />
        </TagsInputItem>
        <TagsInputInput :is-empty="isEmpty" placeholder="Add tag..." />
      </TagsInput>
    `
  })
}
