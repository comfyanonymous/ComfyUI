---
globs:
  - '**/*.stories.ts'
---

# Storybook Conventions

## File Placement

Place `*.stories.ts` files alongside their components:

```
src/components/MyComponent/
├── MyComponent.vue
└── MyComponent.stories.ts
```

## Story Structure

```typescript
import type { Meta, StoryObj } from '@storybook/vue3'
import ComponentName from './ComponentName.vue'

const meta: Meta<typeof ComponentName> = {
  title: 'Category/ComponentName',
  component: ComponentName,
  parameters: { layout: 'centered' }
}

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    /* props */
  }
}
```

## Required Story Variants

Include when applicable:

- **Default** - Minimal props
- **WithData** - Realistic data
- **Loading** - Loading state
- **Error** - Error state
- **Empty** - No data

## Mock Data

Use realistic ComfyUI schemas for mocks (node definitions, components).

## Running Storybook

```bash
pnpm storybook        # Development server
pnpm build-storybook  # Production build
```
