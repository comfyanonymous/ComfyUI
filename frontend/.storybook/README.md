# Storybook Configuration for ComfyUI Frontend

## What is Storybook?

Storybook is a frontend workshop for building UI components and pages in isolation. It allows developers to:

- Build components independently from the main application
- Test components with different props and states
- Document component APIs and usage patterns
- Share components across teams and projects
- Catch visual regressions through visual testing

## Storybook vs Other Testing Tools

| Tool                    | Purpose                             | Use Case                                                     |
| ----------------------- | ----------------------------------- | ------------------------------------------------------------ |
| **Storybook**           | Component isolation & documentation | Developing, testing, and showcasing individual UI components |
| **Playwright**          | End-to-end testing                  | Full user workflow testing across multiple pages             |
| **Vitest**              | Unit testing                        | Testing business logic, utilities, and component behavior    |
| **Vue Testing Library** | Component testing                   | Testing component interactions and DOM output                |

### When to Use Storybook

**✅ Use Storybook for:**

- Developing new UI components in isolation
- Creating component documentation and examples
- Testing different component states and props
- Sharing components with designers and stakeholders
- Visual regression testing
- Building a component library or design system

**❌ Don't use Storybook for:**

- Testing complex user workflows (use Playwright)
- Testing business logic (use Vitest)
- Integration testing between components (use Vue Testing Library)

## How to Use Storybook

### Development Commands

```bash
# Start Storybook development server
pnpm storybook

# Build static Storybook for deployment
pnpm build-storybook
```

### Creating Stories

Stories are located alongside components in `src/` directories with the pattern `*.stories.ts`:

```typescript
// MyComponent.stories.ts
import type { Meta, StoryObj } from '@storybook/vue3'
import MyComponent from './MyComponent.vue'

const meta: Meta<typeof MyComponent> = {
  title: 'Components/MyComponent',
  component: MyComponent,
  parameters: {
    layout: 'centered'
  }
}

export default meta
type Story = StoryObj<typeof meta>

export const Default: Story = {
  args: {
    title: 'Hello World'
  }
}

export const WithVariant: Story = {
  args: {
    title: 'Variant Example',
    variant: 'secondary'
  }
}
```

### Available Features

- **Vue 3 Support**: Full Vue 3 composition API and reactivity
- **PrimeVue Integration**: All PrimeVue components and theming
- **ComfyUI Theming**: Custom ComfyUI theme preset applied
- **Pinia Stores**: Access to application stores for components that need state
- **TypeScript**: Full TypeScript support with proper type checking
- **CSS/SCSS**: Component styling support
- **Auto-documentation**: Automatic prop tables and component documentation
- **Chromatic Integration**: Automated visual regression testing for component stories

## Development Tips

## ComfyUI Storybook Guidelines

### Scope – When to Create Stories

- **PrimeVue components**:  
  No need to create stories. Just refer to the official PrimeVue documentation.
- **Custom shared components (design system components)**:  
  Always create stories. These components are built in collaboration with designers, and Storybook serves as both documentation and a communication tool.
- **Container components (logic-heavy)**:  
  Do not create stories. Only the underlying pure UI components should be included in Storybook.

### Maintenance Philosophy

- Stories are lightweight and generally stable.  
  Once created, they rarely need updates unless:
  - The design changes
  - New props (e.g. size, color variants) are introduced
- For existing usage patterns, simply copy real code examples into Storybook to create stories.

### File Placement

- Keep `*.stories.ts` files at the **same level as the component** (similar to test files).
- This makes it easier to check usage examples without navigating to another directory.

### Developer/Designer Workflow

- **UI vs Container**: Separate pure UI components from container components.  
  Only UI components should live in Storybook.
- **Communication Tool**: Storybook is not just about code quality—it enables designers and developers to see:
  - Which props exist
  - What cases are covered
  - How variants (e.g. size, colors) look in isolation
- **Example**:  
  `PackActionButton.vue` wraps a PrimeVue button with additional logic.  
  → Only create a story for the base UI button, not for the wrapper.

### Suggested Workflow

1. Use PrimeVue docs for standard components
2. Use Storybook for **shared/custom components** that define our design system
3. Keep story files alongside components
4. When in doubt, focus on components reused across the app or those that need to be showcased to designers

### Best Practices

1. **Keep Stories Simple**: Each story should demonstrate one specific use case
2. **Use Realistic Data**: Use data that resembles real application usage
3. **Document Edge Cases**: Create stories for loading states, errors, and edge cases
4. **Group Related Stories**: Use consistent naming and grouping for related components

### Component Testing Strategy

```typescript
// Example: Testing different component states
export const Loading: Story = {
  args: {
    isLoading: true
  }
}

export const Error: Story = {
  args: {
    error: 'Failed to load data'
  }
}

export const WithLongText: Story = {
  args: {
    description: 'Very long description that might cause layout issues...'
  }
}
```

### Debugging Tips

- Use browser DevTools to inspect component behavior
- Check the Storybook console for Vue warnings or errors
- Use the Controls addon to dynamically change props
- Leverage the Actions addon to test event handling

## Configuration Files

- **`main.ts`**: Core Storybook configuration and Vite integration
- **`preview.ts`**: Global decorators, parameters, and Vue app setup
- **`manager.ts`**: Storybook UI customization (if needed)
- **`preview-head.html`**: Injects custom HTML into the `<head>` of every Storybook iframe (used for global styles, fonts, or fixes for iframe-specific issues)

## Chromatic Visual Testing

This project uses [Chromatic](https://chromatic.com) for automated visual regression testing of Storybook components.

### How It Works

- **Automated Testing**: Every push to `main` and `sno-storybook` branches triggers Chromatic builds
- **Pull Request Reviews**: PRs against `main` branch include visual diffs for component changes
- **Baseline Management**: Changes on `main` branch are automatically accepted as new baselines
- **Cross-browser Testing**: Tests across multiple browsers and viewports

### Viewing Results

1. Check the GitHub Actions tab for Chromatic workflow status
2. Click on the Chromatic build link in PR comments to review visual changes
3. Accept or reject visual changes directly in the Chromatic UI

### Best Practices for Visual Testing

- **Consistent Stories**: Ensure stories render consistently across different environments
- **Meaningful Names**: Use descriptive story names that clearly indicate the component state
- **Edge Cases**: Include stories for loading, error, and empty states
- **Realistic Data**: Use data that closely resembles production usage

## Integration with ComfyUI

This Storybook setup includes:

- ComfyUI-specific theming and styling
- Pre-configured Pinia stores for state management
- Internationalization (i18n) support
- PrimeVue component library integration
- Proper alias resolution for `@/` imports

## Icon Usage in Storybook

In this project, only the `<i class="icon-[lucide--folder]" />` syntax from unplugin-icons is supported in Storybook.

**Example:**

```vue
<script setup lang="ts"></script>

<template>
  <i class="icon-[lucide--trophy] text-neutral size-4" />
  <i class="icon-[lucide--settings] text-neutral size-4" />
</template>
```

This approach ensures icons render correctly in Storybook and remain consistent with the rest of the app.
