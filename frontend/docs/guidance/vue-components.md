---
globs:
  - '**/*.vue'
---

# Vue Component Conventions

Applies to all `.vue` files anywhere in the codebase.

## Vue 3 Composition API

- Use `<script setup lang="ts">` for component logic
- Destructure props (Vue 3.5 style with defaults) like `const { color = 'blue' } = defineProps<...>()`
- Use `ref`/`reactive` for state
- Use `computed()` for derived state
- Use lifecycle hooks: `onMounted`, `onUpdated`, etc.

## Component Communication

- Prefer `emit/@event-name` for state changes (promotes loose coupling)
- Use `defineExpose` only for imperative operations (`form.validate()`, `modal.open()`)
- Proper props and emits definitions

## VueUse Composables

Prefer VueUse composables over manual event handling:

- `useElementHover` instead of manual mouseover/mouseout listeners
- `useIntersectionObserver` for visibility detection instead of scroll handlers
- `useFocusTrap` for modal/dialog focus management
- `useEventListener` for auto-cleanup event listeners

Prefer Vue native options when available:

- `defineModel` instead of `useVModel` for two-way binding with props

## Styling

- Use inline Tailwind CSS only (no `<style>` blocks)
- Use `cn()` from `@/utils/tailwindUtil` for conditional classes
- Refer to packages/design-system/src/css/style.css for design tokens and tailwind configuration
- Exception: when third-party libraries render runtime DOM outside Vue templates
  (for example xterm internals inside PrimeVue terminal wrappers), scoped
  `:deep()` selectors are allowed. Add a brief inline comment explaining why the
  exception is required.

## Best Practices

- Extract complex conditionals to `computed`
- In unmounted hooks, implement cleanup for async operations
