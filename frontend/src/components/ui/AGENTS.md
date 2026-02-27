# UI Component Guidelines

## Adding New Components

```bash
pnpm dlx shadcn-vue@latest add <component-name> --yes
```

After adding, create `ComponentName.stories.ts` with Default, Disabled, and variant stories.

## Reka UI Wrapper Components

- Use reactive props destructuring with rest: `const { class: className, ...restProps } = defineProps<Props>()`
- Use `useForwardProps(restProps)` for prop forwarding, or `computed()` if adding defaults
- Import siblings directly (`./Component.vue`), not from barrel (`'.'`)
- Use `cn()` for class merging with `className`
- Use Iconify icons: `<i class="icon-[lucide--check]" />`
- Use design tokens: `bg-secondary-background`, `text-muted-foreground`, `border-border-default`
- Tailwind 4 CSS variables use parentheses: `h-(--my-var)` not `h-[--my-var]`
