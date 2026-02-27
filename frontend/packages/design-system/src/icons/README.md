# ComfyUI Icons Guide

ComfyUI supports three types of icons that can be used throughout the interface. All icons are automatically imported - no manual imports needed!

## Quick Start - Code Examples

### 1. PrimeIcons

```vue
<template>
  <!-- Basic usage -->
  <i class="pi pi-plus" />
  <i class="pi pi-cog" />
  <i class="pi pi-check text-green-500" />

  <!-- In PrimeVue components -->
  <button icon="pi pi-save" label="Save" />
  <button icon="pi pi-times" severity="danger" />
</template>
```

[Browse all PrimeIcons →](https://primevue.org/icons/#list)

### 2. Iconify Icons (Recommended)

```vue
<template>
  <!-- Primary icon set: Lucide -->
  <i class="icon-[lucide--download]" />
  <i class="icon-[lucide--settings]" />
  <i class="icon-[lucide--workflow]" class="text-2xl" />

  <!-- Other popular icon sets -->
  <i-mdi:folder-open />
  <!-- Material Design Icons -->
  <i-heroicons:document-text />
  <!-- Heroicons -->
  <i-tabler:brand-github />
  <!-- Tabler Icons -->
  <i-carbon:cloud-upload />
  <!-- Carbon Icons -->

  <!-- With styling -->
  <i class="icon-[lucide--save]" class="w-6 h-6 text-blue-500" />
</template>
```

[Browse 200,000+ icons →](https://icon-sets.iconify.design/)

### 3. Custom Icons

```vue
<template>
  <!-- Your custom SVG icons from packages/design-system/src/icons/ -->
  <i-comfy:workflow />
  <i-comfy:node-tree />
  <i-comfy:my-custom-icon class="text-xl" />

  <!-- In PrimeVue button -->
  <Button severity="secondary">
    <template #icon>
      <i-comfy:workflow />
    </template>
  </Button>
</template>
```

## Icon Usage Patterns

### In Buttons

```vue
<template>
  <!-- PrimeIcon in button (simple) -->
  <Button icon="pi pi-check" label="Confirm" />

  <!-- Iconify/Custom in button (template) -->
  <Button>
    <template #icon>
      <i class="icon-[lucide--save]" />
    </template>
    Save File
  </Button>
</template>
```

### Conditional Icons

```vue
<template>
  <i class="icon-[lucide--eye]" v-if="isVisible" />
  <i class="icon-[lucide--eye-off]" v-else />

  <!-- Or with ternary -->
  <component :is="isLocked ? 'i-lucide:lock' : 'i-lucide:lock-open'" />
</template>
```

### With Tooltips

```vue
<template>
  <i
    class="icon-[lucide--info]"
    v-tooltip="'Click for more information'"
    class="cursor-pointer"
  />
</template>
```

## Using Iconify Icons

### Finding Icons

1. Visit [Iconify Icon Sets](https://icon-sets.iconify.design/)
2. Search or browse collections
3. Click on any icon to get its name
4. Use with `i-[collection]:[icon-name]` format

### Popular Collections

- **Lucide** (`i-lucide:`) - Our primary icon set, clean and consistent
- **Material Design Icons** (`i-mdi:`) - Comprehensive Material Design icons
- **Heroicons** (`i-heroicons:`) - Beautiful hand-crafted SVG icons
- **Tabler** (`i-tabler:`) - 3000+ free SVG icons
- **Carbon** (`i-carbon:`) - IBM's design system icons

## Adding Custom Icons

### 1. Add Your SVG

Place your SVG file in `packages/design-system/src/icons/`:

```
packages/design-system/src/icons/
├── workflow-duplicate.svg
├── node-preview.svg
└── your-icon.svg
```

### 2. SVG Format Requirements

```xml
<svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
  <!-- Use currentColor for theme compatibility -->
  <path fill="currentColor" d="..." />
</svg>
```

**Important:**

- Use `viewBox` for proper scaling (24x24 is standard)
- Don't include `width` or `height` attributes
- Use `currentColor` for theme-aware icons
- Keep SVGs optimized and simple

### 3. Use Immediately

```vue
<template>
  <i-comfy:your-icon />
</template>
```

No imports needed - icons are auto-discovered!

## Icon Guidelines

### Naming Conventions

- **Files**: `kebab-case.svg` (workflow-icon.svg)
- **Usage**: `<i-comfy:workflow-icon />`

### Size & Styling

```vue
<template>
  <!-- Size with Tailwind classes -->
  <i class="icon-[lucide--plus]" class="w-4 h-4" />
  <!-- 16px -->
  <i class="icon-[lucide--plus]" class="w-6 h-6" />
  <!-- 24px (default) -->
  <i class="icon-[lucide--plus]" class="w-8 h-8" />
  <!-- 32px -->

  <!-- Or text size -->
  <i class="icon-[lucide--plus]" class="text-sm" />
  <i class="icon-[lucide--plus]" class="text-2xl" />

  <!-- Colors -->
  <i class="icon-[lucide--check]" class="text-green-500" />
  <i class="icon-[lucide--x]" class="text-red-500" />
</template>
```

### Theme Compatibility

Always use `currentColor` in SVGs for automatic theme adaptation:

```xml
<!-- ✅ Good: Adapts to light/dark theme -->
<svg viewBox="0 0 24 24">
  <path fill="currentColor" d="..." />
</svg>

<!-- ❌ Bad: Fixed colors -->
<svg viewBox="0 0 24 24">
  <path fill="#000000" d="..." />
</svg>
```

## Migration Guide

### From PrimeIcons to Iconify/Custom

```vue
<template>
  <!-- Before -->
  <Button icon="pi pi-download" />

  <!-- After -->
  <Button>
    <template #icon>
      <i class="icon-[lucide--download]" />
    </template>
  </Button>
</template>
```

### From Inline SVG to Custom Icon

```vue
<template>
  <!-- Before: Inline SVG -->
  <svg class="w-6 h-6" viewBox="0 0 24 24">
    <path d="..." />
  </svg>

  <!-- After: Save as custom/my-icon.svg and use -->
  <i-comfy:my-icon class="w-6 h-6" />
</template>
```

## Technical Details

### Auto-Import System

Icons are automatically imported using `unplugin-icons` - no manual imports needed! Just use the icon component directly.

### Configuration

The icon system has two layers:

1. **Tailwind CSS Plugin** (`@iconify/tailwind4`):
   - Configured via `@plugin` directive in `packages/design-system/src/css/style.css`
   - Uses `from-folder(comfy, ...)` to load SVGs from `packages/design-system/src/icons/`
   - Auto-cleans and optimizes SVGs at build time

2. **Vite Runtime** (`vite.config.mts`):
   - Enables direct SVG import as Vue components
   - Supports dynamic icon loading

```css
/* CSS configuration for Tailwind icon classes */
@plugin "@iconify/tailwind4" {
  prefix: 'icon';
  scale: 1.2;
  icon-sets: from-folder(comfy, './packages/design-system/src/icons');
}
```

```typescript
// Vite configuration for component-based usage
Icons({
  compiler: 'vue3',
  customCollections: {
    comfy: FileSystemIconLoader('packages/design-system/src/icons')
  }
})
```

### TypeScript Support

Icons are fully typed. If TypeScript doesn't recognize a new custom icon:

1. Restart the dev server
2. Ensure the SVG file is valid
3. Check filename follows kebab-case

## Troubleshooting

### Icon Not Showing

1. **Check filename**: Must be kebab-case without special characters
2. **Restart dev server**: Required after adding new icons
3. **Verify SVG**: Ensure it's valid SVG syntax (build script validates automatically)
4. **Check console**: Look for Vue component resolution errors
5. **Build script errors**: Check console during build - malformed SVGs are logged but don't break builds

### Icon Wrong Color

- Replace hardcoded colors with `currentColor`
- Use `stroke="currentColor"` for outlines
- Use `fill="currentColor"` for filled shapes

### Icon Wrong Size

- Remove `width` and `height` from SVG
- Ensure `viewBox` is present
- Use CSS classes for sizing: `class="w-6 h-6"`

## Best Practices

1. **Optimize SVGs**: Use tools like [SVGO](https://jakearchibald.github.io/svgomg/) to minimize file size
2. **Consistent viewBox**: Stick to 24x24 or 16x16 for consistency
3. **Semantic names**: Use descriptive names like `workflow-duplicate` not `icon1`
4. **Theme support**: Always use `currentColor` for adaptable icons
5. **Test both themes**: Verify icons look good in light and dark modes

## Adding Icon Collections

To add an entire icon set from npm:

1. Install the icon package
2. Configure in `vite.config.mts`
3. Use with the appropriate prefix

See the [unplugin-icons documentation](https://github.com/unplugin/unplugin-icons) for details.

## Resources

- [PrimeIcons List](https://primevue.org/icons/#list)
- [Iconify Icon Browser](https://icon-sets.iconify.design/)
- [Lucide Icons](https://lucide.dev/icons/)
- [unplugin-icons docs](https://github.com/unplugin/unplugin-icons)
