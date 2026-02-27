# @comfyorg/tailwind-utils

Shared Tailwind CSS utility functions for the ComfyUI Frontend monorepo.

## Usage

The `cn` function combines `clsx` and `tailwind-merge` to handle conditional classes and resolve Tailwind conflicts.

```typescript
import { cn } from '@comfyorg/tailwind-utils'

// Use with conditional classes (object)
<div :class="cn('transition-opacity', { 'opacity-75': !isHovered })" />

// Use with conditional classes (ternary)
<button
  :class="cn('px-4 py-2', isActive ? 'bg-blue-500' : 'bg-smoke-500')"
/>
```

## Installation

This package is part of the ComfyUI Frontend monorepo and is automatically available to all workspace packages.

```json
{
  "dependencies": {
    "@comfyorg/tailwind-utils": "workspace:*"
  }
}
```
