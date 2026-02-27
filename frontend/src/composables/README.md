# Composables

This directory contains Vue composables for the ComfyUI frontend application. Composables are reusable pieces of logic that encapsulate stateful functionality and can be shared across components.

## Table of Contents

- [Composables](#composables)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Composable Architecture](#composable-architecture)
  - [Composable Categories](#composable-categories)
    - [Auth](#auth)
    - [Bottom Panel Tabs](#bottom-panel-tabs)
    - [Element](#element)
    - [Functional](#functional)
    - [Manager](#manager)
      - [Node Pack](#node-pack)
    - [Node](#node)
    - [Settings](#settings)
    - [Sidebar Tabs](#sidebar-tabs)
    - [Tree](#tree)
    - [Widgets](#widgets)
    - [Root-level Composables](#root-level-composables)
  - [Usage Guidelines](#usage-guidelines)
  - [VueUse Library](#vueuse-library)
  - [Development Guidelines](#development-guidelines)
    - [Composable Template](#composable-template)
  - [Common Patterns](#common-patterns)
    - [State Management](#state-management)
    - [Event Handling with VueUse](#event-handling-with-vueuse)
    - [Fetch \& Load with VueUse](#fetch--load-with-vueuse)

## Overview

Vue composables are a core part of Vue 3's Composition API and provide a way to extract and reuse stateful logic between multiple components. In ComfyUI, composables are used to encapsulate behaviors like:

- State management
- DOM interactions
- Feature-specific functionality
- UI behaviors
- Data fetching

Composables enable a more modular and functional approach to building components, allowing for better code reuse and separation of concerns. They help keep your component code cleaner by extracting complex logic into separate, reusable functions.

As described in the [Vue.js documentation](https://vuejs.org/guide/reusability/composables.html), composables are:

> Functions that leverage Vue's Composition API to encapsulate and reuse stateful logic.

## Composable Architecture

The composable architecture in ComfyUI follows these principles:

1. **Single Responsibility**: Each composable should focus on a specific concern
2. **Composition**: Composables can use other composables
3. **Reactivity**: Composables leverage Vue's reactivity system
4. **Reusability**: Composables are designed to be used across multiple components

The following diagram shows how composables fit into the application architecture:

```
┌─────────────────────────────────────────────────────────┐
│                    Vue Components                        │
│                                                         │
│     ┌─────────────┐     ┌─────────────┐                 │
│     │ Component A │     │ Component B │                 │
│     └──────┬──────┘     └──────┬──────┘                 │
│            │                   │                        │
└────────────┼───────────────────┼────────────────────────┘
             │                   │
             ▼                   ▼
┌────────────┴───────────────────┴────────────────────────┐
│                    Composables                           │
│                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │
│  │ useFeatureA │  │ useFeatureB │  │ useFeatureC │      │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘      │
│         │                │                │             │
└─────────┼────────────────┼────────────────┼─────────────┘
          │                │                │
          ▼                ▼                ▼
┌─────────┴────────────────┴────────────────┴─────────────┐
│                    Services & Stores                     │
└─────────────────────────────────────────────────────────┘
```

## Composable Categories

The following tables list ALL composables in the system as of 2026-01-30:

### Auth

Composables for authentication and user management:

| Composable               | Description                                     |
| ------------------------ | ----------------------------------------------- |
| `useCurrentUser`         | Provides access to the current user information |
| `useFirebaseAuthActions` | Handles Firebase authentication operations      |

### Bottom Panel Tabs

Composables for terminal and bottom panel functionality:

| Composable          | Description                              |
| ------------------- | ---------------------------------------- |
| `useTerminal`       | Core terminal functionality              |
| `useTerminalBuffer` | Manages terminal output buffer           |
| `useTerminalTabs`   | Handles multiple terminal tab management |

### Element

Composables for DOM and element interactions:

| Composable                    | Description                                 |
| ----------------------------- | ------------------------------------------- |
| `useAbsolutePosition`         | Handles element positioning                 |
| `useCanvasPositionConversion` | Converts between canvas and DOM coordinates |
| `useDomClipping`              | Manages clipping of DOM elements            |
| `useResponsiveCollapse`       | Manages responsive collapsing of elements   |

### Functional

Utility composables for common patterns:

| Composable         | Description                        |
| ------------------ | ---------------------------------- |
| `useChainCallback` | Chains multiple callbacks together |

### Manager

Composables for ComfyUI Manager integration (located in
`@/workbench/extensions/manager/composables`):

| Composable                   | Description                                 |
| ---------------------------- | ------------------------------------------- |
| `useManagerStatePersistence` | Persists manager UI state                   |
| `useManagerState`            | Determines availability of manager UI modes |
| `useManagerQueue`            | Handles manager task queue state            |
| `useConflictAcknowledgment`  | Tracks conflict dismissal state             |
| `useConflictDetection`       | Orchestrates conflict detection workflow    |
| `useImportFailedDetection`   | Handles import-failed conflict dialogs      |
| `useRegistrySearch`          | Manages registry search UI state            |

#### Node Pack

Node package helpers live under
`@/workbench/extensions/manager/composables/nodePack`:

| Composable                | Description                               |
| ------------------------- | ----------------------------------------- |
| `useInstalledPacks`       | Manages installed node packages           |
| `useMissingNodes`         | Detects and handles missing nodes         |
| `useNodePacks`            | Core node package functionality           |
| `usePackUpdateStatus`     | Tracks package update availability        |
| `usePacksSelection`       | Provides selection helpers for pack lists |
| `usePacksStatus`          | Aggregates status across multiple packs   |
| `useUpdateAvailableNodes` | Detects available updates for nodes       |
| `useWorkflowPacks`        | Manages packages used in workflows        |

### Node

Composables for node-specific functionality:

| Composable                  | Description                                |
| --------------------------- | ------------------------------------------ |
| `useNodeAnimatedImage`      | Handles animated images in nodes           |
| `useNodeBadge`              | Handles node badge display and interaction |
| `useNodeCanvasImagePreview` | Canvas-based image preview for nodes       |
| `useNodeDragAndDrop`        | Handles drag and drop for nodes            |
| `useNodeFileInput`          | Manages file input widgets in nodes        |
| `useNodeImage`              | Manages node image preview                 |
| `useNodeImageUpload`        | Handles image upload for nodes             |
| `useNodePaste`              | Manages paste operations for nodes         |
| `useNodePricing`            | Handles pricing display for nodes          |
| `useNodeProgressText`       | Displays progress text in nodes            |
| `useWatchWidget`            | Watches widget value changes               |

### Settings

Composables for settings management:

| Composable         | Description                                |
| ------------------ | ------------------------------------------ |
| `useSettingSearch` | Provides search functionality for settings |
| `useSettingUI`     | Manages settings UI interactions           |

### Sidebar Tabs

Composables for sidebar functionality:

| Composable                  | Description                           |
| --------------------------- | ------------------------------------- |
| `useModelLibrarySidebarTab` | Manages the model library sidebar tab |
| `useNodeLibrarySidebarTab`  | Manages the node library sidebar tab  |
| `useWorkflowsSidebarTab`    | Manages the workflows sidebar tab     |

### Tree

Composables for tree structure operations:

| Composable                | Description                             |
| ------------------------- | --------------------------------------- |
| `useTreeFolderOperations` | Handles folder operations in tree views |

### Widgets

Composables for widget functionality:

| Composable              | Description                             |
| ----------------------- | --------------------------------------- |
| `useBooleanWidget`      | Manages boolean widget interactions     |
| `useComboWidget`        | Manages combo box widget interactions   |
| `useFloatWidget`        | Manages float input widget interactions |
| `useImagePreviewWidget` | Manages image preview widget            |
| `useImageUploadWidget`  | Handles image upload widget             |
| `useIntWidget`          | Manages integer input widget            |
| `useMarkdownWidget`     | Handles markdown display widget         |
| `useProgressTextWidget` | Manages progress text widget            |
| `useRemoteWidget`       | Handles remote widget connections       |
| `useStringWidget`       | Manages string input widget             |

### Root-level Composables

General-purpose composables:

| Composable                  | Description                                               |
| --------------------------- | --------------------------------------------------------- |
| `useBrowserTabTitle`        | Manages browser tab title updates                         |
| `useCachedRequest`          | Provides request caching functionality                    |
| `useCanvasDrop`             | Handles drop operations on canvas                         |
| `useCivitaiModel`           | Integrates with Civitai model API                         |
| `useContextMenuTranslation` | Handles context menu translations                         |
| `useCopy`                   | Provides copy functionality                               |
| `useCopyToClipboard`        | Manages clipboard operations                              |
| `useCoreCommands`           | Provides core command functionality                       |
| `useDownload`               | Handles file download operations                          |
| `useErrorHandling`          | Centralized error handling                                |
| `useGlobalLitegraph`        | Access to global LiteGraph instance                       |
| `useLitegraphSettings`      | Manages LiteGraph configuration                           |
| `useManagerQueue`           | Handles manager queue operations                          |
| `usePaste`                  | Provides paste functionality                              |
| `usePragmaticDragAndDrop`   | Integrates Atlassian's drag-and-drop library              |
| `useProgressFavicon`        | Updates favicon with progress indicator                   |
| `useRefreshableSelection`   | Manages refreshable selections                            |
| `useRegistrySearch`         | Searches the ComfyUI registry                             |
| `useServerLogs`             | Manages server log display                                |
| `useTemplateWorkflows`      | Manages template workflow loading, selection, and display |
| `useTreeExpansion`          | Handles tree node expansion state                         |

| `useWorkflowAutoSave` | Handles automatic workflow saving |
| `useWorkflowPersistence` | Manages workflow persistence |
| `useWorkflowValidation` | Validates workflow integrity |

## Usage Guidelines

When using composables in components, follow these guidelines:

1. **Import and call** composables at the top level of the `setup` function
2. **Destructure returned values** to use in your component
3. **Respect reactivity** by not destructuring reactive objects
4. **Handle cleanup** by using `onUnmounted` when necessary
5. **Use VueUse** for common functionality instead of writing from scratch

Example usage:

```vue
<template>
  <div
    :class="{ dragging: isDragging }"
    @mousedown="startDrag"
    @mouseup="endDrag"
  >
    <img v-if="imageUrl" :src="imageUrl" alt="Node preview" />
  </div>
</template>

<script setup lang="ts">
import { useNodeDragAndDrop } from '@/composables/node/useNodeDragAndDrop'
import { useNodeImage } from '@/composables/node/useNodeImage'

// Use composables at the top level
const { isDragging, startDrag, endDrag } = useNodeDragAndDrop()
const { imageUrl, loadImage } = useNodeImage()

// Use returned values in your component
</script>
```

## VueUse Library

ComfyUI leverages the [VueUse](https://vueuse.org/) library, which provides a collection of essential Vue Composition API utilities. Instead of implementing common functionality from scratch, prefer using VueUse composables for:

- DOM event handling (`useEventListener`, `useMouseInElement`)
- Element measurements (`useElementBounding`, `useElementSize`)
- Asynchronous operations (`useAsyncState`, `useFetch`)
- Animation and timing (`useTransition`, `useTimeout`, `useInterval`)
- Browser APIs (`useLocalStorage`, `useClipboard`)
- Sensors (`useDeviceMotion`, `useDeviceOrientation`)
- State management (`createGlobalState`, `useStorage`)
- ...and [more](https://vueuse.org/functions.html)

Examples:

```js
// Instead of manually adding/removing event listeners
import { useEventListener } from '@vueuse/core'

useEventListener(window, 'resize', handleResize)

// Instead of manually tracking element measurements
import { useElementBounding } from '@vueuse/core'

const { width, height, top, left } = useElementBounding(elementRef)

// Instead of manual async state management
import { useAsyncState } from '@vueuse/core'

const { state, isReady, isLoading } = useAsyncState(
  fetch('https://api.example.com/data').then((r) => r.json()),
  { data: [] }
)
```

For a complete list of available functions, see the [VueUse documentation](https://vueuse.org/functions.html).

## Development Guidelines

When creating or modifying composables, follow these best practices:

1. **Name with `use` prefix**: All composables should start with "use"
2. **Return an object**: Composables should return an object with named properties/methods
3. **Handle cleanup**: Use `onUnmounted` to clean up resources
4. **Document parameters and return values**: Add JSDoc comments
5. **Test composables**: Write unit tests for composable functionality
6. **Use VueUse**: Leverage VueUse composables instead of reimplementing common functionality
7. **Implement proper cleanup**: Cancel debounced functions, pending requests, and clear maps
8. **Use watchDebounced/watchThrottled**: For performance-sensitive reactive operations

### Composable Template

Here's a template for creating a new composable:

```typescript
import { ref, computed, onMounted, onUnmounted } from 'vue'

/**
 * Composable for [functionality description]
 * @param options Configuration options
 * @returns Object containing state and methods
 */
export function useExample(options = {}) {
  // State
  const state = ref({
    // Initial state
  })

  // Computed values
  const derivedValue = computed(() => {
    // Compute from state
    return state.value.someProperty
  })

  // Methods
  function doSomething() {
    // Implementation
  }

  // Lifecycle hooks
  onMounted(() => {
    // Setup
  })

  onUnmounted(() => {
    // Cleanup
  })

  // Return exposed state and methods
  return {
    state,
    derivedValue,
    doSomething
  }
}
```

## Common Patterns

Composables in ComfyUI frequently use these patterns:

### State Management

```typescript
export function useState() {
  const count = ref(0)

  function increment() {
    count.value++
  }

  return {
    count,
    increment
  }
}
```

### Event Handling with VueUse

```typescript
import { useEventListener } from '@vueuse/core'

export function useKeyPress(key) {
  const isPressed = ref(false)

  useEventListener('keydown', (e) => {
    if (e.key === key) {
      isPressed.value = true
    }
  })

  useEventListener('keyup', (e) => {
    if (e.key === key) {
      isPressed.value = false
    }
  })

  return { isPressed }
}
```

### Fetch & Load with VueUse

```typescript
import { useAsyncState } from '@vueuse/core'

export function useFetchData(url) {
  const {
    state: data,
    isLoading,
    error,
    execute: refresh
  } = useAsyncState(
    async () => {
      const response = await fetch(url)
      if (!response.ok) throw new Error('Failed to fetch data')
      return response.json()
    },
    null,
    { immediate: true }
  )

  return { data, isLoading, error, refresh }
}
```

For more information on Vue composables, refer to the [Vue.js Composition API documentation](https://vuejs.org/guide/reusability/composables.html) and the [VueUse documentation](https://vueuse.org/).
