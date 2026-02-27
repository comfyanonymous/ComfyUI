# Stores

This directory contains Pinia stores for the ComfyUI frontend application. Stores provide centralized state management for the application.

## Table of Contents

- [Overview](#overview)
- [Store Architecture](#store-architecture)
- [Core Stores](#core-stores)
- [Store Development Guidelines](#store-development-guidelines)
- [Common Patterns](#common-patterns)
- [Testing Stores](#testing-stores)

## Overview

Stores in ComfyUI use [Pinia](https://pinia.vuejs.org/), Vue's official state management library. Each store is responsible for managing a specific domain of the application state, such as user data, workflow information, graph state, and UI configuration.

Stores provide a way to maintain global application state that can be accessed from any component, regardless of where those components are in the component hierarchy. This solves the problem of "prop drilling" (passing data down through multiple levels of components) and allows components that aren't directly related to share and modify the same state.

For example, without global state:

```
                  App
                   │
        ┌──────────┴──────────┐
        │                     │
    HeaderBar               Canvas
        │                     │
        │                     │
    UserMenu            NodeProperties
```

In this structure, if the `UserMenu` component needs to update something that affects `NodeProperties`, the data would need to be passed up to `App` and then down again, through all intermediate components.

With Pinia stores, components can directly access and update the shared state:

```
    ┌─────────────────┐
    │                 │
    │  Pinia Stores   │
    │                 │
    └───────┬─────────┘
            │
            │ Accessed by
            ▼
┌──────────────────────────┐
│                          │
│       Components         │
│                          │
└──────────────────────────┘
```

## Store Architecture

The store architecture in ComfyUI follows these principles:

1. **Domain-driven**: Each store focuses on a specific domain of the application
2. **Single source of truth**: Stores serve as the definitive source for specific data
3. **Composition**: Stores can interact with other stores when needed
4. **Actions for logic**: Business logic is encapsulated in store actions
5. **Getters for derived state**: Computed values are exposed via getters

The following diagram illustrates the store architecture and data flow:

```
┌─────────────────────────────────────────────────────────┐
│                    Vue Components                        │
│                                                         │
│   ┌───────────────┐            ┌───────────────┐        │
│   │  Component A  │            │  Component B  │        │
│   └───────┬───────┘            └───────┬───────┘        │
│           │                            │                │
└───────────┼────────────────────────────┼────────────────┘
            │                            │
            │     ┌───────────────┐      │
            └────►│  Composables  │◄─────┘
                  └───────┬───────┘
                          │
┌─────────────────────────┼─────────────────────────────┐
│         Pinia Stores    │                             │
│                         │                             │
│     ┌───────────────────▼───────────────────────┐     │
│     │                 Actions                    │     │
│     └───────────────────┬───────────────────────┘     │
│                         │                             │
│     ┌───────────────────▼───────────────────────┐     │
│     │                  State                     │     │
│     └───────────────────┬───────────────────────┘     │
│                         │                             │
│     ┌───────────────────▼───────────────────────┐     │
│     │                 Getters                    │     │
│     └───────────────────┬───────────────────────┘     │
│                         │                             │
└─────────────────────────┼─────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   External Services                      │
│       (API, localStorage, WebSocket, etc.)              │
└─────────────────────────────────────────────────────────┘
```

## Core Stores

The following table lists ALL 46 store instances in the system as of 2026-01-29:

### Main Stores

| File                         | Store                         | Description                                             | Category   |
| ---------------------------- | ----------------------------- | ------------------------------------------------------- | ---------- |
| aboutPanelStore.ts           | useAboutPanelStore            | Manages the About panel state and badges                | UI         |
| apiKeyAuthStore.ts           | useApiKeyAuthStore            | Handles API key authentication                          | Auth       |
| comfyManagerStore.ts         | useComfyManagerStore          | Manages ComfyUI application state                       | Core       |
| comfyRegistryStore.ts        | useComfyRegistryStore         | Handles extensions registry                             | Registry   |
| commandStore.ts              | useCommandStore               | Manages commands and command execution                  | Core       |
| dialogStore.ts               | useDialogStore                | Controls dialog/modal display and state                 | UI         |
| domWidgetStore.ts            | useDomWidgetStore             | Manages DOM widget state                                | Widgets    |
| electronDownloadStore.ts     | useElectronDownloadStore      | Handles Electron-specific download operations           | Platform   |
| executionStore.ts            | useExecutionStore             | Tracks workflow execution state                         | Execution  |
| extensionStore.ts            | useExtensionStore             | Manages extension registration and state                | Extensions |
| firebaseAuthStore.ts         | useFirebaseAuthStore          | Handles Firebase authentication                         | Auth       |
| graphStore.ts                | useTitleEditorStore           | Manages title editing for nodes and groups              | UI         |
| graphStore.ts                | useCanvasStore                | Manages the graph canvas state and interactions         | Core       |
| helpCenterStore.ts           | useHelpCenterStore            | Manages help center visibility and state                | UI         |
| imagePreviewStore.ts         | useNodeOutputStore            | Manages node outputs and execution results              | Media      |
| maintenanceTaskStore.ts      | useMaintenanceTaskStore       | Handles system maintenance tasks                        | System     |
| menuItemStore.ts             | useMenuItemStore              | Handles menu items and their state                      | UI         |
| modelStore.ts                | useModelStore                 | Manages AI models information                           | Models     |
| modelToNodeStore.ts          | useModelToNodeStore           | Maps models to compatible nodes                         | Models     |
| nodeBookmarkStore.ts         | useNodeBookmarkStore          | Manages node bookmarks and favorites                    | Nodes      |
| nodeDefStore.ts              | useNodeDefStore               | Manages node definitions and schemas                    | Nodes      |
| nodeDefStore.ts              | useNodeFrequencyStore         | Tracks node usage frequency                             | Nodes      |
| queueStore.ts                | useQueueStore                 | Manages execution queue and task history                | Execution  |
| queueStore.ts                | useQueuePendingTaskCountStore | Tracks pending task counts                              | Execution  |
| queueStore.ts                | useQueueSettingsStore         | Manages queue execution settings                        | Execution  |
| releaseStore.ts              | useReleaseStore               | Manages application release information                 | System     |
| serverConfigStore.ts         | useServerConfigStore          | Handles server configuration                            | Config     |
| settingStore.ts              | useSettingStore               | Manages application settings                            | Config     |
| subgraphNavigationStore.ts   | useSubgraphNavigationStore    | Handles subgraph navigation state                       | Navigation |
| systemStatsStore.ts          | useSystemStatsStore           | Tracks system performance statistics                    | System     |
| toastStore.ts                | useToastStore                 | Manages toast notifications                             | UI         |
| userFileStore.ts             | useUserFileStore              | Manages user file operations                            | Files      |
| userStore.ts                 | useUserStore                  | Manages user data and preferences                       | User       |
| versionCompatibilityStore.ts | useVersionCompatibilityStore  | Manages frontend/backend version compatibility warnings | Core       |
| widgetStore.ts               | useWidgetStore                | Manages widget configurations                           | Widgets    |
| workflowStore.ts             | useWorkflowStore              | Handles workflow data and operations                    | Workflows  |
| workflowStore.ts             | useWorkflowBookmarkStore      | Manages workflow bookmarks and favorites                | Workflows  |
| workflowTemplatesStore.ts    | useWorkflowTemplatesStore     | Manages workflow templates                              | Workflows  |
| workspaceStore.ts            | useWorkspaceStore             | Manages overall workspace state                         | Workspace  |

### Workspace Stores

Located in `stores/workspace/`:

| File                 | Store                | Description                                 | Category |
| -------------------- | -------------------- | ------------------------------------------- | -------- |
| bottomPanelStore.ts  | useBottomPanelStore  | Controls bottom panel visibility and state  | UI       |
| colorPaletteStore.ts | useColorPaletteStore | Manages color palette configurations        | UI       |
| nodeHelpStore.ts     | useNodeHelpStore     | Handles node help and documentation display | UI       |
| searchBoxStore.ts    | useSearchBoxStore    | Manages search box functionality            | UI       |
| sidebarTabStore.ts   | useSidebarTabStore   | Controls sidebar tab states and navigation  | UI       |

## Store Development Guidelines

When developing or modifying stores, follow these best practices:

1. **Define clear purpose**: Each store should have a specific responsibility
2. **Use actions for async operations**: Encapsulate asynchronous logic in actions
3. **Keep stores focused**: Each store should manage related state
4. **Document public API**: Add comments for state properties, actions, and getters
5. **Use getters for derived state**: Compute derived values using getters
6. **Test store functionality**: Write unit tests for stores

### Store Template

Here's a template for creating a new Pinia store, following the setup style used in ComfyUI:

```typescript
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

export const useExampleStore = defineStore('example', () => {
  // State
  const items = ref([])
  const isLoading = ref(false)
  const error = ref(null)

  // Getters
  const itemCount = computed(() => items.value.length)
  const hasError = computed(() => error.value !== null)

  // Actions
  function addItem(item) {
    items.value.push(item)
  }

  async function fetchItems() {
    isLoading.value = true
    error.value = null

    try {
      const response = await fetch('/api/items')
      const data = await response.json()
      items.value = data
    } catch (err) {
      error.value = err.message
    } finally {
      isLoading.value = false
    }
  }

  // Expose state, getters, and actions
  return {
    // State
    items,
    isLoading,
    error,

    // Getters
    itemCount,
    hasError,

    // Actions
    addItem,
    fetchItems
  }
})
```

## Common Patterns

Stores in ComfyUI frequently use these patterns:

### API Integration

```typescript
import { defineStore } from 'pinia'
import { ref } from 'vue'
import { api } from '@/scripts/api'

export const useDataStore = defineStore('data', () => {
  const data = ref([])
  const loading = ref(false)
  const error = ref(null)

  async function fetchData() {
    loading.value = true
    try {
      const result = await api.getExtensions()
      data.value = result
    } catch (err) {
      error.value = err.message
    } finally {
      loading.value = false
    }
  }

  return {
    data,
    loading,
    error,
    fetchData
  }
})
```

### Store Composition

```typescript
import { defineStore, storeToRefs } from 'pinia'
import { computed, ref, watch } from 'vue'
import { useOtherStore } from './otherStore'

export const useComposedStore = defineStore('composed', () => {
  const otherStore = useOtherStore()
  const { someData } = storeToRefs(otherStore)

  // Local state
  const localState = ref(0)

  // Computed value based on other store
  const derivedValue = computed(() => {
    return computeFromOtherData(someData.value, localState.value)
  })

  // Action that uses another store
  async function complexAction() {
    await otherStore.someAction()
    localState.value += 1
  }

  return {
    localState,
    derivedValue,
    complexAction
  }
})
```

### Persistent State

```typescript
import { defineStore } from 'pinia'
import { ref, watch } from 'vue'

export const usePreferencesStore = defineStore('preferences', () => {
  // Load from localStorage if available
  const theme = ref(localStorage.getItem('theme') || 'light')
  const fontSize = ref(parseInt(localStorage.getItem('fontSize') || '14'))

  // Save to localStorage when changed
  watch(theme, (newTheme) => {
    localStorage.setItem('theme', newTheme)
  })

  watch(fontSize, (newSize) => {
    localStorage.setItem('fontSize', newSize.toString())
  })

  function setTheme(newTheme) {
    theme.value = newTheme
  }

  return {
    theme,
    fontSize,
    setTheme
  }
})
```

## Testing Stores

Stores should be tested to ensure they behave as expected. Here's an example of how to test a store:

```typescript
import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

import { api } from '@/scripts/api'
import { useExampleStore } from '@/stores/exampleStore'

// Mock API dependencies
vi.mock('@/scripts/api', () => ({
  api: {
    getData: vi.fn()
  }
}))

describe('useExampleStore', () => {
  let store: ReturnType<typeof useExampleStore>

  beforeEach(() => {
    // Create a fresh pinia instance and make it active
    setActivePinia(createPinia())
    store = useExampleStore()

    // Clear all mocks
    vi.clearAllMocks()
  })

  it('should initialize with default state', () => {
    expect(store.items).toEqual([])
    expect(store.isLoading).toBe(false)
    expect(store.error).toBeNull()
  })

  it('should add an item', () => {
    store.addItem('test')
    expect(store.items).toEqual(['test'])
    expect(store.itemCount).toBe(1)
  })

  it('should fetch items', async () => {
    // Setup mock response
    vi.mocked(api.getData).mockResolvedValue(['item1', 'item2'])

    // Call the action
    await store.fetchItems()

    // Verify state changes
    expect(store.isLoading).toBe(false)
    expect(store.items).toEqual(['item1', 'item2'])
    expect(store.error).toBeNull()
  })
})
```

For more information on Pinia, refer to the [Pinia documentation](https://pinia.vuejs.org/introduction.html).
