# Pinia Store Testing Guide

This guide covers patterns and examples for testing Pinia stores in the ComfyUI Frontend codebase.

## Table of Contents

1. [Setting Up Store Tests](#setting-up-store-tests)
2. [Testing Store State](#testing-store-state)
3. [Testing Store Actions](#testing-store-actions)
4. [Testing Store Getters](#testing-store-getters)
5. [Mocking Dependencies in Stores](#mocking-dependencies-in-stores)
6. [Testing Store Watchers](#testing-store-watchers)
7. [Testing Store Integration](#testing-store-integration)

## Setting Up Store Tests

Basic setup for testing Pinia stores:

```typescript
// Example from: tests-ui/tests/store/workflowStore.test.ts
import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useWorkflowStore } from '@/domains/workflow/ui/stores/workflowStore'

describe('useWorkflowStore', () => {
  let store: ReturnType<typeof useWorkflowStore>

  beforeEach(() => {
    // Create a fresh testing pinia and activate it for each test
    setActivePinia(createTestingPinia({ stubActions: false }))

    // Initialize the store
    store = useWorkflowStore()

    // Clear any mocks
    vi.clearAllMocks()
  })

  it('should initialize with default state', () => {
    expect(store.workflows).toEqual([])
    expect(store.activeWorkflow).toBeUndefined()
    expect(store.openWorkflows).toEqual([])
  })
})
```

## Testing Store State

Testing store state changes:

```typescript
// Example from: tests-ui/tests/store/workflowStore.test.ts
it('should create a temporary workflow with a unique path', () => {
  const workflow = store.createTemporary()
  expect(workflow.path).toBe('workflows/Unsaved Workflow.json')

  const workflow2 = store.createTemporary()
  expect(workflow2.path).toBe('workflows/Unsaved Workflow (2).json')
})

it('should create a temporary workflow not clashing with persisted workflows', async () => {
  await syncRemoteWorkflows(['a.json'])
  const workflow = store.createTemporary('a.json')
  expect(workflow.path).toBe('workflows/a (2).json')
})
```

## Testing Store Actions

Testing store actions:

```typescript
// Example from: tests-ui/tests/store/workflowStore.test.ts
describe('openWorkflow', () => {
  it('should load and open a temporary workflow', async () => {
    // Create a test workflow
    const workflow = store.createTemporary('test.json')
    const mockWorkflowData = { nodes: [], links: [] }

    // Mock the load response
    vi.spyOn(workflow, 'load').mockImplementation(async () => {
      workflow.changeTracker = { activeState: mockWorkflowData } as any
      return workflow as LoadedComfyWorkflow
    })

    // Open the workflow
    await store.openWorkflow(workflow)

    // Verify the workflow is now active
    expect(store.activeWorkflow?.path).toBe(workflow.path)

    // Verify the workflow is in the open workflows list
    expect(store.isOpen(workflow)).toBe(true)
  })

  it('should not reload an already active workflow', async () => {
    const workflow = await store.createTemporary('test.json').load()
    vi.spyOn(workflow, 'load')

    // Set as active workflow
    store.activeWorkflow = workflow

    await store.openWorkflow(workflow)

    // Verify load was not called
    expect(workflow.load).not.toHaveBeenCalled()
  })
})
```

## Testing Store Getters

Testing store getters:

```typescript
// Example from: tests-ui/tests/store/modelStore.test.ts
describe('getters', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    store = useModelStore()

    // Set up test data
    store.models = {
      checkpoints: [
        {
          name: 'model1.safetensors',
          path: 'models/checkpoints/model1.safetensors'
        },
        { name: 'model2.ckpt', path: 'models/checkpoints/model2.ckpt' }
      ],
      loras: [
        { name: 'lora1.safetensors', path: 'models/loras/lora1.safetensors' }
      ]
    }

    // Mock API
    vi.mocked(api.getModelInfo).mockImplementation(async (modelName) => {
      if (modelName.includes('model1')) {
        return { info: { resolution: 768 } }
      }
      return { info: { resolution: 512 } }
    })
  })

  it('should return models grouped by type', () => {
    expect(store.modelsByType.checkpoints.length).toBe(2)
    expect(store.modelsByType.loras.length).toBe(1)
  })

  it('should filter models by name', () => {
    store.searchTerm = 'model1'
    expect(store.filteredModels.checkpoints.length).toBe(1)
    expect(store.filteredModels.checkpoints[0].name).toBe('model1.safetensors')
  })
})
```

## Mocking Dependencies in Stores

Mocking API and other dependencies:

```typescript
// Example from: tests-ui/tests/store/workflowStore.test.ts
// Add mock for api at the top of the file
vi.mock('@/scripts/api', () => ({
  api: {
    getUserData: vi.fn(),
    storeUserData: vi.fn(),
    listUserDataFullInfo: vi.fn(),
    apiURL: vi.fn(),
    addEventListener: vi.fn()
  }
}))

// Mock comfyApp globally for the store setup
vi.mock('@/scripts/app', () => ({
  app: {
    canvas: null // Start with canvas potentially undefined or null
  }
}))

describe('syncWorkflows', () => {
  const syncRemoteWorkflows = async (filenames: string[]) => {
    vi.mocked(api.listUserDataFullInfo).mockResolvedValue(
      filenames.map((filename) => ({
        path: filename,
        modified: new Date().getTime(),
        size: 1 // size !== -1 for remote workflows
      }))
    )
    return await store.syncWorkflows()
  }

  it('should sync workflows', async () => {
    await syncRemoteWorkflows(['a.json', 'b.json'])
    expect(store.workflows.length).toBe(2)
  })
})
```

## Testing Store Watchers

Testing store watchers and reactive behavior:

```typescript
// Example from: tests-ui/tests/store/workflowStore.test.ts
import { nextTick } from 'vue'

describe('Subgraphs', () => {
  it('should update automatically when activeWorkflow changes', async () => {
    // Arrange: Set initial canvas state
    const initialSubgraph = {
      name: 'Initial Subgraph',
      pathToRootGraph: [{ name: 'Root' }, { name: 'Initial Subgraph' }],
      isRootGraph: false
    }
    vi.mocked(comfyApp.canvas).subgraph = initialSubgraph as any

    // Trigger initial update
    store.updateActiveGraph()
    await nextTick()

    // Verify initial state
    expect(store.isSubgraphActive).toBe(true)
    expect(store.subgraphNamePath).toEqual(['Initial Subgraph'])

    // Act: Change the active workflow and canvas state
    const workflow2 = store.createTemporary('workflow2.json')
    vi.spyOn(workflow2, 'load').mockImplementation(async () => {
      workflow2.changeTracker = { activeState: {} } as any
      workflow2.originalContent = '{}'
      workflow2.content = '{}'
      return workflow2 as LoadedComfyWorkflow
    })

    // Change canvas state
    vi.mocked(comfyApp.canvas).subgraph = undefined

    await store.openWorkflow(workflow2)
    await nextTick() // Allow watcher to trigger

    // Assert: Check state was updated by the watcher
    expect(store.isSubgraphActive).toBe(false)
    expect(store.subgraphNamePath).toEqual([])
  })
})
```

## Testing Store Integration

Testing store integration with other parts of the application:

```typescript
// Example from: tests-ui/tests/store/workflowStore.test.ts
describe('renameWorkflow', () => {
  it('should rename workflow and update bookmarks', async () => {
    const workflow = store.createTemporary('dir/test.json')
    const bookmarkStore = useWorkflowBookmarkStore()

    // Set up initial bookmark
    expect(workflow.path).toBe('workflows/dir/test.json')
    await bookmarkStore.setBookmarked(workflow.path, true)
    expect(bookmarkStore.isBookmarked(workflow.path)).toBe(true)

    // Mock super.rename
    vi.spyOn(Object.getPrototypeOf(workflow), 'rename').mockImplementation(
      async function (this: any, newPath: string) {
        this.path = newPath
        return this
      } as any
    )

    // Perform rename
    const newPath = 'workflows/dir/renamed.json'
    await store.renameWorkflow(workflow, newPath)

    // Check that bookmark was transferred
    expect(bookmarkStore.isBookmarked(newPath)).toBe(true)
    expect(bookmarkStore.isBookmarked('workflows/dir/test.json')).toBe(false)
  })
})
```
