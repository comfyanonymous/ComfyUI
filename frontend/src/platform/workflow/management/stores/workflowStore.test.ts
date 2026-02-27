import { createTestingPinia } from '@pinia/testing'
import { setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { nextTick } from 'vue'

import type { LGraph, Subgraph } from '@/lib/litegraph/src/litegraph'
import type {
  ComfyWorkflow,
  LoadedComfyWorkflow
} from '@/platform/workflow/management/stores/workflowStore'
import {
  useWorkflowBookmarkStore,
  useWorkflowStore
} from '@/platform/workflow/management/stores/workflowStore'
import { useWorkflowDraftStore } from '@/platform/workflow/persistence/stores/workflowDraftStore'
import { api } from '@/scripts/api'
import { app as comfyApp } from '@/scripts/app'
import { defaultGraph, defaultGraphJSON } from '@/scripts/defaultGraph'
import { isSubgraph } from '@/utils/typeGuardUtil'
import {
  createMockCanvas,
  createMockChangeTracker
} from '@/utils/__tests__/litegraphTestUtils'

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
    canvas: {} // Start with empty canvas object
  }
}))

// Mock isSubgraph
vi.mock('@/utils/typeGuardUtil', () => ({
  isSubgraph: vi.fn(() => false)
}))

describe('useWorkflowStore', () => {
  let store: ReturnType<typeof useWorkflowStore>
  let bookmarkStore: ReturnType<typeof useWorkflowBookmarkStore>

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

  const syncRemoteWorkflowsWithMeta = async (
    files: Array<{ path: string; modified: number; size: number }>
  ) => {
    vi.mocked(api.listUserDataFullInfo).mockResolvedValue(files)
    return await store.syncWorkflows()
  }

  beforeEach(() => {
    setActivePinia(createTestingPinia({ stubActions: false }))
    store = useWorkflowStore()
    bookmarkStore = useWorkflowBookmarkStore()
    vi.clearAllMocks()

    // Add default mock implementations
    vi.mocked(api.getUserData).mockResolvedValue({
      status: 200,
      json: () => Promise.resolve({ favorites: [] })
    } as Response)
    vi.mocked(api.storeUserData).mockResolvedValue({
      status: 200
    } as Response)
  })

  describe('syncWorkflows', () => {
    it('should sync workflows', async () => {
      await syncRemoteWorkflows(['a.json', 'b.json'])
      expect(store.workflows.length).toBe(2)
    })

    it('should exclude temporary workflows', async () => {
      const workflow = store.createTemporary('c.json')
      await syncRemoteWorkflows(['a.json', 'b.json'])
      expect(store.workflows.length).toBe(3)
      expect(store.workflows.filter((w) => w.isTemporary)).toEqual([workflow])
    })

    it('should not unload the active workflow during sync', async () => {
      await syncRemoteWorkflowsWithMeta([
        { path: 'a.json', modified: 100, size: 1 }
      ])
      const workflow = store.getWorkflowByPath('workflows/a.json')!

      vi.mocked(api.getUserData).mockResolvedValue({
        status: 200,
        text: () => Promise.resolve(defaultGraphJSON)
      } as Response)

      await store.openWorkflow(workflow)
      expect(store.activeWorkflow?.path).toBe('workflows/a.json')
      expect(store.activeWorkflow?.isLoaded).toBe(true)

      await syncRemoteWorkflowsWithMeta([
        { path: 'a.json', modified: 200, size: 2 }
      ])

      expect(store.activeWorkflow?.isLoaded).toBe(true)
      expect(workflow.lastModified).toBe(200)
      expect(workflow.size).toBe(2)
    })

    it('should not unload a loaded workflow if metadata has not changed', async () => {
      await syncRemoteWorkflowsWithMeta([
        { path: 'a.json', modified: 100, size: 1 }
      ])
      const workflow = store.getWorkflowByPath('workflows/a.json')!

      vi.mocked(api.getUserData).mockResolvedValue({
        status: 200,
        text: () => Promise.resolve(defaultGraphJSON)
      } as Response)

      await workflow.load()
      expect(workflow.isLoaded).toBe(true)

      await syncRemoteWorkflowsWithMeta([
        { path: 'a.json', modified: 100, size: 1 }
      ])
      expect(workflow.isLoaded).toBe(true)
    })

    it('should unload a loaded workflow if metadata has changed and it is not active', async () => {
      await syncRemoteWorkflowsWithMeta([
        { path: 'a.json', modified: 100, size: 1 }
      ])
      const workflow = store.getWorkflowByPath('workflows/a.json')!

      vi.mocked(api.getUserData).mockResolvedValue({
        status: 200,
        text: () => Promise.resolve(defaultGraphJSON)
      } as Response)

      await workflow.load()
      expect(workflow.isLoaded).toBe(true)
      expect(workflow.content).toBe(defaultGraphJSON)
      expect(workflow.originalContent).toBe(defaultGraphJSON)

      // Metadata change should invalidate cached content for non-active workflows.
      await syncRemoteWorkflowsWithMeta([
        { path: 'a.json', modified: 200, size: 2 }
      ])

      expect(workflow.lastModified).toBe(200)
      expect(workflow.size).toBe(2)
      expect(workflow.isLoaded).toBe(false)
      expect(workflow.content).toBeNull()
      expect(workflow.originalContent).toBeNull()
    })
  })

  describe('createTemporary', () => {
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
  })

  describe('openWorkflow', () => {
    it('should load and open a temporary workflow', async () => {
      // Create a test workflow
      const workflow = store.createTemporary('test.json')

      // Mock the load response
      vi.spyOn(workflow, 'load').mockImplementation(async () => {
        workflow.changeTracker = createMockChangeTracker({
          workflow
        })
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

    it('should load a remote workflow', async () => {
      await syncRemoteWorkflows(['a.json'])
      const workflow = store.getWorkflowByPath('workflows/a.json')!
      expect(workflow).not.toBeNull()
      expect(workflow.path).toBe('workflows/a.json')
      expect(workflow.isLoaded).toBe(false)
      expect(workflow.isTemporary).toBe(false)
      vi.mocked(api.getUserData).mockResolvedValue({
        status: 200,
        text: () => Promise.resolve(defaultGraphJSON)
      } as Response)
      await workflow.load()

      expect(workflow.isLoaded).toBe(true)
      expect(workflow.content).toEqual(defaultGraphJSON)
      expect(workflow.originalContent).toEqual(defaultGraphJSON)
      expect(workflow.activeState).toEqual(defaultGraph)
      expect(workflow.initialState).toEqual(defaultGraph)
      expect(workflow.isModified).toBe(false)
    })

    it('should load and open a remote workflow', async () => {
      await syncRemoteWorkflows(['a.json', 'b.json'])

      const workflow = store.getWorkflowByPath('workflows/a.json')!
      expect(workflow).not.toBeNull()
      expect(workflow.path).toBe('workflows/a.json')
      expect(workflow.isLoaded).toBe(false)
      vi.mocked(api.getUserData).mockResolvedValue({
        status: 200,
        text: () => Promise.resolve(defaultGraphJSON)
      } as Response)

      const loadedWorkflow = await store.openWorkflow(workflow)

      expect(loadedWorkflow).toBe(workflow)
      expect(loadedWorkflow.path).toBe('workflows/a.json')
      expect(store.activeWorkflow?.path).toBe('workflows/a.json')
      expect(store.isOpen(loadedWorkflow)).toBe(true)
      expect(loadedWorkflow.content).toEqual(defaultGraphJSON)
      expect(loadedWorkflow.originalContent).toEqual(defaultGraphJSON)
      expect(loadedWorkflow.isLoaded).toBe(true)
      expect(loadedWorkflow.activeState).toEqual(defaultGraph)
      expect(loadedWorkflow.initialState).toEqual(defaultGraph)
      expect(loadedWorkflow.isModified).toBe(false)
    })
  })

  describe('openWorkflowsInBackground', () => {
    let workflowA: ComfyWorkflow
    let workflowB: ComfyWorkflow
    let workflowC: ComfyWorkflow

    const openWorkflowPaths = () =>
      store.openWorkflows.filter((w) => store.isOpen(w)).map((w) => w.path)

    beforeEach(async () => {
      await syncRemoteWorkflows(['a.json', 'b.json', 'c.json'])
      workflowA = store.getWorkflowByPath('workflows/a.json')!
      workflowB = store.getWorkflowByPath('workflows/b.json')!
      workflowC = store.getWorkflowByPath('workflows/c.json')!
      vi.mocked(api.getUserData).mockResolvedValue({
        status: 200,
        text: () => Promise.resolve(defaultGraphJSON)
      } as Response)
    })

    it('should open workflows adjacent to the active workflow', async () => {
      await store.openWorkflow(workflowA)
      store.openWorkflowsInBackground({
        left: [workflowB.path],
        right: [workflowC.path]
      })
      expect(openWorkflowPaths()).toEqual([
        workflowB.path,
        workflowA.path,
        workflowC.path
      ])
    })

    it('should not change the active workflow', async () => {
      await store.openWorkflow(workflowA)
      store.openWorkflowsInBackground({
        left: [workflowC.path],
        right: [workflowB.path]
      })
      expect(store.activeWorkflow).not.toBeUndefined()
      expect(store.activeWorkflow!.path).toBe(workflowA.path)
    })

    it('should open workflows when none are active', async () => {
      expect(store.openWorkflows.length).toBe(0)
      store.openWorkflowsInBackground({
        left: [workflowA.path],
        right: [workflowB.path]
      })
      expect(openWorkflowPaths()).toEqual([workflowA.path, workflowB.path])
    })

    it('should not open duplicate workflows', async () => {
      store.openWorkflowsInBackground({
        left: [workflowA.path, workflowB.path, workflowA.path],
        right: [workflowB.path, workflowA.path, workflowB.path]
      })
      expect(openWorkflowPaths()).toEqual([workflowA.path, workflowB.path])
    })

    it('should not open workflow that is already open', async () => {
      await store.openWorkflow(workflowA)
      store.openWorkflowsInBackground({
        left: [workflowA.path]
      })
      expect(openWorkflowPaths()).toEqual([workflowA.path])
      expect(store.activeWorkflow?.path).toBe(workflowA.path)
    })

    it('should ignore invalid or deleted workflow paths', async () => {
      await store.openWorkflow(workflowA)
      store.openWorkflowsInBackground({
        left: ['workflows/invalid::$-path.json'],
        right: ['workflows/deleted-since-last-session.json']
      })
      expect(openWorkflowPaths()).toEqual([workflowA.path])
      expect(store.activeWorkflow?.path).toBe(workflowA.path)
    })

    it('should do nothing when given an empty argument', async () => {
      await store.openWorkflow(workflowA)

      store.openWorkflowsInBackground({})
      expect(openWorkflowPaths()).toEqual([workflowA.path])

      store.openWorkflowsInBackground({ left: [], right: [] })
      expect(openWorkflowPaths()).toEqual([workflowA.path])

      expect(store.activeWorkflow?.path).toBe(workflowA.path)
    })
  })

  describe('renameWorkflow', () => {
    it('should rename workflow and update bookmarks', async () => {
      const workflow = store.createTemporary('dir/test.json')

      // Set up initial bookmark
      expect(workflow.path).toBe('workflows/dir/test.json')
      await bookmarkStore.setBookmarked(workflow.path, true)
      expect(bookmarkStore.isBookmarked(workflow.path)).toBe(true)

      // Mock super.rename
      vi.spyOn(Object.getPrototypeOf(workflow), 'rename').mockImplementation(
        async function (this: unknown, ...args: unknown[]) {
          const newPath = args[0] as string
          ;(this as typeof workflow).path = newPath
          return this as typeof workflow
        }
      )

      // Perform rename
      const newPath = 'workflows/dir/renamed.json'
      await store.renameWorkflow(workflow, newPath)

      // Check that bookmark was transferred
      expect(bookmarkStore.isBookmarked(newPath)).toBe(true)
      expect(bookmarkStore.isBookmarked('workflows/dir/test.json')).toBe(false)
    })

    it('should rename workflow without affecting bookmarks if not bookmarked', async () => {
      const workflow = store.createTemporary('test.json')

      // Verify not bookmarked initially
      expect(bookmarkStore.isBookmarked(workflow.path)).toBe(false)

      // Mock super.rename
      vi.spyOn(Object.getPrototypeOf(workflow), 'rename').mockImplementation(
        async function (this: unknown, ...args: unknown[]) {
          const newPath = args[0] as string
          ;(this as typeof workflow).path = newPath
          return this as typeof workflow
        }
      )

      // Perform rename
      const newName = 'renamed'
      await workflow.rename(newName)

      // Check that no bookmarks were affected
      expect(bookmarkStore.isBookmarked(workflow.path)).toBe(false)
      expect(bookmarkStore.isBookmarked('test.json')).toBe(false)
    })
  })

  describe('closeWorkflow', () => {
    it('should close a workflow', async () => {
      const workflow = store.createTemporary('test.json')
      await store.openWorkflow(workflow)
      expect(store.isOpen(workflow)).toBe(true)
      expect(store.getWorkflowByPath(workflow.path)).not.toBeNull()
      await store.closeWorkflow(workflow)
      expect(store.isOpen(workflow)).toBe(false)
      expect(store.getWorkflowByPath(workflow.path)).toBeNull()
    })
  })

  describe('deleteWorkflow', () => {
    it('should close and delete an open workflow', async () => {
      const workflow = store.createTemporary('test.json')

      // Mock the necessary methods
      vi.spyOn(workflow, 'delete').mockResolvedValue()

      // Open the workflow first
      await store.openWorkflow(workflow)

      // Delete the workflow
      await store.deleteWorkflow(workflow)

      // Verify workflow was closed and deleted
      expect(workflow.delete).toHaveBeenCalled()
    })

    it('should remove bookmark when deleting a bookmarked workflow', async () => {
      const workflow = store.createTemporary('test.json')

      // Mock delete method
      vi.spyOn(workflow, 'delete').mockResolvedValue()

      // Bookmark the workflow
      await bookmarkStore.setBookmarked(workflow.path, true)
      expect(bookmarkStore.isBookmarked(workflow.path)).toBe(true)

      // Delete the workflow
      await store.deleteWorkflow(workflow)

      // Verify bookmark was removed
      expect(bookmarkStore.isBookmarked(workflow.path)).toBe(false)
    })
  })

  describe('save', () => {
    it('should save workflow content and reset modification state', async () => {
      await syncRemoteWorkflows(['test.json'])
      const workflow = store.getWorkflowByPath('workflows/test.json')!

      workflow.changeTracker = createMockChangeTracker({
        workflow
      })
      vi.mocked(api.storeUserData).mockResolvedValue({
        status: 200,
        json: () =>
          Promise.resolve({
            path: 'workflows/test.json',
            modified: Date.now(),
            size: 2
          })
      } as Response)

      // Save the workflow
      await workflow.save()

      // Verify the content was updated
      expect(workflow.content).toBe(
        JSON.stringify(workflow.changeTracker!.activeState)
      )
      expect(workflow.changeTracker!.reset).toHaveBeenCalled()
      expect(workflow.isModified).toBe(false)
    })

    it('should save workflow even if isModified is screwed by changeTracker', async () => {
      await syncRemoteWorkflows(['test.json'])
      const workflow = store.getWorkflowByPath('workflows/test.json')!
      workflow.isModified = false

      workflow.changeTracker = createMockChangeTracker({
        workflow
      })
      vi.mocked(api.storeUserData).mockResolvedValue({
        status: 200,
        json: () =>
          Promise.resolve({
            path: 'workflows/test.json',
            modified: Date.now(),
            size: 2
          })
      } as Response)

      // Save the workflow
      await workflow.save()

      // Verify storeUserData was called
      expect(api.storeUserData).toHaveBeenCalled()

      // Verify the content was updated
      expect(workflow.changeTracker!.reset).toHaveBeenCalled()
      expect(workflow.isModified).toBe(false)
    })
  })

  describe('saveAs', () => {
    it('should save workflow to new path and reset modification state', async () => {
      await syncRemoteWorkflows(['test.json'])
      const workflow = store.getWorkflowByPath('workflows/test.json')!
      workflow.isModified = true

      workflow.changeTracker = createMockChangeTracker({
        workflow
      })
      vi.mocked(api.storeUserData).mockResolvedValue({
        status: 200,
        json: () =>
          Promise.resolve({
            path: 'workflows/new-test.json',
            modified: Date.now(),
            size: 2
          })
      } as Response)

      // Save the workflow with new path
      const newWorkflow = await workflow.saveAs('workflows/new-test.json')

      // Verify the content was updated
      expect(workflow.path).toBe('workflows/test.json')
      expect(workflow.isModified).toBe(true)

      expect(newWorkflow.path).toBe('workflows/new-test.json')
      expect(newWorkflow.content).toBe(
        JSON.stringify(workflow.changeTracker!.activeState)
      )
      expect(newWorkflow.isModified).toBe(false)
    })
  })

  describe('Subgraphs', () => {
    beforeEach(async () => {
      // Ensure canvas exists for these tests
      vi.mocked(comfyApp).canvas = createMockCanvas({
        subgraph: undefined
      }) as typeof comfyApp.canvas

      // Setup an active workflow as updateActiveGraph depends on it
      const workflow = store.createTemporary('test-subgraph-workflow.json')
      // Mock load to avoid actual file operations/parsing
      vi.spyOn(workflow, 'load').mockImplementation(async () => {
        workflow.changeTracker = createMockChangeTracker({
          workflow
        })
        workflow.originalContent = '{}'
        workflow.content = '{}'
        return workflow as LoadedComfyWorkflow
      })
      await store.openWorkflow(workflow)

      // Reset mocks before each subgraph test
      vi.mocked(comfyApp.canvas).subgraph = undefined // Use undefined for root graph
    })

    it('should handle when comfyApp.canvas is not available', async () => {
      // Arrange
      vi.mocked(comfyApp).canvas = null! as typeof comfyApp.canvas

      // Act
      console.debug(store.isSubgraphActive)
      store.updateActiveGraph()
      await nextTick()

      // Assert
      console.debug(store.isSubgraphActive)
      expect(store.isSubgraphActive).toBe(false) // Should default to false
      expect(store.activeSubgraph).toBeUndefined() // Should default to empty
    })

    it('should correctly update state when the root graph is active', async () => {
      // Arrange: Ensure comfyApp indicates root graph is active
      vi.mocked(comfyApp.canvas).subgraph = undefined // Use undefined for root graph

      // Act: Trigger the update
      store.updateActiveGraph()
      await nextTick() // Wait for Vue reactivity

      // Assert: Check store state
      expect(store.isSubgraphActive).toBe(false)
      expect(store.activeSubgraph).toBeUndefined()
    })

    it('should correctly update state when a subgraph is active', async () => {
      // Arrange: Setup mock subgraph structure
      const mockSubgraph = {
        name: 'Level 2 Subgraph',
        isRootGraph: false,
        pathToRootGraph: [
          { name: 'Root' }, // Root Graph (index 0, ignored)
          { name: 'Level 1 Subgraph' },
          { name: 'Level 2 Subgraph' }
        ]
      } as Partial<Subgraph> as Subgraph
      vi.mocked(comfyApp.canvas).subgraph = mockSubgraph

      // Mock isSubgraph to return true for our mockSubgraph
      vi.mocked(isSubgraph).mockImplementation(
        (obj): obj is Subgraph => obj === mockSubgraph
      )

      // Act: Trigger the update
      store.updateActiveGraph()
      await nextTick() // Wait for Vue reactivity

      // Assert: Check store state
      expect(store.isSubgraphActive).toBe(true)
      expect(store.activeSubgraph).toEqual(mockSubgraph)
    })

    it('should update automatically when activeWorkflow changes', async () => {
      // Arrange: Set initial canvas state (e.g., a subgraph)
      const initialSubgraph = {
        name: 'Initial Subgraph',
        pathToRootGraph: [{ name: 'Root' }, { name: 'Initial Subgraph' }],
        isRootGraph: false
      } as Partial<Subgraph> as Subgraph
      vi.mocked(comfyApp.canvas).subgraph = initialSubgraph

      // Mock isSubgraph to return true for our initialSubgraph
      vi.mocked(isSubgraph).mockImplementation(
        (obj): obj is Subgraph => obj === initialSubgraph
      )

      // Trigger initial update based on the *first* workflow opened in beforeEach
      store.updateActiveGraph()
      await nextTick()

      // Verify initial state
      expect(store.isSubgraphActive).toBe(true)
      expect(store.activeSubgraph).toEqual(initialSubgraph)

      // Act: Change the active workflow
      const workflow2 = store.createTemporary('workflow2.json')
      // Mock load for the second workflow
      vi.spyOn(workflow2, 'load').mockImplementation(async () => {
        workflow2.changeTracker = createMockChangeTracker({
          workflow: workflow2
        })
        workflow2.originalContent = '{}'
        workflow2.content = '{}'
        return workflow2 as LoadedComfyWorkflow
      })

      // Before changing workflow, set the canvas state to something different (e.g., root)
      // This ensures the watcher *does* cause a state change we can assert
      vi.mocked(comfyApp.canvas).subgraph = undefined

      // Mock isSubgraph to return false for undefined
      vi.mocked(isSubgraph).mockImplementation(
        (_obj): _obj is Subgraph => false
      )

      await store.openWorkflow(workflow2) // This changes activeWorkflow and triggers the watch
      await nextTick() // Allow watcher and potential async operations in updateActiveGraph to complete

      // Assert: Check that the state was updated by the watcher based on the *new* canvas state
      expect(store.isSubgraphActive).toBe(false) // Should reflect the change to undefined subgraph
      expect(store.activeSubgraph).toBeUndefined()
    })
  })

  describe('NodeLocatorId conversions', () => {
    beforeEach(() => {
      // Setup mock graph structure with subgraphs
      const mockRootGraph = {
        _nodes: [] as unknown[],
        nodes: [] as unknown[],
        subgraphs: new Map(),
        getNodeById: (id: string | number) => {
          if (String(id) === '123') return mockNode
          return null
        }
      }

      const mockSubgraph = {
        id: 'a1b2c3d4-e5f6-7890-abcd-ef1234567890',
        rootGraph: mockRootGraph as LGraph,
        _nodes: [],
        nodes: [],
        clear() {
          return undefined
        }
      } as Partial<Subgraph> as Subgraph

      const mockNode = {
        id: 123,
        isSubgraphNode: () => true,
        subgraph: mockSubgraph
      }

      mockRootGraph._nodes = [mockNode]
      mockRootGraph.nodes = [mockNode]
      mockRootGraph.subgraphs = new Map([[mockSubgraph.id, mockSubgraph]])

      vi.mocked(comfyApp).rootGraph = mockRootGraph as LGraph
      vi.mocked(comfyApp.canvas).subgraph = mockSubgraph
      store.activeSubgraph = mockSubgraph
    })

    describe('nodeIdToNodeLocatorId', () => {
      it('should convert node ID to NodeLocatorId for subgraph nodes', () => {
        const result = store.nodeIdToNodeLocatorId(456)
        expect(result).toBe('a1b2c3d4-e5f6-7890-abcd-ef1234567890:456')
      })

      it('should return simple node ID for root graph nodes', () => {
        store.activeSubgraph = undefined
        const result = store.nodeIdToNodeLocatorId(123)
        expect(result).toBe('123')
      })

      it('should use provided subgraph instead of active one', () => {
        const customSubgraph = {
          id: 'custom-uuid-1234-5678-90ab-cdef12345678',
          rootGraph: undefined! as LGraph,
          _nodes: [],
          nodes: [],
          clear: vi.fn()
        } as Partial<Subgraph> as Subgraph
        const result = store.nodeIdToNodeLocatorId(789, customSubgraph)
        expect(result).toBe('custom-uuid-1234-5678-90ab-cdef12345678:789')
      })
    })

    describe('nodeExecutionIdToNodeLocatorId', () => {
      it('should convert execution ID to NodeLocatorId', () => {
        const result = store.nodeExecutionIdToNodeLocatorId('123:456')
        expect(result).toBe('a1b2c3d4-e5f6-7890-abcd-ef1234567890:456')
      })

      it('should return simple node ID for root level nodes', () => {
        const result = store.nodeExecutionIdToNodeLocatorId('123')
        expect(result).toBe('123')
      })

      it('should return null for invalid execution IDs', () => {
        const result = store.nodeExecutionIdToNodeLocatorId('999:456')
        expect(result).toBeNull()
      })
    })

    describe('nodeLocatorIdToNodeId', () => {
      it('should extract node ID from NodeLocatorId', () => {
        const result = store.nodeLocatorIdToNodeId(
          'a1b2c3d4-e5f6-7890-abcd-ef1234567890:456'
        )
        expect(result).toBe(456)
      })

      it('should handle string node IDs', () => {
        const result = store.nodeLocatorIdToNodeId(
          'a1b2c3d4-e5f6-7890-abcd-ef1234567890:node_1'
        )
        expect(result).toBe('node_1')
      })

      it('should handle simple node IDs (root graph)', () => {
        const result = store.nodeLocatorIdToNodeId('123')
        expect(result).toBe(123)

        const stringResult = store.nodeLocatorIdToNodeId('node_1')
        expect(stringResult).toBe('node_1')
      })

      it('should return null for invalid NodeLocatorId', () => {
        const result = store.nodeLocatorIdToNodeId('invalid:format')
        expect(result).toBeNull()
      })
    })

    describe('nodeLocatorIdToNodeExecutionId', () => {
      it('should convert NodeLocatorId to execution ID', () => {
        // Need to mock isSubgraph to identify our mockSubgraph
        vi.mocked(isSubgraph).mockImplementation((obj): obj is Subgraph => {
          return obj === store.activeSubgraph
        })

        const result = store.nodeLocatorIdToNodeExecutionId(
          'a1b2c3d4-e5f6-7890-abcd-ef1234567890:456'
        )
        expect(result).toBe('123:456')
      })

      it('should handle simple node IDs (root graph)', () => {
        const result = store.nodeLocatorIdToNodeExecutionId('123')
        expect(result).toBe('123')
      })

      it('should return null for unknown subgraph UUID', () => {
        const result = store.nodeLocatorIdToNodeExecutionId(
          'unknown-uuid-1234-5678-90ab-cdef12345678:456'
        )
        expect(result).toBeNull()
      })

      it('should return null for invalid NodeLocatorId', () => {
        const result = store.nodeLocatorIdToNodeExecutionId('invalid:format')
        expect(result).toBeNull()
      })
    })
  })

  describe('Tab Activation History', () => {
    let workflowA: ComfyWorkflow
    let workflowB: ComfyWorkflow
    let workflowC: ComfyWorkflow

    beforeEach(async () => {
      await syncRemoteWorkflows(['a.json', 'b.json', 'c.json'])
      workflowA = store.getWorkflowByPath('workflows/a.json')!
      workflowB = store.getWorkflowByPath('workflows/b.json')!
      workflowC = store.getWorkflowByPath('workflows/c.json')!
      vi.mocked(api.getUserData).mockResolvedValue({
        status: 200,
        text: () => Promise.resolve(defaultGraphJSON)
      } as Response)
    })

    it('should return most recently active workflow', async () => {
      // Open workflows in order: A -> B -> C
      await store.openWorkflow(workflowA)
      await store.openWorkflow(workflowB)
      await store.openWorkflow(workflowC)

      // C is current, B should be most recent
      const mostRecent = store.getMostRecentWorkflow()
      expect(mostRecent?.path).toBe(workflowB.path)
    })

    it('should skip closed workflows (lazy cleanup)', async () => {
      // Open workflows: A -> B -> C
      await store.openWorkflow(workflowA)
      await store.openWorkflow(workflowB)
      await store.openWorkflow(workflowC)

      // Close B (the most recent before C)
      await store.closeWorkflow(workflowB)

      // C is current, B is closed, so A should be returned
      const mostRecent = store.getMostRecentWorkflow()
      expect(mostRecent?.path).toBe(workflowA.path)
    })

    it('should return null when no valid history exists', async () => {
      // Open only one workflow
      await store.openWorkflow(workflowA)

      // No previous workflows, should return null
      const mostRecent = store.getMostRecentWorkflow()
      expect(mostRecent).toBeNull()
    })

    it('should track history when opening workflows', async () => {
      // Open A, then B, then A again
      await store.openWorkflow(workflowA)
      await store.openWorkflow(workflowB)
      await store.openWorkflow(workflowA)

      // A is current, B should be most recent
      const mostRecent = store.getMostRecentWorkflow()
      expect(mostRecent?.path).toBe(workflowB.path)
    })

    it('should handle workflow activated multiple times', async () => {
      // Open: A -> B -> A -> C
      await store.openWorkflow(workflowA)
      await store.openWorkflow(workflowB)
      await store.openWorkflow(workflowA)
      await store.openWorkflow(workflowC)

      // C is current, A should be most recent (not B)
      const mostRecent = store.getMostRecentWorkflow()
      expect(mostRecent?.path).toBe(workflowA.path)
    })

    it('should clean up history when all previous workflows are closed', async () => {
      // Open: A -> B -> C
      await store.openWorkflow(workflowA)
      await store.openWorkflow(workflowB)
      await store.openWorkflow(workflowC)

      // Close A and B
      await store.closeWorkflow(workflowA)
      await store.closeWorkflow(workflowB)

      // C is current, no valid history
      const mostRecent = store.getMostRecentWorkflow()
      expect(mostRecent).toBeNull()
    })
  })

  describe('closeWorkflow draft cleanup', () => {
    it('should remove draft for persisted workflows on close', async () => {
      const draftStore = useWorkflowDraftStore()
      await syncRemoteWorkflows(['a.json'])
      const workflow = store.getWorkflowByPath('workflows/a.json')!

      draftStore.saveDraft('workflows/a.json', {
        data: '{"dirty":true}',
        updatedAt: Date.now(),
        name: 'a.json',
        isTemporary: false
      })
      expect(draftStore.getDraft('workflows/a.json')).toBeDefined()

      await store.closeWorkflow(workflow)

      expect(draftStore.getDraft('workflows/a.json')).toBeUndefined()
    })

    it('should remove draft for temporary workflows on close', async () => {
      const draftStore = useWorkflowDraftStore()
      const workflow = store.createTemporary('temp.json')

      draftStore.saveDraft(workflow.path, {
        data: '{"dirty":true}',
        updatedAt: Date.now(),
        name: 'temp.json',
        isTemporary: true
      })
      expect(draftStore.getDraft(workflow.path)).toBeDefined()

      await store.closeWorkflow(workflow)

      expect(draftStore.getDraft(workflow.path)).toBeUndefined()
    })
  })
})
