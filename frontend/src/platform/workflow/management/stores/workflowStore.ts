import _ from 'es-toolkit/compat'
import { until, useAsyncState } from '@vueuse/core'
import { defineStore } from 'pinia'
import { computed, markRaw, ref, shallowRef, watch } from 'vue'
import type { Raw } from 'vue'

import type {
  LGraph,
  LGraphNode,
  Subgraph
} from '@/lib/litegraph/src/litegraph'
import type {
  ComfyWorkflowJSON,
  NodeId
} from '@/platform/workflow/validation/schemas/workflowSchema'
import { useWorkflowDraftStore } from '@/platform/workflow/persistence/stores/workflowDraftStore'
import { useWorkflowThumbnail } from '@/renderer/core/thumbnail/useWorkflowThumbnail'
import { api } from '@/scripts/api'
import { app as comfyApp } from '@/scripts/app'
import { defaultGraphJSON } from '@/scripts/defaultGraph'
import type { NodeExecutionId, NodeLocatorId } from '@/types/nodeIdentification'
import {
  createNodeExecutionId,
  createNodeLocatorId,
  parseNodeExecutionId,
  parseNodeLocatorId
} from '@/types/nodeIdentification'
import { generateUUID, getPathDetails } from '@/utils/formatUtil'
import { syncEntities } from '@/utils/syncUtil'
import { isSubgraph } from '@/utils/typeGuardUtil'
import { ComfyWorkflow } from './comfyWorkflow'
import type { LoadedComfyWorkflow } from './comfyWorkflow'
export { ComfyWorkflow, type LoadedComfyWorkflow }

/**
 * Exposed store interface for the workflow store.
 * Explicitly typed to avoid trigger following error:
 * error TS7056: The inferred type of this node exceeds the maximum length the
 * compiler will serialize. An explicit type annotation is needed.
 */
interface WorkflowStore {
  activeWorkflow: LoadedComfyWorkflow | null
  attachWorkflow: (workflow: ComfyWorkflow, openIndex?: number) => void
  isActive: (workflow: ComfyWorkflow) => boolean
  openWorkflows: ComfyWorkflow[]
  openedWorkflowIndexShift: (shift: number) => ComfyWorkflow | null
  getMostRecentWorkflow: () => ComfyWorkflow | null
  openWorkflow: (workflow: ComfyWorkflow) => Promise<LoadedComfyWorkflow>
  openWorkflowsInBackground: (paths: {
    left?: string[]
    right?: string[]
  }) => void
  isOpen: (workflow: ComfyWorkflow) => boolean
  isBusy: boolean
  closeWorkflow: (workflow: ComfyWorkflow) => Promise<void>
  createTemporary: (
    path?: string,
    workflowData?: ComfyWorkflowJSON
  ) => ComfyWorkflow
  createNewTemporary: (
    path?: string,
    workflowData?: ComfyWorkflowJSON
  ) => ComfyWorkflow
  renameWorkflow: (workflow: ComfyWorkflow, newPath: string) => Promise<void>
  deleteWorkflow: (workflow: ComfyWorkflow) => Promise<void>
  saveWorkflow: (workflow: ComfyWorkflow) => Promise<void>

  workflows: ComfyWorkflow[]
  bookmarkedWorkflows: ComfyWorkflow[]
  persistedWorkflows: ComfyWorkflow[]
  modifiedWorkflows: ComfyWorkflow[]
  getWorkflowByPath: (path: string) => ComfyWorkflow | null
  syncWorkflows: (dir?: string) => Promise<void>
  reorderWorkflows: (from: number, to: number) => void

  /** `true` if any subgraph is currently being viewed. */
  isSubgraphActive: boolean
  activeSubgraph: Subgraph | undefined
  /** Updates the {@link subgraphNamePath} and {@link isSubgraphActive} values. */
  updateActiveGraph: () => void
  executionIdToCurrentId: (id: string) => string | undefined
  nodeIdToNodeLocatorId: (nodeId: NodeId, subgraph?: Subgraph) => NodeLocatorId
  nodeToNodeLocatorId: (node: LGraphNode) => NodeLocatorId
  nodeExecutionIdToNodeLocatorId: (
    nodeExecutionId: NodeExecutionId | string
  ) => NodeLocatorId | null
  nodeLocatorIdToNodeId: (locatorId: NodeLocatorId | string) => NodeId | null
  nodeLocatorIdToNodeExecutionId: (
    locatorId: NodeLocatorId | string,
    targetSubgraph?: Subgraph
  ) => NodeExecutionId | null
}

export const useWorkflowStore = defineStore('workflow', () => {
  /**
   * History of tab activations. Most recent at the end.
   * Tracks the order in which tabs were activated to support "go to previous" behavior.
   * Lazily cleaned on access.
   */
  const tabActivationHistory = ref<string[]>([])
  const MAX_HISTORY_SIZE = 32

  /**
   * Detach the workflow from the store. lightweight helper function.
   * @param workflow The workflow to detach.
   * @returns The index of the workflow in the openWorkflowPaths array, or -1 if the workflow was not open.
   */
  const detachWorkflow = (workflow: ComfyWorkflow) => {
    delete workflowLookup.value[workflow.path]
    const index = openWorkflowPaths.value.indexOf(workflow.path)
    if (index !== -1) {
      openWorkflowPaths.value = openWorkflowPaths.value.filter(
        (path) => path !== workflow.path
      )
    }
    return index
  }

  /**
   * Attach the workflow to the store. lightweight helper function.
   * @param workflow The workflow to attach.
   * @param openIndex The index to open the workflow at.
   */
  const attachWorkflow = (workflow: ComfyWorkflow, openIndex: number = -1) => {
    workflowLookup.value[workflow.path] = workflow

    if (openIndex !== -1) {
      openWorkflowPaths.value.splice(openIndex, 0, workflow.path)
    }
  }

  /**
   * The active workflow currently being edited.
   */
  const activeWorkflow = ref<LoadedComfyWorkflow | null>(null)
  const isActive = (workflow: ComfyWorkflow) =>
    activeWorkflow.value?.path === workflow.path
  /**
   * All workflows.
   */
  const workflowLookup = ref<Record<string, ComfyWorkflow>>({})
  const workflows = computed<ComfyWorkflow[]>(() =>
    Object.values(workflowLookup.value)
  )
  const getWorkflowByPath = (path: string): ComfyWorkflow | null =>
    workflowLookup.value[path] ?? null

  /**
   * The paths of the open workflows. It is setup as a ref to allow user
   * to reorder the workflows opened.
   */
  const openWorkflowPaths = ref<string[]>([])
  const openWorkflowPathSet = computed(() => new Set(openWorkflowPaths.value))
  const openWorkflows = computed(() =>
    openWorkflowPaths.value.map((path) => workflowLookup.value[path])
  )
  const reorderWorkflows = (from: number, to: number) => {
    const movedTab = openWorkflowPaths.value[from]
    openWorkflowPaths.value.splice(from, 1)
    openWorkflowPaths.value.splice(to, 0, movedTab)
  }
  const isOpen = (workflow: ComfyWorkflow) =>
    openWorkflowPathSet.value.has(workflow.path)

  /**
   * Add paths to the list of open workflow paths without loading the files
   * or changing the active workflow.
   *
   * @param paths - The workflows to open, specified as:
   *   - `left`: Workflows to be added to the left.
   *   - `right`: Workflows to be added to the right.
   *
   * Invalid paths (non-strings or paths not found in `workflowLookup.value`)
   * will be ignored. Duplicate paths are automatically removed.
   */
  const openWorkflowsInBackground = (paths: {
    left?: string[]
    right?: string[]
  }) => {
    const { left = [], right = [] } = paths
    if (!left.length && !right.length) return

    const isValidPath = (
      path: unknown
    ): path is keyof typeof workflowLookup.value =>
      typeof path === 'string' && path in workflowLookup.value

    openWorkflowPaths.value = _.union(
      left,
      openWorkflowPaths.value,
      right
    ).filter(isValidPath)
  }

  /**
   * Set the workflow as the active workflow.
   * @param workflow The workflow to open.
   */
  const openWorkflow = async (
    workflow: ComfyWorkflow
  ): Promise<LoadedComfyWorkflow> => {
    if (isActive(workflow)) return workflow as LoadedComfyWorkflow

    if (!openWorkflowPaths.value.includes(workflow.path)) {
      openWorkflowPaths.value.push(workflow.path)
    }
    const loadedWorkflow = await workflow.load()
    activeWorkflow.value = loadedWorkflow
    comfyApp.canvas.bg_tint = loadedWorkflow.tintCanvasBg

    // Track activation in history (move to end if already present)
    const historyIndex = tabActivationHistory.value.indexOf(workflow.path)
    if (historyIndex !== -1) {
      tabActivationHistory.value.splice(historyIndex, 1)
    }
    tabActivationHistory.value.push(workflow.path)
    // Trim history if too large
    if (tabActivationHistory.value.length > MAX_HISTORY_SIZE) {
      tabActivationHistory.value.shift()
    }

    return loadedWorkflow
  }

  const getUnconflictedPath = (basePath: string): string => {
    const { directory, filename, suffix } = getPathDetails(basePath)
    let counter = 2
    let newPath = basePath
    while (workflowLookup.value[newPath]) {
      newPath = `${directory}/${filename} (${counter}).${suffix}`
      counter++
    }
    return newPath
  }
  const saveAs = (
    existingWorkflow: ComfyWorkflow,
    path: string
  ): ComfyWorkflow => {
    // Generate new id when saving existing workflow as a new file
    const id = generateUUID()
    const state = JSON.parse(
      JSON.stringify(existingWorkflow.activeState)
    ) as ComfyWorkflowJSON
    state.id = id

    const workflow: ComfyWorkflow =
      new (existingWorkflow.constructor as typeof ComfyWorkflow)({
        path,
        modified: Date.now(),
        size: -1
      })
    workflow.initialMode = existingWorkflow.initialMode
    workflow.originalContent = workflow.content = JSON.stringify(state)
    workflowLookup.value[workflow.path] = workflow
    return workflow
  }

  /**
   * Helper to create a new temporary workflow
   */
  const createNewWorkflow = (
    path: string,
    workflowData?: ComfyWorkflowJSON
  ): ComfyWorkflow => {
    const workflow = new ComfyWorkflow({
      path,
      modified: Date.now(),
      size: -1
    })

    workflow.originalContent = workflow.content = workflowData
      ? JSON.stringify(workflowData)
      : defaultGraphJSON

    workflowLookup.value[workflow.path] = workflow
    return workflow
  }

  /**
   * Create a temporary workflow, attempting to reuse an existing workflow if conditions match
   */
  const createTemporary = (path?: string, workflowData?: ComfyWorkflowJSON) => {
    const fullPath = getUnconflictedPath(
      ComfyWorkflow.basePath + (path ?? 'Unsaved Workflow.json')
    )

    // Try to reuse an existing loaded workflow with the same filename
    // that is not stored in the workflows directory
    if (path && workflowData) {
      const existingWorkflow = workflows.value.find(
        (w) => w.fullFilename === path
      )
      if (
        existingWorkflow?.changeTracker &&
        !existingWorkflow.directory.startsWith(
          ComfyWorkflow.basePath.slice(0, -1)
        )
      ) {
        existingWorkflow.changeTracker.reset(workflowData)
        return existingWorkflow
      }
    }

    return createNewWorkflow(fullPath, workflowData)
  }

  /**
   * Create a new temporary workflow without attempting to reuse existing workflows
   */
  const createNewTemporary = (
    path?: string,
    workflowData?: ComfyWorkflowJSON
  ): ComfyWorkflow => {
    const fullPath = getUnconflictedPath(
      ComfyWorkflow.basePath + (path ?? 'Unsaved Workflow.json')
    )
    return createNewWorkflow(fullPath, workflowData)
  }

  const closeWorkflow = async (workflow: ComfyWorkflow) => {
    openWorkflowPaths.value = openWorkflowPaths.value.filter(
      (path) => path !== workflow.path
    )
    useWorkflowDraftStore().removeDraft(workflow.path)
    if (workflow.isTemporary) {
      clearThumbnail(workflow.key)
      delete workflowLookup.value[workflow.path]
    } else {
      workflow.unload()
    }
  }

  /**
   * Get the workflow at the given index shift from the active workflow.
   * @param shift The shift to the next workflow. Positive for next, negative for previous.
   * @returns The next workflow or null if the shift is out of bounds.
   */
  const openedWorkflowIndexShift = (shift: number): ComfyWorkflow | null => {
    const index = openWorkflowPaths.value.indexOf(
      activeWorkflow.value?.path ?? ''
    )

    if (index !== -1) {
      const length = openWorkflows.value.length
      const nextIndex = (index + shift + length) % length
      const nextWorkflow = openWorkflows.value[nextIndex]
      return nextWorkflow ?? null
    }
    return null
  }

  /**
   * Get the most recently active workflow from history (excluding current).
   * Lazily cleans invalid paths from history.
   * @returns The most recent valid workflow or null if none found.
   */
  const getMostRecentWorkflow = (): ComfyWorkflow | null => {
    const currentPath = activeWorkflow.value?.path
    const validPaths: string[] = []

    // Scan backwards through history
    for (let i = tabActivationHistory.value.length - 1; i >= 0; i--) {
      const path = tabActivationHistory.value[i]

      // Skip current workflow
      if (path === currentPath) continue

      // Check if workflow is still open
      if (openWorkflowPathSet.value.has(path)) {
        validPaths.unshift(path)
        const workflow = workflowLookup.value[path]
        if (workflow) {
          // Lazy cleanup: keep only valid paths
          tabActivationHistory.value = validPaths
          return workflow
        }
      }
    }

    // Cleanup: no valid workflows found, clear history
    tabActivationHistory.value = []
    return null
  }

  const persistedWorkflows = computed(() =>
    Array.from(workflows.value).filter(
      (workflow) =>
        workflow.isPersisted && !workflow.path.startsWith('subgraphs/')
    )
  )

  const {
    isReady: isSyncReady,
    isLoading: isSyncLoading,
    execute: executeSyncWorkflows
  } = useAsyncState(
    async (dir: string = '') => {
      await syncEntities(
        dir ? 'workflows/' + dir : 'workflows',
        workflowLookup.value,
        (file) =>
          new ComfyWorkflow({
            path: file.path,
            modified: file.modified,
            size: file.size
          }),
        (existingWorkflow, file) => {
          const isActiveWorkflow =
            activeWorkflow.value?.path === existingWorkflow.path

          const nextLastModified = Math.max(
            existingWorkflow.lastModified,
            file.modified
          )

          const isMetadataUnchanged =
            nextLastModified === existingWorkflow.lastModified &&
            file.size === existingWorkflow.size

          if (!isMetadataUnchanged) {
            existingWorkflow.lastModified = nextLastModified
            existingWorkflow.size = file.size
          }

          // Never unload the active workflow - it may contain unsaved in-memory edits.
          if (isActiveWorkflow) {
            return
          }

          // If nothing changed, keep any loaded content cached.
          if (isMetadataUnchanged) {
            return
          }

          existingWorkflow.unload()
        },
        /* exclude */ (workflow) => workflow.isTemporary
      )
    },
    undefined,
    { immediate: false }
  )

  async function syncWorkflows(dir: string = '') {
    return executeSyncWorkflows(0, dir)
  }

  async function loadWorkflows(): Promise<void> {
    if (isSyncReady.value) return

    if (isSyncLoading.value) {
      await until(isSyncLoading).toBe(false)
      return
    }

    await syncWorkflows()
  }

  const bookmarkStore = useWorkflowBookmarkStore()
  const bookmarkedWorkflows = computed(() =>
    workflows.value.filter((workflow) =>
      bookmarkStore.isBookmarked(workflow.path)
    )
  )
  const modifiedWorkflows = computed(() =>
    workflows.value.filter((workflow) => workflow.isModified)
  )

  /** A filesystem operation is currently in progress (e.g. save, rename, delete) */
  const isBusy = ref<boolean>(false)
  const { moveWorkflowThumbnail, clearThumbnail } = useWorkflowThumbnail()

  const renameWorkflow = async (workflow: ComfyWorkflow, newPath: string) => {
    isBusy.value = true
    try {
      // Capture all needed values upfront
      const oldPath = workflow.path
      const oldKey = workflow.key
      const wasBookmarked = bookmarkStore.isBookmarked(oldPath)
      const draftStore = useWorkflowDraftStore()

      const openIndex = detachWorkflow(workflow)
      // Perform the actual rename operation first
      try {
        await workflow.rename(newPath)
      } finally {
        attachWorkflow(workflow, openIndex)
      }

      draftStore.moveDraft(oldPath, newPath, workflow.key)

      // Move thumbnail from old key to new key (using workflow keys, not full paths)
      const newKey = workflow.key
      moveWorkflowThumbnail(oldKey, newKey)
      // Update bookmarks
      if (wasBookmarked) {
        await bookmarkStore.setBookmarked(oldPath, false)
        await bookmarkStore.setBookmarked(newPath, true)
      }
    } finally {
      isBusy.value = false
    }
  }

  const deleteWorkflow = async (workflow: ComfyWorkflow) => {
    isBusy.value = true
    try {
      await workflow.delete()
      useWorkflowDraftStore().removeDraft(workflow.path)
      if (bookmarkStore.isBookmarked(workflow.path)) {
        await bookmarkStore.setBookmarked(workflow.path, false)
      }
      // Clear thumbnail when workflow is deleted
      clearThumbnail(workflow.key)
      delete workflowLookup.value[workflow.path]
    } finally {
      isBusy.value = false
    }
  }

  /**
   * Save a workflow.
   * @param workflow The workflow to save.
   */
  const saveWorkflow = async (workflow: ComfyWorkflow) => {
    isBusy.value = true
    try {
      // Detach the workflow and re-attach to force refresh the tree objects.
      const openIndex = detachWorkflow(workflow)
      try {
        await workflow.save()
      } finally {
        attachWorkflow(workflow, openIndex)
      }
    } finally {
      isBusy.value = false
    }
  }

  /** @see WorkflowStore.isSubgraphActive */
  const isSubgraphActive = ref(false)

  /** @see WorkflowStore.activeSubgraph */
  const activeSubgraph = shallowRef<Raw<Subgraph>>()

  /** @see WorkflowStore.updateActiveGraph */
  const updateActiveGraph = () => {
    const subgraph = comfyApp.canvas?.subgraph
    activeSubgraph.value = subgraph ? markRaw(subgraph) : undefined
    if (!comfyApp.canvas) return

    isSubgraphActive.value = isSubgraph(subgraph)
  }

  const subgraphNodeIdToSubgraph = (id: string, graph: LGraph | Subgraph) => {
    const node = graph.getNodeById(id)
    if (node?.isSubgraphNode()) return node.subgraph
  }

  const getSubgraphsFromInstanceIds = (
    currentGraph: LGraph | Subgraph,
    subgraphNodeIds: string[],
    subgraphs: Subgraph[] = []
  ): Subgraph[] => {
    const currentPart = subgraphNodeIds.shift()
    if (currentPart === undefined) return subgraphs

    const subgraph = subgraphNodeIdToSubgraph(currentPart, currentGraph)
    if (subgraph === undefined) throw new Error('Subgraph not found')

    subgraphs.push(subgraph)
    return getSubgraphsFromInstanceIds(subgraph, subgraphNodeIds, subgraphs)
  }

  //FIXME: use existing util function
  const executionIdToCurrentId = (id: string): string | undefined => {
    const subgraph = activeSubgraph.value

    // Short-circuit: ID belongs to the parent workflow / no active subgraph
    if (!id.includes(':')) {
      return !subgraph ? id : undefined
    } else if (!subgraph) {
      return
    }

    // Parse the execution ID (e.g., "123:456:789")
    const subgraphNodeIds = id.split(':')

    // Start from the root graph
    const graph = comfyApp.rootGraph

    // If the last subgraph is the active subgraph, return the node ID
    const subgraphs = getSubgraphsFromInstanceIds(graph, subgraphNodeIds)
    if (subgraphs.at(-1) === subgraph) {
      return subgraphNodeIds.at(-1)
    }
  }

  watch(activeWorkflow, updateActiveGraph)

  /**
   * Convert a node ID to a NodeLocatorId
   * @param nodeId The local node ID
   * @param subgraph The subgraph containing the node (defaults to active subgraph)
   * @returns The NodeLocatorId (for root graph nodes, returns the node ID as-is)
   */
  const nodeIdToNodeLocatorId = (
    nodeId: NodeId,
    subgraph?: Subgraph
  ): NodeLocatorId => {
    const targetSubgraph = subgraph ?? activeSubgraph.value
    if (!targetSubgraph) {
      // Node is in the root graph, return the node ID as-is
      return String(nodeId)
    }

    return createNodeLocatorId(targetSubgraph.id, nodeId)
  }
  /**
   * Convert a node to a NodeLocatorId
   * Does not assume the node resides in  the active graph
   * @param The actual node instance
   * @returns The NodeLocatorId (for root graph nodes, returns the node ID as-is)
   */
  const nodeToNodeLocatorId = (node: LGraphNode): NodeLocatorId => {
    if (isSubgraph(node.graph))
      return createNodeLocatorId(node.graph.id, node.id)
    return String(node.id)
  }

  /**
   * Convert an execution ID to a NodeLocatorId
   * @param nodeExecutionId The execution node ID (e.g., "123:456:789")
   * @returns The NodeLocatorId or null if conversion fails
   */
  const nodeExecutionIdToNodeLocatorId = (
    nodeExecutionId: NodeExecutionId | string
  ): NodeLocatorId | null => {
    // Handle simple node IDs (root graph - no colons)
    if (!nodeExecutionId.includes(':')) {
      return nodeExecutionId
    }

    const parts = parseNodeExecutionId(nodeExecutionId)
    if (!parts || parts.length === 0) return null

    const nodeId = parts[parts.length - 1]
    const subgraphNodeIds = parts.slice(0, -1)

    if (subgraphNodeIds.length === 0) {
      // Node is in root graph, return the node ID as-is
      return String(nodeId)
    }

    try {
      const subgraphs = getSubgraphsFromInstanceIds(
        comfyApp.rootGraph,
        subgraphNodeIds.map((id) => String(id))
      )
      const immediateSubgraph = subgraphs[subgraphs.length - 1]
      return createNodeLocatorId(immediateSubgraph.id, nodeId)
    } catch {
      return null
    }
  }

  /**
   * Extract the node ID from a NodeLocatorId
   * @param locatorId The NodeLocatorId
   * @returns The local node ID or null if invalid
   */
  const nodeLocatorIdToNodeId = (
    locatorId: NodeLocatorId | string
  ): NodeId | null => {
    const parsed = parseNodeLocatorId(locatorId)
    return parsed?.localNodeId ?? null
  }

  /**
   * Convert a NodeLocatorId to an execution ID for a specific context
   * @param locatorId The NodeLocatorId
   * @param targetSubgraph The subgraph context (defaults to active subgraph)
   * @returns The execution ID or null if the node is not accessible from the target context
   */
  const nodeLocatorIdToNodeExecutionId = (
    locatorId: NodeLocatorId | string,
    targetSubgraph?: Subgraph
  ): NodeExecutionId | null => {
    const parsed = parseNodeLocatorId(locatorId)
    if (!parsed) return null

    const { subgraphUuid, localNodeId } = parsed

    // If no subgraph UUID, this is a root graph node
    if (!subgraphUuid) {
      return String(localNodeId)
    }

    // Find the path from root to the subgraph with this UUID
    const findSubgraphPath = (
      graph: LGraph | Subgraph,
      targetUuid: string,
      path: NodeId[] = []
    ): NodeId[] | null => {
      if (isSubgraph(graph) && graph.id === targetUuid) {
        return path
      }

      for (const node of graph._nodes) {
        if (node.isSubgraphNode() && node.subgraph) {
          const result = findSubgraphPath(node.subgraph, targetUuid, [
            ...path,
            node.id
          ])
          if (result) return result
        }
      }

      return null
    }

    const path = findSubgraphPath(comfyApp.rootGraph, subgraphUuid)
    if (!path) return null

    // If we have a target subgraph, check if the path goes through it
    if (
      targetSubgraph &&
      !path.some((_, idx) => {
        const subgraphs = getSubgraphsFromInstanceIds(
          comfyApp.rootGraph,
          path.slice(0, idx + 1).map((id) => String(id))
        )
        return subgraphs[subgraphs.length - 1] === targetSubgraph
      })
    ) {
      return null
    }

    return createNodeExecutionId([...path, localNodeId])
  }

  return {
    activeWorkflow,
    attachWorkflow,
    isActive,
    openWorkflows,
    openedWorkflowIndexShift,
    getMostRecentWorkflow,
    openWorkflow,
    openWorkflowsInBackground,
    isOpen,
    isBusy,
    closeWorkflow,
    createTemporary,
    createNewTemporary,
    renameWorkflow,
    deleteWorkflow,
    saveAs,
    saveWorkflow,
    reorderWorkflows,

    workflows,
    bookmarkedWorkflows,
    persistedWorkflows,
    modifiedWorkflows,
    getWorkflowByPath,
    syncWorkflows,
    loadWorkflows,

    isSubgraphActive,
    activeSubgraph,
    updateActiveGraph,
    executionIdToCurrentId,
    nodeIdToNodeLocatorId,
    nodeToNodeLocatorId,
    nodeExecutionIdToNodeLocatorId,
    nodeLocatorIdToNodeId,
    nodeLocatorIdToNodeExecutionId
  }
}) satisfies () => WorkflowStore

export const useWorkflowBookmarkStore = defineStore('workflowBookmark', () => {
  const bookmarks = ref<Set<string>>(new Set())

  const isBookmarked = (path: string) => bookmarks.value.has(path)

  const loadBookmarks = async () => {
    const resp = await api.getUserData('workflows/.index.json')
    if (resp.status === 200) {
      const info = await resp.json()
      bookmarks.value = new Set(info?.favorites ?? [])
    }
  }

  const saveBookmarks = async () => {
    await api.storeUserData('workflows/.index.json', {
      favorites: Array.from(bookmarks.value)
    })
  }

  const setBookmarked = async (path: string, value: boolean) => {
    if (bookmarks.value.has(path) === value) return
    if (value) {
      bookmarks.value.add(path)
    } else {
      bookmarks.value.delete(path)
    }
    await saveBookmarks()
  }

  const toggleBookmarked = async (path: string) => {
    await setBookmarked(path, !bookmarks.value.has(path))
  }

  return {
    isBookmarked,
    loadBookmarks,
    saveBookmarks,
    setBookmarked,
    toggleBookmarked
  }
})
