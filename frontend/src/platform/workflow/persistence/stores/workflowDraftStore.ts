import { useStorage } from '@vueuse/core'
import { defineStore } from 'pinia'
import { computed } from 'vue'

import type {
  DraftCacheState,
  WorkflowDraftSnapshot
} from '@/platform/workflow/persistence/base/draftCache'
import {
  MAX_DRAFTS,
  createDraftCacheState,
  mostRecentDraftPath,
  moveDraft as moveDraftEntry,
  removeDraft as removeDraftEntry,
  touchEntry,
  upsertDraft
} from '@/platform/workflow/persistence/base/draftCache'
import { api } from '@/scripts/api'
import { app as comfyApp } from '@/scripts/app'

const DRAFTS_STORAGE_KEY = 'Comfy.Workflow.Drafts'
const ORDER_STORAGE_KEY = 'Comfy.Workflow.DraftOrder'

interface LoadPersistedWorkflowOptions {
  workflowName: string | null
  preferredPath?: string | null
  fallbackToLatestDraft?: boolean
}

export const useWorkflowDraftStore = defineStore('workflowDraft', () => {
  const storedDrafts = useStorage<Record<string, WorkflowDraftSnapshot>>(
    DRAFTS_STORAGE_KEY,
    {}
  )
  const storedOrder = useStorage<string[]>(ORDER_STORAGE_KEY, [])

  const mostRecentDraft = computed(() => mostRecentDraftPath(storedOrder.value))

  const currentState = (): DraftCacheState =>
    createDraftCacheState(storedDrafts.value, storedOrder.value)

  const updateState = (state: DraftCacheState) => {
    storedDrafts.value = state.drafts
    storedOrder.value = state.order
  }

  const saveDraft = (path: string, snapshot: WorkflowDraftSnapshot) => {
    try {
      updateState(upsertDraft(currentState(), path, snapshot, MAX_DRAFTS))
    } catch (error) {
      if (
        error instanceof DOMException &&
        error.name === 'QuotaExceededError'
      ) {
        const state = currentState()
        if (state.order.length > 0) {
          const oldestPath = state.order[0]
          updateState(removeDraftEntry(state, oldestPath))
          updateState(upsertDraft(currentState(), path, snapshot, MAX_DRAFTS))
        } else {
          throw error
        }
      } else {
        throw error
      }
    }
  }

  const removeDraft = (path: string) => {
    updateState(removeDraftEntry(currentState(), path))
  }

  const moveDraft = (oldPath: string, newPath: string, name: string) => {
    updateState(moveDraftEntry(currentState(), oldPath, newPath, name))
  }

  const markDraftUsed = (path: string) => {
    if (!(path in storedDrafts.value)) return
    storedOrder.value = touchEntry(storedOrder.value, path)
  }

  const getDraft = (path: string) => storedDrafts.value[path]

  const tryLoadGraph = async (
    payload: string | null,
    workflowName: string | null,
    onFailure?: () => void
  ) => {
    if (!payload) return false
    try {
      const workflow = JSON.parse(payload)
      await comfyApp.loadGraphData(
        workflow,
        /* clean= */ true,
        /* restore_view= */ true,
        workflowName
      )
      return true
    } catch (err) {
      console.error('Failed to load persisted workflow', err)
      onFailure?.()
      return false
    }
  }

  const loadDraft = async (path: string) => {
    const draft = getDraft(path)
    if (!draft) return false
    const loaded = await tryLoadGraph(draft.data, draft.name, () => {
      removeDraft(path)
    })
    if (loaded) {
      markDraftUsed(path)
    }
    return loaded
  }

  const loadPersistedWorkflow = async (
    options: LoadPersistedWorkflowOptions
  ): Promise<boolean> => {
    const {
      workflowName,
      preferredPath,
      fallbackToLatestDraft = false
    } = options

    if (preferredPath && (await loadDraft(preferredPath))) {
      return true
    }

    if (!preferredPath && fallbackToLatestDraft) {
      const fallbackPath = mostRecentDraft.value
      if (fallbackPath && (await loadDraft(fallbackPath))) {
        return true
      }
    }

    const clientId = api.initialClientId ?? api.clientId
    if (clientId) {
      const sessionPayload = sessionStorage.getItem(`workflow:${clientId}`)
      if (await tryLoadGraph(sessionPayload, workflowName)) {
        return true
      }
    }

    const localPayload = localStorage.getItem('workflow')
    return await tryLoadGraph(localPayload, workflowName)
  }

  return {
    saveDraft,
    removeDraft,
    moveDraft,
    markDraftUsed,
    getDraft,
    loadPersistedWorkflow,
    reset: () => {
      updateState(createDraftCacheState())
    }
  }
})
