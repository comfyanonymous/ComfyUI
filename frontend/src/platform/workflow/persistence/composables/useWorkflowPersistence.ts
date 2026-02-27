import { useToast } from 'primevue'
import { tryOnScopeDispose } from '@vueuse/core'
import { computed, ref, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'

import {
  hydratePreservedQuery,
  mergePreservedQueryIntoQuery
} from '@/platform/navigation/preservedQueryManager'
import { PRESERVED_QUERY_NAMESPACES } from '@/platform/navigation/preservedQueryNamespaces'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useWorkflowService } from '@/platform/workflow/core/services/workflowService'
import {
  ComfyWorkflow,
  useWorkflowStore
} from '@/platform/workflow/management/stores/workflowStore'
import { useWorkflowDraftStore } from '@/platform/workflow/persistence/stores/workflowDraftStore'
import { useTemplateUrlLoader } from '@/platform/workflow/templates/composables/useTemplateUrlLoader'
import { api } from '@/scripts/api'
import { app as comfyApp } from '@/scripts/app'
import { getStorageValue, setStorageValue } from '@/scripts/utils'
import { useCommandStore } from '@/stores/commandStore'

export function useWorkflowPersistence() {
  const { t } = useI18n()
  const workflowStore = useWorkflowStore()
  const settingStore = useSettingStore()
  const route = useRoute()
  const router = useRouter()
  const templateUrlLoader = useTemplateUrlLoader()
  const TEMPLATE_NAMESPACE = PRESERVED_QUERY_NAMESPACES.TEMPLATE
  const workflowDraftStore = useWorkflowDraftStore()
  const toast = useToast()

  const ensureTemplateQueryFromIntent = async () => {
    hydratePreservedQuery(TEMPLATE_NAMESPACE)
    const mergedQuery = mergePreservedQueryIntoQuery(
      TEMPLATE_NAMESPACE,
      route.query
    )

    if (mergedQuery) {
      await router.replace({ query: mergedQuery })
    }

    return mergedQuery ?? route.query
  }

  const workflowPersistenceEnabled = computed(() =>
    settingStore.get('Comfy.Workflow.Persist')
  )

  const lastSavedJsonByPath = ref<Record<string, string>>({})

  watch(workflowPersistenceEnabled, (enabled) => {
    if (!enabled) {
      workflowDraftStore.reset()
      lastSavedJsonByPath.value = {}
    }
  })

  const persistCurrentWorkflow = () => {
    if (!workflowPersistenceEnabled.value) return
    const activeWorkflow = workflowStore.activeWorkflow
    if (!activeWorkflow) return
    const graphData = comfyApp.rootGraph.serialize()
    const workflowJson = JSON.stringify(graphData)
    const workflowPath = activeWorkflow.path

    if (workflowJson === lastSavedJsonByPath.value[workflowPath]) return

    try {
      workflowDraftStore.saveDraft(activeWorkflow.path, {
        data: workflowJson,
        updatedAt: Date.now(),
        name: activeWorkflow.key,
        isTemporary: activeWorkflow.isTemporary
      })
    } catch (error) {
      console.error('Failed to save draft', error)
      toast.add({
        severity: 'error',
        summary: t('g.error'),
        detail: t('toastMessages.failedToSaveDraft'),
        life: 3000
      })
      return
    }

    try {
      localStorage.setItem('workflow', workflowJson)
      if (api.clientId) {
        sessionStorage.setItem(`workflow:${api.clientId}`, workflowJson)
      }
    } catch (error) {
      // Only log our own keys and aggregate stats
      const ourKeys = Object.keys(sessionStorage).filter(
        (key) => key.startsWith('workflow:') || key === 'workflow'
      )
      console.error('QuotaExceededError details:', {
        workflowSizeKB: Math.round(workflowJson.length / 1024),
        totalStorageItems: Object.keys(sessionStorage).length,
        ourWorkflowKeys: ourKeys.length,
        ourWorkflowSizes: ourKeys.map((key) => ({
          key,
          sizeKB: Math.round(sessionStorage[key].length / 1024)
        })),
        error: error instanceof Error ? error.message : String(error)
      })
      throw error
    }

    lastSavedJsonByPath.value[workflowPath] = workflowJson

    if (!activeWorkflow.isTemporary && !activeWorkflow.isModified) {
      workflowDraftStore.removeDraft(activeWorkflow.path)
      return
    }
  }

  const loadPreviousWorkflowFromStorage = async () => {
    const workflowName = getStorageValue('Comfy.PreviousWorkflow')
    const preferredPath = workflowName
      ? `${ComfyWorkflow.basePath}${workflowName}`
      : null
    return await workflowDraftStore.loadPersistedWorkflow({
      workflowName,
      preferredPath,
      fallbackToLatestDraft: !workflowName
    })
  }

  const loadDefaultWorkflow = async () => {
    if (!settingStore.get('Comfy.TutorialCompleted')) {
      await settingStore.set('Comfy.TutorialCompleted', true)
      await useWorkflowService().loadBlankWorkflow()
      await useCommandStore().execute('Comfy.BrowseTemplates')
    } else {
      await comfyApp.loadGraphData()
    }
  }

  const initializeWorkflow = async () => {
    if (!workflowPersistenceEnabled.value) return

    try {
      const restored = await loadPreviousWorkflowFromStorage()
      if (!restored) {
        await loadDefaultWorkflow()
      }
    } catch (err) {
      console.error('Error loading previous workflow', err)
      await loadDefaultWorkflow()
    }
  }

  const loadTemplateFromUrlIfPresent = async () => {
    const query = await ensureTemplateQueryFromIntent()
    const hasTemplateUrl = query.template && typeof query.template === 'string'

    if (hasTemplateUrl) {
      await templateUrlLoader.loadTemplateFromUrl()
    }
  }

  // Setup watchers
  watch(
    () => workflowStore.activeWorkflow?.key,
    (activeWorkflowKey) => {
      if (!activeWorkflowKey) return
      setStorageValue('Comfy.PreviousWorkflow', activeWorkflowKey)
      // When the activeWorkflow changes, the graph has already been loaded.
      // Saving the current state of the graph to the localStorage.
      persistCurrentWorkflow()
    }
  )

  api.addEventListener('graphChanged', persistCurrentWorkflow)

  // Clean up event listener when component unmounts
  tryOnScopeDispose(() => {
    api.removeEventListener('graphChanged', persistCurrentWorkflow)
  })

  // Restore workflow tabs states
  const openWorkflows = computed(() => workflowStore.openWorkflows)
  const activeWorkflow = computed(() => workflowStore.activeWorkflow)
  const restoreState = computed<{ paths: string[]; activeIndex: number }>(
    () => {
      if (!openWorkflows.value || !activeWorkflow.value) {
        return { paths: [], activeIndex: -1 }
      }

      const paths = openWorkflows.value
        .map((workflow) => workflow?.path)
        .filter(
          (path): path is string =>
            typeof path === 'string' && path.startsWith(ComfyWorkflow.basePath)
        )
      const activeIndex = paths.indexOf(activeWorkflow.value.path)

      return { paths, activeIndex }
    }
  )

  // Get storage values before setting watchers
  const parsedWorkflows = JSON.parse(
    getStorageValue('Comfy.OpenWorkflowsPaths') || '[]'
  )
  const storedWorkflows = Array.isArray(parsedWorkflows)
    ? parsedWorkflows.filter(
        (entry): entry is string => typeof entry === 'string'
      )
    : []
  const parsedIndex = JSON.parse(
    getStorageValue('Comfy.ActiveWorkflowIndex') || '-1'
  )
  const storedActiveIndex =
    typeof parsedIndex === 'number' && Number.isFinite(parsedIndex)
      ? parsedIndex
      : -1
  watch(restoreState, ({ paths, activeIndex }) => {
    if (workflowPersistenceEnabled.value) {
      setStorageValue('Comfy.OpenWorkflowsPaths', JSON.stringify(paths))
      setStorageValue('Comfy.ActiveWorkflowIndex', JSON.stringify(activeIndex))
    }
  })

  const restoreWorkflowTabsState = () => {
    if (!workflowPersistenceEnabled.value) return
    const isRestorable = storedWorkflows?.length > 0 && storedActiveIndex >= 0
    if (!isRestorable) return

    storedWorkflows.forEach((path: string) => {
      if (workflowStore.getWorkflowByPath(path)) return
      const draft = workflowDraftStore.getDraft(path)
      if (!draft?.isTemporary) return
      try {
        const workflowData = JSON.parse(draft.data)
        workflowStore.createTemporary(draft.name, workflowData)
      } catch (err) {
        console.warn(
          'Failed to parse workflow draft, creating with default',
          err
        )
        workflowDraftStore.removeDraft(path)
        workflowStore.createTemporary(draft.name)
      }
    })

    workflowStore.openWorkflowsInBackground({
      left: storedWorkflows.slice(0, storedActiveIndex),
      right: storedWorkflows.slice(storedActiveIndex)
    })
  }

  return {
    initializeWorkflow,
    loadTemplateFromUrlIfPresent,
    restoreWorkflowTabsState
  }
}
