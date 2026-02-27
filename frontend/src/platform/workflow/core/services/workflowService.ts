import { toRaw } from 'vue'

import { downloadBlob } from '@/base/common/downloadUtil'
import { t } from '@/i18n'
import { LGraph, LGraphCanvas } from '@/lib/litegraph/src/litegraph'
import type { Point, SerialisableGraph } from '@/lib/litegraph/src/litegraph'
import { useSettingStore } from '@/platform/settings/settingStore'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { useWorkflowDraftStore } from '@/platform/workflow/persistence/stores/workflowDraftStore'
import { syncLinearMode } from '@/platform/workflow/management/stores/comfyWorkflow'
import {
  ComfyWorkflow,
  useWorkflowStore
} from '@/platform/workflow/management/stores/workflowStore'
import { useTelemetry } from '@/platform/telemetry'
import type { ComfyWorkflowJSON } from '@/platform/workflow/validation/schemas/workflowSchema'
import { useWorkflowThumbnail } from '@/renderer/core/thumbnail/useWorkflowThumbnail'
import { app } from '@/scripts/app'
import { blankGraph, defaultGraph } from '@/scripts/defaultGraph'
import { useMissingModelsDialog } from '@/composables/useMissingModelsDialog'
import { useMissingNodesDialog } from '@/composables/useMissingNodesDialog'
import { useDialogService } from '@/services/dialogService'
import { useAppMode } from '@/composables/useAppMode'
import type { AppMode } from '@/composables/useAppMode'
import { useDomWidgetStore } from '@/stores/domWidgetStore'
import { useExecutionErrorStore } from '@/stores/executionErrorStore'
import { useWorkspaceStore } from '@/stores/workspaceStore'
import { appendJsonExt } from '@/utils/formatUtil'

function linearModeToAppMode(linearMode: unknown): AppMode | null {
  if (typeof linearMode !== 'boolean') return null
  return linearMode ? 'app' : 'graph'
}

export const useWorkflowService = () => {
  const settingStore = useSettingStore()
  const workflowStore = useWorkflowStore()
  const toastStore = useToastStore()
  const dialogService = useDialogService()
  const missingModelsDialog = useMissingModelsDialog()
  const missingNodesDialog = useMissingNodesDialog()
  const workflowThumbnail = useWorkflowThumbnail()
  const domWidgetStore = useDomWidgetStore()
  const executionErrorStore = useExecutionErrorStore()
  const workflowDraftStore = useWorkflowDraftStore()

  async function getFilename(defaultName: string): Promise<string | null> {
    if (settingStore.get('Comfy.PromptFilename')) {
      let filename = await dialogService.prompt({
        title: t('workflowService.exportWorkflow'),
        message: t('workflowService.enterFilenamePrompt'),
        defaultValue: defaultName
      })
      if (!filename) return null
      if (!filename.toLowerCase().endsWith('.json')) {
        filename += '.json'
      }
      return filename
    }
    return defaultName
  }

  /**
   * Adds scale and offset from litegraph canvas to the workflow JSON.
   * @param workflow The workflow to add the view restore data to
   */
  function addViewRestore(workflow: ComfyWorkflowJSON) {
    if (!settingStore.get('Comfy.EnableWorkflowViewRestore')) return

    const { offset, scale } = app.canvas.ds
    const [x, y] = offset

    workflow.extra ??= {}
    workflow.extra.ds = { scale, offset: [x, y] }
  }

  /**
   * Export the current workflow as a JSON file
   * @param filename The filename to save the workflow as
   * @param promptProperty The property of the prompt to export
   */
  const exportWorkflow = async (
    filename: string,
    promptProperty: 'workflow' | 'output'
  ): Promise<void> => {
    const workflow = workflowStore.activeWorkflow
    if (workflow?.path) {
      filename = workflow.filename
    }
    const p = await app.graphToPrompt()

    addViewRestore(p.workflow)
    const json = JSON.stringify(p[promptProperty], null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const file = await getFilename(filename)
    if (!file) return
    downloadBlob(file, blob)
  }
  /**
   * Save a workflow as a new file
   * @param workflow The workflow to save
   * @param options.filename Pre-supplied filename (skips the prompt dialog)
   */
  const saveWorkflowAs = async (
    workflow: ComfyWorkflow,
    options: { filename?: string; initialMode?: AppMode } = {}
  ): Promise<boolean> => {
    const newFilename = options.filename ?? (await workflow.promptSave())
    if (!newFilename) return false

    const newPath = workflow.directory + '/' + appendJsonExt(newFilename)
    const existingWorkflow = workflowStore.getWorkflowByPath(newPath)

    const isSelfOverwrite =
      existingWorkflow?.path === workflow.path && !existingWorkflow?.isTemporary

    if (existingWorkflow && !existingWorkflow.isTemporary) {
      const res = await dialogService.confirm({
        title: t('sideToolbar.workflowTab.confirmOverwriteTitle'),
        type: 'overwrite',
        message: t('sideToolbar.workflowTab.confirmOverwrite'),
        itemList: [newPath]
      })

      if (res !== true) return false

      if (!isSelfOverwrite) {
        const deleted = await deleteWorkflow(existingWorkflow, true)
        if (!deleted) return false
      }
    }

    if (options.initialMode) workflow.initialMode = options.initialMode

    syncLinearMode(workflow, [app.rootGraph], { flushLinearData: true })
    workflow.changeTracker?.checkState()

    if (isSelfOverwrite) {
      await saveWorkflow(workflow)
    } else if (workflow.isTemporary) {
      await renameWorkflow(workflow, newPath)
      await workflowStore.saveWorkflow(workflow)
    } else {
      const tempWorkflow = workflowStore.saveAs(workflow, newPath)
      await openWorkflow(tempWorkflow)
      await workflowStore.saveWorkflow(tempWorkflow)
    }
    return true
  }

  /**
   * Save a workflow
   * @param workflow The workflow to save
   */
  const saveWorkflow = async (workflow: ComfyWorkflow) => {
    if (workflow.isTemporary) {
      await saveWorkflowAs(workflow)
    } else {
      syncLinearMode(workflow, [app.rootGraph], { flushLinearData: true })
      workflow.changeTracker?.checkState()
      await workflowStore.saveWorkflow(workflow)
    }
  }

  /**
   * Load the default workflow
   */
  const loadDefaultWorkflow = async () => {
    await app.loadGraphData(defaultGraph)
  }

  /**
   * Load a blank workflow
   */
  const loadBlankWorkflow = async () => {
    await app.loadGraphData(blankGraph)
  }

  /**
   * Reload the current workflow
   * This is used to refresh the node definitions update, e.g. when the locale changes.
   */
  const reloadCurrentWorkflow = async () => {
    const workflow = workflowStore.activeWorkflow
    if (workflow) {
      await openWorkflow(workflow, { force: true })
    }
  }

  /**
   * Open a workflow in the current workspace
   * @param workflow The workflow to open
   * @param options The options for opening the workflow
   */
  const openWorkflow = async (
    workflow: ComfyWorkflow,
    options: { force: boolean } = { force: false }
  ) => {
    if (workflowStore.isActive(workflow) && !options.force) return

    const loadFromRemote = !workflow.isLoaded
    if (loadFromRemote) {
      await workflow.load()
    }

    await app.loadGraphData(
      toRaw(workflow.activeState) as ComfyWorkflowJSON,
      /* clean=*/ true,
      /* restore_view=*/ true,
      workflow,
      {
        showMissingModelsDialog: loadFromRemote,
        showMissingNodesDialog: loadFromRemote,
        checkForRerouteMigration: false,
        deferWarnings: true
      }
    )
    showPendingWarnings()
  }

  /**
   * Close a workflow with confirmation if there are unsaved changes
   * @param workflow The workflow to close
   * @returns true if the workflow was closed, false if the user cancelled
   */
  const closeWorkflow = async (
    workflow: ComfyWorkflow,
    options: { warnIfUnsaved: boolean; hint?: string } = {
      warnIfUnsaved: true
    }
  ): Promise<boolean> => {
    if (workflow.isModified && options.warnIfUnsaved) {
      const confirmed = await dialogService.confirm({
        title: t('sideToolbar.workflowTab.dirtyCloseTitle'),
        type: 'dirtyClose',
        message: t('sideToolbar.workflowTab.dirtyClose'),
        itemList: [workflow.path],
        hint: options.hint
      })
      // Cancel
      if (confirmed === null) return false

      if (confirmed === true) {
        await saveWorkflow(workflow)
      }
    }

    workflowDraftStore.removeDraft(workflow.path)

    // If this is the last workflow, create a new default temporary workflow
    if (workflowStore.openWorkflows.length === 1) {
      await loadDefaultWorkflow()
    }
    // If this is the active workflow, load the most recent workflow from history
    if (workflowStore.isActive(workflow)) {
      const mostRecentWorkflow = workflowStore.getMostRecentWorkflow()
      if (mostRecentWorkflow) {
        await openWorkflow(mostRecentWorkflow)
      } else {
        // Fallback to next workflow if no history
        await loadNextOpenedWorkflow()
      }
    }

    await workflowStore.closeWorkflow(workflow)
    return true
  }

  const renameWorkflow = async (workflow: ComfyWorkflow, newPath: string) => {
    await workflowStore.renameWorkflow(workflow, newPath)
  }

  /**
   * Delete a workflow
   * @param workflow The workflow to delete
   * @returns `true` if the workflow was deleted, `false` if the user cancelled
   */
  const deleteWorkflow = async (
    workflow: ComfyWorkflow,
    silent = false
  ): Promise<boolean> => {
    const bypassConfirm = !settingStore.get('Comfy.Workflow.ConfirmDelete')
    let confirmed: boolean | null = bypassConfirm || silent

    if (!confirmed) {
      confirmed = await dialogService.confirm({
        title: t('sideToolbar.workflowTab.confirmDeleteTitle'),
        type: 'delete',
        message: t('sideToolbar.workflowTab.confirmDelete'),
        itemList: [workflow.path]
      })
      if (!confirmed) return false
    }

    if (workflowStore.isOpen(workflow)) {
      const closed = await closeWorkflow(workflow, {
        warnIfUnsaved: !confirmed
      })
      if (!closed) return false
    }
    await workflowStore.deleteWorkflow(workflow)
    if (!silent) {
      toastStore.add({
        severity: 'info',
        summary: t('sideToolbar.workflowTab.deleted'),
        life: 1000
      })
    }
    return true
  }

  /**
   * This method is called before loading a new graph.
   * There are 3 major functions that loads a new graph to the graph editor:
   * 1. loadGraphData
   * 2. loadApiJson
   * 3. importA1111
   *
   * This function is used to save the current workflow states before loading
   * a new graph.
   */
  const beforeLoadNewGraph = () => {
    // Use workspaceStore here as it is patched in unit tests.
    const workflowStore = useWorkspaceStore().workflow
    const activeWorkflow = workflowStore.activeWorkflow
    if (activeWorkflow) {
      activeWorkflow.changeTracker.store()
      if (settingStore.get('Comfy.Workflow.Persist') && activeWorkflow.path) {
        const activeState = activeWorkflow.activeState
        if (activeState) {
          try {
            const workflowJson = JSON.stringify(activeState)
            workflowDraftStore.saveDraft(activeWorkflow.path, {
              data: workflowJson,
              updatedAt: Date.now(),
              name: activeWorkflow.key,
              isTemporary: activeWorkflow.isTemporary
            })
          } catch {
            toastStore.add({
              severity: 'error',
              summary: t('g.error'),
              detail: t('toastMessages.failedToSaveDraft'),
              life: 3000
            })
          }
        }
      }
      // Capture thumbnail before loading new graph
      void workflowThumbnail.storeThumbnail(activeWorkflow)
      domWidgetStore.clear()
    }
  }

  /**
   * Set the active workflow after the new graph is loaded.
   *
   * The call relationship is
   * useWorkflowService().openWorkflow -> app.loadGraphData -> useWorkflowService().afterLoadNewGraph
   * app.loadApiJson -> useWorkflowService().afterLoadNewGraph
   * app.importA1111 -> useWorkflowService().afterLoadNewGraph
   *
   * @param value The value to set as the active workflow.
   * @param workflowData The initial workflow data loaded to the graph editor.
   */
  const afterLoadNewGraph = async (
    value: string | ComfyWorkflow | null,
    workflowData: ComfyWorkflowJSON
  ) => {
    const workflowStore = useWorkspaceStore().workflow
    const { isAppMode } = useAppMode()
    const wasAppMode = isAppMode.value

    // Determine the initial app mode for fresh loads from serialized state.
    // null means linearMode was never explicitly set (not builder-saved).
    const freshLoadMode = linearModeToAppMode(workflowData.extra?.linearMode)

    function trackIfEnteringApp(workflow: ComfyWorkflow) {
      if (!wasAppMode && workflow.initialMode === 'app') {
        useTelemetry()?.trackEnterLinear({ source: 'workflow' })
      }
    }

    if (value === null || typeof value === 'string') {
      const path = value as string | null

      // Check if a persisted workflow with this path exists
      if (path) {
        const fullPath = ComfyWorkflow.basePath + appendJsonExt(path)
        const existingWorkflow = workflowStore.getWorkflowByPath(fullPath)

        // If the workflow exists and is NOT loaded yet (restoration case),
        // use the existing workflow instead of creating a new one.
        // If it IS loaded, this is a re-import case - create new with suffix.
        if (existingWorkflow?.isPersisted && !existingWorkflow.isLoaded) {
          const loadedWorkflow =
            await workflowStore.openWorkflow(existingWorkflow)
          if (loadedWorkflow.initialMode === undefined) {
            // Prefer the file's linearMode over the draft's since the file
            // is the authoritative saved state.
            loadedWorkflow.initialMode =
              linearModeToAppMode(
                loadedWorkflow.initialState?.extra?.linearMode
              ) ?? freshLoadMode
            trackIfEnteringApp(loadedWorkflow)
          }
          syncLinearMode(loadedWorkflow, [workflowData, app.rootGraph])
          loadedWorkflow.changeTracker.reset(workflowData)
          loadedWorkflow.changeTracker.restore()
          return
        }
      }

      const tempWorkflow = workflowStore.createNewTemporary(
        path ? appendJsonExt(path) : undefined,
        workflowData
      )
      tempWorkflow.initialMode = freshLoadMode
      trackIfEnteringApp(tempWorkflow)
      syncLinearMode(tempWorkflow, [workflowData, app.rootGraph])
      await workflowStore.openWorkflow(tempWorkflow)
      return
    }

    const loadedWorkflow = await workflowStore.openWorkflow(value)
    if (loadedWorkflow.initialMode === undefined) {
      loadedWorkflow.initialMode = freshLoadMode
      trackIfEnteringApp(loadedWorkflow)
    }
    syncLinearMode(loadedWorkflow, [workflowData, app.rootGraph])
    loadedWorkflow.changeTracker.reset(workflowData)
    loadedWorkflow.changeTracker.restore()
  }

  /**
   * Insert the given workflow into the current graph editor.
   */
  const insertWorkflow = async (
    workflow: ComfyWorkflow,
    options: { position?: Point } = {}
  ) => {
    const loadedWorkflow = await workflow.load()
    const workflowJSON = toRaw(loadedWorkflow.initialState)
    const old = localStorage.getItem('litegrapheditor_clipboard')
    // unknown conversion: ComfyWorkflowJSON is stricter than LiteGraph's
    // serialisation schema.
    const graph = new LGraph(workflowJSON as unknown as SerialisableGraph)
    const canvasElement = document.createElement('canvas')
    const canvas = new LGraphCanvas(canvasElement, graph, {
      skip_events: true,
      skip_render: true
    })
    canvas.selectItems()
    canvas.copyToClipboard()
    app.canvas.pasteFromClipboard(options)
    if (old !== null) {
      localStorage.setItem('litegrapheditor_clipboard', old)
    }
  }

  const loadNextOpenedWorkflow = async () => {
    const nextWorkflow = workflowStore.openedWorkflowIndexShift(1)
    if (nextWorkflow) {
      await openWorkflow(nextWorkflow)
    }
  }

  const loadPreviousOpenedWorkflow = async () => {
    const previousWorkflow = workflowStore.openedWorkflowIndexShift(-1)
    if (previousWorkflow) {
      await openWorkflow(previousWorkflow)
    }
  }

  /**
   * Takes an existing workflow and duplicates it with a new name
   */
  const duplicateWorkflow = async (workflow: ComfyWorkflow) => {
    const state = JSON.parse(JSON.stringify(workflow.activeState))
    const suffix = workflow.isPersisted ? ' (Copy)' : ''
    // Remove the suffix `(2)` or similar
    const filename = workflow.filename.replace(/\s*\(\d+\)$/, '') + suffix

    await app.loadGraphData(state, true, true, filename)
  }

  /**
   * Show and clear any pending warnings (missing nodes/models) stored on the
   * active workflow. Called after a workflow becomes visible so dialogs don't
   * overlap with subsequent loads.
   */
  function showPendingWarnings(workflow?: ComfyWorkflow | null) {
    const wf = workflow ?? workflowStore.activeWorkflow
    if (!wf?.pendingWarnings) return

    const { missingNodeTypes, missingModels } = wf.pendingWarnings
    wf.pendingWarnings = null

    if (missingNodeTypes?.length) {
      // Remove modal once Node Replacement is implemented in TabErrors.
      if (settingStore.get('Comfy.Workflow.ShowMissingNodesWarning')) {
        missingNodesDialog.show({ missingNodeTypes })
      }

      executionErrorStore.surfaceMissingNodes(missingNodeTypes)
    }

    if (
      missingModels &&
      settingStore.get('Comfy.Workflow.ShowMissingModelsWarning')
    ) {
      missingModelsDialog.show(missingModels)
    }
  }

  return {
    exportWorkflow,
    saveWorkflowAs,
    saveWorkflow,
    loadDefaultWorkflow,
    loadBlankWorkflow,
    reloadCurrentWorkflow,
    openWorkflow,
    closeWorkflow,
    renameWorkflow,
    deleteWorkflow,
    insertWorkflow,
    loadNextOpenedWorkflow,
    loadPreviousOpenedWorkflow,
    duplicateWorkflow,
    showPendingWarnings,
    afterLoadNewGraph,
    beforeLoadNewGraph
  }
}
