import { useCurrentUser } from '@/composables/auth/useCurrentUser'
import { useFirebaseAuthActions } from '@/composables/auth/useFirebaseAuthActions'
import { useSelectedLiteGraphItems } from '@/composables/canvas/useSelectedLiteGraphItems'
import { useSubgraphOperations } from '@/composables/graph/useSubgraphOperations'
import { useExternalLink } from '@/composables/useExternalLink'
import { useModelSelectorDialog } from '@/composables/useModelSelectorDialog'
import {
  DEFAULT_DARK_COLOR_PALETTE,
  DEFAULT_LIGHT_COLOR_PALETTE
} from '@/constants/coreColorPalettes'

import { tryToggleWidgetPromotion } from '@/core/graph/subgraph/promotionUtils'
import { t } from '@/i18n'
import {
  LGraphEventMode,
  LGraphGroup,
  LGraphNode,
  LiteGraph
} from '@/lib/litegraph/src/litegraph'
import type { Point } from '@/lib/litegraph/src/litegraph'
import { useBillingContext } from '@/composables/billing/useBillingContext'
import { useAssetBrowserDialog } from '@/platform/assets/composables/useAssetBrowserDialog'
import { createModelNodeFromAsset } from '@/platform/assets/utils/createModelNodeFromAsset'
import { useSettingStore } from '@/platform/settings/settingStore'
import { buildSupportUrl } from '@/platform/support/config'
import { useTelemetry } from '@/platform/telemetry'
import type { ExecutionTriggerSource } from '@/platform/telemetry/types'
import { useToastStore } from '@/platform/updates/common/toastStore'
import { useWorkflowService } from '@/platform/workflow/core/services/workflowService'
import { useWorkflowStore } from '@/platform/workflow/management/stores/workflowStore'
import type { ComfyWorkflow } from '@/platform/workflow/management/stores/workflowStore'
import {
  useCanvasStore,
  useTitleEditorStore
} from '@/renderer/core/canvas/canvasStore'
import { api } from '@/scripts/api'
import { app } from '@/scripts/app'
import { useSettingsDialog } from '@/platform/settings/composables/useSettingsDialog'
import { useDialogService } from '@/services/dialogService'
import { useLitegraphService } from '@/services/litegraphService'
import type { ComfyCommand } from '@/stores/commandStore'
import { useExecutionStore } from '@/stores/executionStore'
import { useHelpCenterStore } from '@/stores/helpCenterStore'
import {
  useQueueSettingsStore,
  useQueueStore,
  useQueueUIStore
} from '@/stores/queueStore'
import { useSubgraphNavigationStore } from '@/stores/subgraphNavigationStore'
import { useSubgraphStore } from '@/stores/subgraphStore'
import { useBottomPanelStore } from '@/stores/workspace/bottomPanelStore'
import { useColorPaletteStore } from '@/stores/workspace/colorPaletteStore'
import { useRightSidePanelStore } from '@/stores/workspace/rightSidePanelStore'
import { useSearchBoxStore } from '@/stores/workspace/searchBoxStore'
import { useWorkspaceStore } from '@/stores/workspaceStore'
import {
  getAllNonIoNodesInSubgraph,
  getExecutionIdsForSelectedNodes
} from '@/utils/graphTraversalUtil'
import { filterOutputNodes } from '@/utils/nodeFilterUtil'
import {
  ManagerUIState,
  useManagerState
} from '@/workbench/extensions/manager/composables/useManagerState'
import { ManagerTab } from '@/workbench/extensions/manager/types/comfyManagerTypes'

import { useWorkflowTemplateSelectorDialog } from './useWorkflowTemplateSelectorDialog'

import { useMaskEditorStore } from '@/stores/maskEditorStore'
import { useDialogStore } from '@/stores/dialogStore'

const moveSelectedNodesVersionAdded = '1.22.2'
export function useCoreCommands(): ComfyCommand[] {
  const { isActiveSubscription, showSubscriptionDialog } = useBillingContext()
  const workflowService = useWorkflowService()
  const workflowStore = useWorkflowStore()
  const settingsDialog = useSettingsDialog()
  const dialogService = useDialogService()
  const colorPaletteStore = useColorPaletteStore()
  const firebaseAuthActions = useFirebaseAuthActions()
  const toastStore = useToastStore()
  const canvasStore = useCanvasStore()
  const executionStore = useExecutionStore()
  const telemetry = useTelemetry()
  const { staticUrls, buildDocsUrl } = useExternalLink()
  const settingStore = useSettingStore()

  const bottomPanelStore = useBottomPanelStore()

  const dialogStore = useDialogStore()
  const maskEditorStore = useMaskEditorStore()

  const { getSelectedNodes, toggleSelectedNodesMode } =
    useSelectedLiteGraphItems()
  const getTracker = () => workflowStore.activeWorkflow?.changeTracker

  function isQueuePanelV2Enabled() {
    return settingStore.get('Comfy.Queue.QPOV2')
  }

  async function toggleQueuePanelV2() {
    await settingStore.set('Comfy.Queue.QPOV2', !isQueuePanelV2Enabled())
  }

  const moveSelectedNodes = (
    positionUpdater: (pos: Point, gridSize: number) => Point
  ) => {
    const selectedNodes = getSelectedNodes()
    if (selectedNodes.length === 0) return

    const gridSize = useSettingStore().get('Comfy.SnapToGrid.GridSize')
    selectedNodes.forEach((node) => {
      node.pos = positionUpdater(node.pos, gridSize)
    })
    app.canvas.state.selectionChanged = true
    app.canvas.setDirty(true, true)
  }

  const commands = [
    {
      id: 'Comfy.NewBlankWorkflow',
      icon: 'pi pi-plus',
      label: 'New Blank Workflow',
      menubarLabel: 'New',
      category: 'essentials' as const,
      function: async () => {
        const previousWorkflowHadNodes = app.rootGraph._nodes.length > 0
        await workflowService.loadBlankWorkflow()
        telemetry?.trackWorkflowCreated({
          workflow_type: 'blank',
          previous_workflow_had_nodes: previousWorkflowHadNodes
        })
      }
    },
    {
      id: 'Comfy.OpenWorkflow',
      icon: 'pi pi-folder-open',
      label: 'Open Workflow',
      menubarLabel: 'Open',
      category: 'essentials' as const,
      function: () => {
        app.ui.loadFile()
      }
    },
    {
      id: 'Comfy.LoadDefaultWorkflow',
      icon: 'pi pi-code',
      label: 'Load Default Workflow',
      function: async () => {
        const previousWorkflowHadNodes = app.rootGraph._nodes.length > 0
        await workflowService.loadDefaultWorkflow()
        telemetry?.trackWorkflowCreated({
          workflow_type: 'default',
          previous_workflow_had_nodes: previousWorkflowHadNodes
        })
      }
    },
    {
      id: 'Comfy.SaveWorkflow',
      icon: 'pi pi-save',
      label: 'Save Workflow',
      menubarLabel: 'Save',
      category: 'essentials' as const,
      function: async () => {
        const workflow = useWorkflowStore().activeWorkflow as ComfyWorkflow
        if (!workflow) return

        await workflowService.saveWorkflow(workflow)
      }
    },
    {
      id: 'Comfy.PublishSubgraph',
      icon: 'pi pi-save',
      label: 'Publish Subgraph',
      menubarLabel: 'Publish',
      function: async (metadata?: Record<string, unknown>) => {
        const name = metadata?.name as string | undefined
        await useSubgraphStore().publishSubgraph(name)
      }
    },
    {
      id: 'Comfy.SaveWorkflowAs',
      icon: 'pi pi-save',
      label: 'Save Workflow As',
      menubarLabel: 'Save As',
      category: 'essentials' as const,
      function: async () => {
        const workflow = useWorkflowStore().activeWorkflow as ComfyWorkflow
        if (!workflow) return

        await workflowService.saveWorkflowAs(workflow)
      }
    },
    {
      id: 'Comfy.RenameWorkflow',
      icon: 'pi pi-pencil',
      label: 'Rename Workflow',
      menubarLabel: 'Rename',
      function: async () => {
        const workflow = workflowStore.activeWorkflow
        if (!workflow || !workflow.isPersisted) return

        const newName = await dialogService.prompt({
          title: t('g.rename'),
          message: t('workflowService.enterFilenamePrompt'),
          defaultValue: workflow.filename
        })
        if (!newName || newName === workflow.filename) return

        const newPath = workflow.directory + '/' + newName + '.json'
        await workflowService.renameWorkflow(workflow, newPath)
      }
    },
    {
      id: 'Comfy.ExportWorkflow',
      icon: 'pi pi-download',
      label: 'Export Workflow',
      menubarLabel: 'Export',
      category: 'essentials' as const,
      function: async () => {
        await workflowService.exportWorkflow('workflow', 'workflow')
      }
    },
    {
      id: 'Comfy.ExportWorkflowAPI',
      icon: 'pi pi-download',
      label: 'Export Workflow (API Format)',
      menubarLabel: 'Export (API)',
      function: async () => {
        await workflowService.exportWorkflow('workflow_api', 'output')
      }
    },
    {
      id: 'Comfy.Undo',
      icon: 'pi pi-undo',
      label: 'Undo',
      category: 'essentials' as const,
      function: async () => {
        // If Mask Editor is open, use its history instead of the graph
        if (dialogStore.isDialogOpen('global-mask-editor')) {
          maskEditorStore.canvasHistory.undo()
        } else {
          await getTracker()?.undo?.()
        }
      }
    },
    {
      id: 'Comfy.Redo',
      icon: 'pi pi-refresh',
      label: 'Redo',
      category: 'essentials' as const,
      function: async () => {
        if (dialogStore.isDialogOpen('global-mask-editor')) {
          maskEditorStore.canvasHistory.redo()
        } else {
          await getTracker()?.redo?.()
        }
      }
    },
    {
      id: 'Comfy.ClearWorkflow',
      icon: 'pi pi-trash',
      label: 'Clear Workflow',
      category: 'essentials' as const,
      function: () => {
        const settingStore = useSettingStore()
        if (
          !settingStore.get('Comfy.ConfirmClear') ||
          confirm('Clear workflow?')
        ) {
          app.clean()
          if (app.canvas.subgraph) {
            // `clear` is not implemented on subgraphs and the parent class's
            // (`LGraph`) `clear` breaks the subgraph structure. For subgraphs,
            // just clear the nodes but preserve input/output nodes and structure
            const subgraph = app.canvas.subgraph
            const nonIoNodes = getAllNonIoNodesInSubgraph(subgraph)
            nonIoNodes.forEach((node) => subgraph.remove(node))
          }
          api.dispatchCustomEvent('graphCleared')
        }
      }
    },
    {
      id: 'Comfy.Canvas.ResetView',
      icon: 'pi pi-expand',
      label: 'Reset View',
      function: () => {
        useLitegraphService().resetView()
      }
    },
    {
      id: 'Comfy.OpenClipspace',
      icon: 'pi pi-clipboard',
      label: 'Clipspace',
      function: () => {
        app.openClipspace()
      }
    },
    {
      id: 'Comfy.RefreshNodeDefinitions',
      icon: 'pi pi-refresh',
      label: 'Refresh Node Definitions',
      category: 'essentials' as const,
      function: async () => {
        await app.refreshComboInNodes()
      }
    },
    {
      id: 'Comfy.Interrupt',
      icon: 'pi pi-stop',
      label: 'Interrupt',
      category: 'essentials' as const,
      function: async () => {
        await api.interrupt(executionStore.activeJobId)
        toastStore.add({
          severity: 'info',
          summary: t('g.interrupted'),
          detail: t('toastMessages.interrupted'),
          life: 1000
        })
      }
    },
    {
      id: 'Comfy.ClearPendingTasks',
      icon: 'pi pi-stop',
      label: 'Clear Pending Tasks',
      category: 'essentials' as const,
      function: async () => {
        await useQueueStore().clear(['queue'])
        toastStore.add({
          severity: 'info',
          summary: t('g.confirmed'),
          detail: t('toastMessages.pendingTasksDeleted'),
          life: 3000
        })
      }
    },
    {
      id: 'Comfy.BrowseTemplates',
      icon: 'pi pi-folder-open',
      label: 'Browse Templates',
      function: () => {
        useWorkflowTemplateSelectorDialog().show()
      }
    },
    {
      id: 'Comfy.Canvas.ZoomIn',
      icon: 'pi pi-plus',
      label: 'Zoom In',
      category: 'view-controls' as const,
      function: () => {
        const ds = app.canvas.ds
        ds.changeScale(
          ds.scale * 1.1,
          ds.element ? [ds.element.width / 2, ds.element.height / 2] : undefined
        )
        app.canvas.setDirty(true, true)
      }
    },
    {
      id: 'Comfy.Canvas.ZoomOut',
      icon: 'pi pi-minus',
      label: 'Zoom Out',
      category: 'view-controls' as const,
      function: () => {
        const ds = app.canvas.ds
        ds.changeScale(
          ds.scale / 1.1,
          ds.element ? [ds.element.width / 2, ds.element.height / 2] : undefined
        )
        app.canvas.setDirty(true, true)
      }
    },
    {
      id: 'Experimental.ToggleVueNodes',
      label: () =>
        `Experimental: ${
          useSettingStore().get('Comfy.VueNodes.Enabled') ? 'Disable' : 'Enable'
        } Nodes 2.0`,
      function: async () => {
        const settingStore = useSettingStore()
        const current = settingStore.get('Comfy.VueNodes.Enabled') ?? false
        await settingStore.set('Comfy.VueNodes.Enabled', !current)
      }
    },
    {
      id: 'Comfy.Canvas.FitView',
      icon: 'pi pi-expand',
      label: 'Fit view to selected nodes',
      menubarLabel: 'Zoom to fit',
      category: 'view-controls' as const,
      function: () => {
        if (app.canvas.empty) {
          toastStore.add({
            severity: 'error',
            summary: t('toastMessages.emptyCanvas'),
            life: 3000
          })
          return
        }
        app.canvas.fitViewToSelectionAnimated()
      }
    },
    {
      id: 'Comfy.Canvas.ToggleLock',
      icon: 'pi pi-lock',
      label: 'Canvas Toggle Lock',
      category: 'view-controls' as const,
      function: () => {
        app.canvas.state.readOnly = !app.canvas.state.readOnly
      }
    },
    {
      id: 'Comfy.Canvas.Lock',
      icon: 'pi pi-lock',
      label: 'Lock Canvas',
      category: 'view-controls' as const,
      function: () => {
        app.canvas.state.readOnly = true
      }
    },
    {
      id: 'Comfy.Canvas.Unlock',
      icon: 'pi pi-lock-open',
      label: 'Unlock Canvas',
      function: () => {
        app.canvas.state.readOnly = false
      }
    },
    {
      id: 'Comfy.Canvas.ToggleLinkVisibility',
      icon: 'pi pi-eye',
      label: 'Canvas Toggle Link Visibility',
      menubarLabel: 'Node Links',
      versionAdded: '1.3.6',

      function: (() => {
        const settingStore = useSettingStore()
        let lastLinksRenderMode = LiteGraph.SPLINE_LINK

        return async () => {
          const currentMode = settingStore.get('Comfy.LinkRenderMode')

          if (currentMode === LiteGraph.HIDDEN_LINK) {
            // If links are hidden, restore the last positive value or default to spline mode
            await settingStore.set('Comfy.LinkRenderMode', lastLinksRenderMode)
          } else {
            // If links are visible, store the current mode and hide links
            lastLinksRenderMode = currentMode
            await settingStore.set(
              'Comfy.LinkRenderMode',
              LiteGraph.HIDDEN_LINK
            )
          }
        }
      })(),
      active: () =>
        useSettingStore().get('Comfy.LinkRenderMode') !== LiteGraph.HIDDEN_LINK
    },
    {
      id: 'Comfy.Canvas.ToggleMinimap',
      icon: 'pi pi-map',
      label: 'Canvas Toggle Minimap',
      menubarLabel: 'Minimap',
      versionAdded: '1.24.1',
      function: async () => {
        const settingStore = useSettingStore()
        await settingStore.set(
          'Comfy.Minimap.Visible',
          !settingStore.get('Comfy.Minimap.Visible')
        )
      },
      active: () => useSettingStore().get('Comfy.Minimap.Visible')
    },
    {
      id: 'Comfy.Queue.ToggleOverlay',
      icon: 'pi pi-history',
      label: () => t('queue.toggleJobHistory'),
      menubarLabel: () => t('queue.jobHistory'),
      versionAdded: '1.37.0',
      category: 'view-controls' as const,
      function: () => {
        useQueueUIStore().toggleOverlay()
      },
      active: () => useQueueUIStore().isOverlayExpanded
    },
    {
      id: 'Comfy.QueuePrompt',
      icon: 'pi pi-play',
      label: 'Queue Prompt',
      versionAdded: '1.3.7',
      category: 'essentials' as const,
      function: async (metadata?: {
        subscribe_to_run?: boolean
        trigger_source?: ExecutionTriggerSource
      }) => {
        useTelemetry()?.trackRunButton(metadata)
        if (!isActiveSubscription.value) {
          showSubscriptionDialog()
          return
        }

        const batchCount = useQueueSettingsStore().batchCount

        useTelemetry()?.trackWorkflowExecution()

        await app.queuePrompt(0, batchCount)
      }
    },
    {
      id: 'Comfy.QueuePromptFront',
      icon: 'pi pi-play',
      label: 'Queue Prompt (Front)',
      versionAdded: '1.3.7',
      category: 'essentials' as const,
      function: async (metadata?: {
        subscribe_to_run?: boolean
        trigger_source?: ExecutionTriggerSource
      }) => {
        useTelemetry()?.trackRunButton(metadata)
        if (!isActiveSubscription.value) {
          showSubscriptionDialog()
          return
        }

        const batchCount = useQueueSettingsStore().batchCount

        useTelemetry()?.trackWorkflowExecution()

        await app.queuePrompt(-1, batchCount)
      }
    },
    {
      id: 'Comfy.QueueSelectedOutputNodes',
      icon: 'pi pi-play',
      label: 'Queue Selected Output Nodes',
      versionAdded: '1.19.6',
      function: async (metadata?: {
        subscribe_to_run?: boolean
        trigger_source?: ExecutionTriggerSource
      }) => {
        useTelemetry()?.trackRunButton(metadata)
        if (!isActiveSubscription.value) {
          showSubscriptionDialog()
          return
        }

        const batchCount = useQueueSettingsStore().batchCount
        const selectedNodes = getSelectedNodes()
        const selectedOutputNodes = filterOutputNodes(selectedNodes)

        if (selectedOutputNodes.length === 0) {
          toastStore.add({
            severity: 'error',
            summary: t('toastMessages.nothingToQueue'),
            detail: t('toastMessages.pleaseSelectOutputNodes'),
            life: 3000
          })
          return
        }

        // Get execution IDs for all selected output nodes and their descendants
        const executionIds =
          getExecutionIdsForSelectedNodes(selectedOutputNodes)

        if (executionIds.length === 0) {
          toastStore.add({
            severity: 'error',
            summary: t('toastMessages.failedToQueue'),
            detail: t('toastMessages.failedExecutionPathResolution'),
            life: 3000
          })
          return
        }
        useTelemetry()?.trackWorkflowExecution()
        await app.queuePrompt(0, batchCount, executionIds)
      }
    },
    {
      id: 'Comfy.ShowSettingsDialog',
      icon: 'pi pi-cog',
      label: 'Show Settings Dialog',
      versionAdded: '1.3.7',
      category: 'view-controls' as const,
      function: () => {
        settingsDialog.show()
      }
    },
    {
      id: 'Comfy.Graph.GroupSelectedNodes',
      icon: 'pi pi-sitemap',
      label: 'Group Selected Nodes',
      versionAdded: '1.3.7',
      category: 'essentials' as const,
      function: () => {
        const { canvas } = app
        if (!canvas.selectedItems?.size) {
          toastStore.add({
            severity: 'error',
            summary: t('toastMessages.nothingToGroup'),
            detail: t('toastMessages.pleaseSelectNodesToGroup'),
            life: 3000
          })
          return
        }
        const group = new LGraphGroup()
        const padding = useSettingStore().get(
          'Comfy.GroupSelectedNodes.Padding'
        )
        group.resizeTo(canvas.selectedItems, padding)
        canvas.graph?.add(group)

        group.recomputeInsideNodes()

        useTitleEditorStore().titleEditorTarget = group
      }
    },
    {
      id: 'Workspace.NextOpenedWorkflow',
      icon: 'pi pi-step-forward',
      label: 'Next Opened Workflow',
      versionAdded: '1.3.9',
      function: async () => {
        await workflowService.loadNextOpenedWorkflow()
      }
    },
    {
      id: 'Workspace.PreviousOpenedWorkflow',
      icon: 'pi pi-step-backward',
      label: 'Previous Opened Workflow',
      versionAdded: '1.3.9',
      function: async () => {
        await workflowService.loadPreviousOpenedWorkflow()
      }
    },
    {
      id: 'Comfy.Canvas.ToggleSelectedNodes.Mute',
      icon: 'pi pi-volume-off',
      label: 'Mute/Unmute Selected Nodes',
      versionAdded: '1.3.11',
      category: 'essentials' as const,
      function: () => {
        toggleSelectedNodesMode(LGraphEventMode.NEVER)
        app.canvas.setDirty(true, true)
      }
    },
    {
      id: 'Comfy.Canvas.ToggleSelectedNodes.Bypass',
      icon: 'pi pi-shield',
      label: 'Bypass/Unbypass Selected Nodes',
      versionAdded: '1.3.11',
      category: 'essentials' as const,
      function: () => {
        toggleSelectedNodesMode(LGraphEventMode.BYPASS)
        app.canvas.setDirty(true, true)
      }
    },
    {
      id: 'Comfy.Canvas.ToggleSelectedNodes.Pin',
      icon: 'pi pi-pin',
      label: 'Pin/Unpin Selected Nodes',
      versionAdded: '1.3.11',
      category: 'essentials' as const,
      function: () => {
        getSelectedNodes().forEach((node) => {
          node.pin(!node.pinned)
        })
        app.canvas.setDirty(true, true)
      }
    },
    {
      id: 'Comfy.Canvas.ToggleSelected.Pin',
      icon: 'pi pi-pin',
      label: 'Pin/Unpin Selected Items',
      versionAdded: '1.3.33',
      function: () => {
        for (const item of app.canvas.selectedItems) {
          if (item instanceof LGraphNode || item instanceof LGraphGroup) {
            item.pin(!item.pinned)
          }
        }
        app.canvas.setDirty(true, true)
      }
    },
    {
      id: 'Comfy.Canvas.Resize',
      icon: 'pi pi-minus',
      label: 'Resize Selected Nodes',
      versionAdded: '',
      function: () => {
        getSelectedNodes().forEach((node) => {
          const optimalSize = node.computeSize()
          node.setSize([optimalSize[0], optimalSize[1]])
        })
        app.canvas.setDirty(true, true)
      }
    },
    {
      id: 'Comfy.Canvas.ToggleSelectedNodes.Collapse',
      icon: 'pi pi-minus',
      label: 'Collapse/Expand Selected Nodes',
      versionAdded: '1.3.11',
      function: () => {
        getSelectedNodes().forEach((node) => {
          node.collapse()
        })
        app.canvas.setDirty(true, true)
      }
    },
    {
      id: 'Comfy.ToggleTheme',
      icon: 'pi pi-moon',
      label: 'Toggle Theme (Dark/Light)',
      versionAdded: '1.3.12',
      function: (() => {
        let previousDarkTheme: string = DEFAULT_DARK_COLOR_PALETTE.id
        let previousLightTheme: string = DEFAULT_LIGHT_COLOR_PALETTE.id

        return async () => {
          const settingStore = useSettingStore()
          const theme = colorPaletteStore.completedActivePalette
          if (theme.light_theme) {
            previousLightTheme = theme.id
            await settingStore.set('Comfy.ColorPalette', previousDarkTheme)
          } else {
            previousDarkTheme = theme.id
            await settingStore.set('Comfy.ColorPalette', previousLightTheme)
          }
        }
      })()
    },
    {
      id: 'Workspace.ToggleBottomPanel',
      icon: 'pi pi-list',
      label: 'Toggle Bottom Panel',
      menubarLabel: 'Bottom Panel',
      versionAdded: '1.3.22',
      category: 'view-controls' as const,
      function: () => {
        bottomPanelStore.toggleBottomPanel()
      },
      active: () => bottomPanelStore.bottomPanelVisible
    },
    {
      id: 'Workspace.ToggleFocusMode',
      icon: 'pi pi-eye',
      label: 'Toggle Focus Mode',
      menubarLabel: 'Focus Mode',
      versionAdded: '1.3.27',
      category: 'view-controls' as const,
      function: () => {
        useWorkspaceStore().toggleFocusMode()
      },
      active: () => useWorkspaceStore().focusMode
    },
    {
      id: 'Comfy.Graph.FitGroupToContents',
      icon: 'pi pi-expand',
      label: 'Fit Group To Contents',
      versionAdded: '1.4.9',
      function: () => {
        for (const group of app.canvas.selectedItems) {
          if (group instanceof LGraphGroup) {
            group.recomputeInsideNodes()
            const padding = useSettingStore().get(
              'Comfy.GroupSelectedNodes.Padding'
            )
            group.resizeTo(group.children, padding)
            app.canvas.setDirty(false, true)
          }
        }
      }
    },
    {
      id: 'Comfy.Help.OpenComfyUIIssues',
      icon: 'pi pi-github',
      label: 'Open ComfyUI Issues',
      menubarLabel: 'ComfyUI Issues',
      versionAdded: '1.5.5',
      function: () => {
        telemetry?.trackHelpResourceClicked({
          resource_type: 'github',
          is_external: true,
          source: 'menu'
        })
        window.open(staticUrls.githubIssues, '_blank')
      }
    },
    {
      id: 'Comfy.Help.OpenComfyUIDocs',
      icon: 'pi pi-info-circle',
      label: 'Open ComfyUI Docs',
      menubarLabel: 'ComfyUI Docs',
      versionAdded: '1.5.5',
      function: () => {
        telemetry?.trackHelpResourceClicked({
          resource_type: 'docs',
          is_external: true,
          source: 'menu'
        })
        window.open(buildDocsUrl('/', { includeLocale: true }), '_blank')
      }
    },
    {
      id: 'Comfy.Help.OpenComfyOrgDiscord',
      icon: 'pi pi-discord',
      label: 'Open Comfy-Org Discord',
      menubarLabel: 'Comfy-Org Discord',
      versionAdded: '1.5.5',
      function: () => {
        telemetry?.trackHelpResourceClicked({
          resource_type: 'discord',
          is_external: true,
          source: 'menu'
        })
        window.open(staticUrls.discord, '_blank')
      }
    },
    {
      id: 'Workspace.SearchBox.Toggle',
      icon: 'pi pi-search',
      label: 'Toggle Search Box',
      versionAdded: '1.5.7',
      function: () => {
        useSearchBoxStore().toggleVisible()
      }
    },
    {
      id: 'Comfy.Help.AboutComfyUI',
      icon: 'pi pi-info-circle',
      label: 'Open About ComfyUI',
      menubarLabel: 'About ComfyUI',
      versionAdded: '1.6.4',
      function: () => {
        settingsDialog.showAbout()
      }
    },
    {
      id: 'Comfy.DuplicateWorkflow',
      icon: 'pi pi-clone',
      label: 'Duplicate Current Workflow',
      versionAdded: '1.6.15',
      function: async () => {
        await workflowService.duplicateWorkflow(workflowStore.activeWorkflow!)
      }
    },
    {
      id: 'Workspace.CloseWorkflow',
      icon: 'pi pi-times',
      label: 'Close Current Workflow',
      versionAdded: '1.7.3',
      function: async () => {
        if (workflowStore.activeWorkflow)
          await workflowService.closeWorkflow(workflowStore.activeWorkflow)
      }
    },
    {
      id: 'Comfy.ContactSupport',
      icon: 'pi pi-question',
      label: 'Contact Support',
      versionAdded: '1.17.8',
      function: () => {
        const { userEmail, resolvedUserInfo } = useCurrentUser()
        const supportUrl = buildSupportUrl({
          userEmail: userEmail.value,
          userId: resolvedUserInfo.value?.id
        })
        window.open(supportUrl, '_blank', 'noopener,noreferrer')
      }
    },
    {
      id: 'Comfy.Help.OpenComfyUIForum',
      icon: 'pi pi-comments',
      label: 'Open ComfyUI Forum',
      menubarLabel: 'ComfyUI Forum',
      versionAdded: '1.8.2',
      function: () => {
        telemetry?.trackHelpResourceClicked({
          resource_type: 'help_feedback',
          is_external: true,
          source: 'menu'
        })
        window.open(staticUrls.forum, '_blank')
      }
    },
    {
      id: 'Comfy.Canvas.CopySelected',
      icon: 'icon-[lucide--copy]',
      label: 'Copy',
      function: () => {
        if (app.canvas.selectedItems?.size) {
          app.canvas.copyToClipboard()
        }
      }
    },
    {
      id: 'Comfy.Canvas.PasteFromClipboard',
      icon: 'icon-[lucide--clipboard-paste]',
      label: 'Paste',
      function: () => {
        app.canvas.pasteFromClipboard()
      }
    },
    {
      id: 'Comfy.Canvas.SelectAll',
      icon: 'icon-[lucide--lasso-select]',
      label: 'Select All',
      function: () => {
        app.canvas.selectItems()
      }
    },
    {
      id: 'Comfy.Canvas.DeleteSelectedItems',
      icon: 'pi pi-trash',
      label: 'Delete Selected Items',
      versionAdded: '1.10.5',
      function: () => {
        app.canvas.deleteSelected()
        app.canvas.setDirty(true, true)
      }
    },
    {
      id: 'Comfy.Manager.CustomNodesManager.ShowCustomNodesMenu',
      icon: 'pi pi-puzzle',
      label: 'Custom Nodes Manager',
      versionAdded: '1.12.10',
      function: async () => {
        await useManagerState().openManager({
          showToastOnLegacyError: true
        })
      }
    },
    {
      id: 'Comfy.Manager.ShowUpdateAvailablePacks',
      icon: 'pi pi-sync',
      label: 'Check for Custom Node Updates',
      versionAdded: '1.17.0',
      function: async () => {
        const managerState = useManagerState()
        const state = managerState.managerUIState.value

        // For DISABLED state, show error toast instead of opening settings
        if (state === ManagerUIState.DISABLED) {
          toastStore.add({
            severity: 'error',
            summary: t('g.error'),
            detail: t('manager.notAvailable'),
            life: 3000
          })
          return
        }

        await managerState.openManager({
          initialTab: ManagerTab.UpdateAvailable,
          showToastOnLegacyError: false
        })
      }
    },
    {
      id: 'Comfy.Manager.ShowMissingPacks',
      icon: 'pi pi-exclamation-circle',
      label: 'Install Missing Custom Nodes',
      versionAdded: '1.17.0',
      function: async () => {
        await useManagerState().openManager({
          initialTab: ManagerTab.Missing,
          showToastOnLegacyError: false
        })
      }
    },
    {
      id: 'Comfy.User.OpenSignInDialog',
      icon: 'pi pi-user',
      label: 'Open Sign In Dialog',
      versionAdded: '1.17.6',
      function: async () => {
        await dialogService.showSignInDialog()
      }
    },
    {
      id: 'Comfy.User.SignOut',
      icon: 'pi pi-sign-out',
      label: 'Sign Out',
      versionAdded: '1.18.1',
      function: async () => {
        await firebaseAuthActions.logout()
      }
    },
    {
      id: 'Comfy.Canvas.MoveSelectedNodes.Up',
      icon: 'pi pi-arrow-up',
      label: 'Move Selected Nodes Up',
      versionAdded: moveSelectedNodesVersionAdded,
      function: () => moveSelectedNodes(([x, y], gridSize) => [x, y - gridSize])
    },
    {
      id: 'Comfy.Canvas.MoveSelectedNodes.Down',
      icon: 'pi pi-arrow-down',
      label: 'Move Selected Nodes Down',
      versionAdded: moveSelectedNodesVersionAdded,
      function: () => moveSelectedNodes(([x, y], gridSize) => [x, y + gridSize])
    },
    {
      id: 'Comfy.Canvas.MoveSelectedNodes.Left',
      icon: 'pi pi-arrow-left',
      label: 'Move Selected Nodes Left',
      versionAdded: moveSelectedNodesVersionAdded,
      function: () => moveSelectedNodes(([x, y], gridSize) => [x - gridSize, y])
    },
    {
      id: 'Comfy.Canvas.MoveSelectedNodes.Right',
      icon: 'pi pi-arrow-right',
      label: 'Move Selected Nodes Right',
      versionAdded: moveSelectedNodesVersionAdded,
      function: () => moveSelectedNodes(([x, y], gridSize) => [x + gridSize, y])
    },
    {
      id: 'Comfy.Graph.ConvertToSubgraph',
      icon: 'icon-[lucide--shrink]',
      label: 'Convert Selection to Subgraph',
      versionAdded: '1.20.1',
      category: 'essentials' as const,
      function: () => {
        const canvas = canvasStore.getCanvas()
        const graph = canvas.subgraph ?? canvas.graph
        if (!graph) throw new TypeError('Canvas has no graph or subgraph set.')

        const res = graph.convertToSubgraph(canvas.selectedItems)
        if (!res) {
          toastStore.add({
            severity: 'error',
            summary: t('toastMessages.cannotCreateSubgraph'),
            detail: t('toastMessages.failedToConvertToSubgraph'),
            life: 3000
          })
          return
        }

        const { node } = res
        canvas.select(node)
        canvasStore.updateSelectedItems()
      }
    },
    {
      id: 'Comfy.Graph.UnpackSubgraph',
      icon: 'icon-[lucide--expand]',
      label: 'Unpack the selected Subgraph',
      versionAdded: '1.26.3',
      function: () => {
        const { unpackSubgraph } = useSubgraphOperations()
        unpackSubgraph()
      }
    },
    {
      id: 'Comfy.Graph.EditSubgraphWidgets',
      label: 'Edit Subgraph Widgets',
      icon: 'icon-[lucide--settings-2]',
      versionAdded: '1.28.5',
      function: () => {
        useRightSidePanelStore().openPanel('subgraph')
      }
    },
    {
      id: 'Comfy.Graph.ToggleWidgetPromotion',
      icon: 'icon-[lucide--arrow-left-right]',
      label: 'Toggle promotion of hovered widget',
      versionAdded: '1.30.1',
      function: tryToggleWidgetPromotion
    },
    {
      id: 'Comfy.OpenManagerDialog',
      icon: 'mdi mdi-puzzle-outline',
      label: 'Manager',
      function: async () => {
        await useManagerState().openManager({
          initialTab: ManagerTab.All,
          showToastOnLegacyError: false
        })
      }
    },
    {
      id: 'Comfy.ToggleHelpCenter',
      icon: 'pi pi-question-circle',
      label: 'Help Center',
      function: () => {
        useHelpCenterStore().toggle()
      },
      active: () => useHelpCenterStore().isVisible
    },
    {
      id: 'Comfy.ToggleCanvasInfo',
      icon: 'pi pi-info-circle',
      label: 'Canvas Performance',
      function: async () => {
        const settingStore = useSettingStore()
        const currentValue = settingStore.get('Comfy.Graph.CanvasInfo')
        await settingStore.set('Comfy.Graph.CanvasInfo', !currentValue)
      },
      active: () => useSettingStore().get('Comfy.Graph.CanvasInfo')
    },
    {
      id: 'Workspace.ToggleBottomPanel.Shortcuts',
      icon: 'pi pi-key',
      label: 'Show Keybindings Dialog',
      versionAdded: '1.24.1',
      category: 'view-controls' as const,
      function: () => {
        bottomPanelStore.togglePanel('shortcuts')
      }
    },
    {
      id: 'Comfy.Graph.ExitSubgraph',
      icon: 'pi pi-arrow-up',
      label: 'Exit Subgraph',
      versionAdded: '1.20.1',
      function: () => {
        const canvas = useCanvasStore().getCanvas()
        const navigationStore = useSubgraphNavigationStore()
        if (!canvas.graph) return

        canvas.setGraph(
          navigationStore.navigationStack.at(-2) ?? canvas.graph.rootGraph
        )
      }
    },
    {
      id: 'Comfy.Subgraph.SetDescription',
      icon: 'pi pi-pencil',
      label: 'Set Subgraph Description',
      versionAdded: '1.39.7',
      function: async (metadata?: Record<string, unknown>) => {
        const canvas = canvasStore.getCanvas()
        const subgraph = canvas.subgraph
        if (!subgraph) return

        const extra = (subgraph.extra ??= {}) as Record<string, unknown>
        const currentDescription = (extra.BlueprintDescription as string) ?? ''

        let description: string | null | undefined
        const rawDescription = metadata?.description
        if (rawDescription != null) {
          description =
            typeof rawDescription === 'string'
              ? rawDescription
              : String(rawDescription)
        }
        description ??= await dialogService.prompt({
          title: t('g.description'),
          message: t('subgraphStore.enterDescription'),
          defaultValue: currentDescription
        })
        if (description === null) return

        extra.BlueprintDescription = description.trim() || undefined
        workflowStore.activeWorkflow?.changeTracker?.checkState()
      }
    },
    {
      id: 'Comfy.Subgraph.SetSearchAliases',
      icon: 'pi pi-search',
      label: 'Set Subgraph Search Aliases',
      versionAdded: '1.39.7',
      function: async (metadata?: Record<string, unknown>) => {
        const canvas = canvasStore.getCanvas()
        const subgraph = canvas.subgraph
        if (!subgraph) return

        const parseAliases = (value: unknown): string[] =>
          (Array.isArray(value) ? value.map(String) : String(value).split(','))
            .map((s) => s.trim())
            .filter(Boolean)

        const extra = (subgraph.extra ??= {}) as Record<string, unknown>

        let aliases: string[]
        const rawAliases = metadata?.aliases
        if (rawAliases == null) {
          const input = await dialogService.prompt({
            title: t('subgraphStore.searchAliases'),
            message: t('subgraphStore.enterSearchAliases'),
            defaultValue: parseAliases(extra.BlueprintSearchAliases ?? '').join(
              ', '
            )
          })
          if (input === null) return
          aliases = parseAliases(input)
        } else {
          aliases = parseAliases(rawAliases)
        }

        extra.BlueprintSearchAliases = aliases.length > 0 ? aliases : undefined
        workflowStore.activeWorkflow?.changeTracker?.checkState()
      }
    },
    {
      id: 'Comfy.Dev.ShowModelSelector',
      icon: 'pi pi-box',
      label: 'Show Model Selector (Dev)',
      versionAdded: '1.26.2',
      category: 'view-controls' as const,
      function: () => {
        const modelSelectorDialog = useModelSelectorDialog()
        modelSelectorDialog.show()
      }
    },
    {
      id: 'Comfy.Manager.CustomNodesManager.ShowLegacyCustomNodesMenu',
      icon: 'pi pi-bars',
      label: 'Custom Nodes (Legacy)',
      versionAdded: '1.16.4',
      function: async () => {
        await useManagerState().openManager({
          legacyCommand: 'Comfy.Manager.CustomNodesManager.ToggleVisibility',
          showToastOnLegacyError: true,
          isLegacyOnly: true
        })
      }
    },
    {
      id: 'Comfy.Manager.ShowLegacyManagerMenu',
      icon: 'mdi mdi-puzzle',
      label: 'Manager Menu (Legacy)',
      versionAdded: '1.16.4',
      function: async () => {
        await useManagerState().openManager({
          showToastOnLegacyError: true,
          isLegacyOnly: true
        })
      }
    },
    {
      id: 'Comfy.Memory.UnloadModels',
      icon: 'mdi mdi-vacuum-outline',
      label: 'Unload Models',
      versionAdded: '1.16.4',
      function: async () => {
        if (!useSettingStore().get('Comfy.Memory.AllowManualUnload')) {
          useToastStore().add({
            severity: 'error',
            summary: t('g.error'),
            detail: t('g.commandProhibited', {
              command: 'Comfy.Memory.UnloadModels'
            }),
            life: 3000
          })
          return
        }
        await api.freeMemory({ freeExecutionCache: false })
      }
    },
    {
      id: 'Comfy.Memory.UnloadModelsAndExecutionCache',
      icon: 'mdi mdi-vacuum-outline',
      label: 'Unload Models and Execution Cache',
      versionAdded: '1.16.4',
      function: async () => {
        if (!useSettingStore().get('Comfy.Memory.AllowManualUnload')) {
          useToastStore().add({
            severity: 'error',
            summary: t('g.error'),
            detail: t('g.commandProhibited', {
              command: 'Comfy.Memory.UnloadModelsAndExecutionCache'
            }),
            life: 3000
          })
          return
        }
        await api.freeMemory({ freeExecutionCache: true })
      }
    },
    {
      id: 'Comfy.BrowseModelAssets',
      icon: 'pi pi-folder-open',
      label: 'Experimental: Browse Model Assets',
      versionAdded: '1.28.3',
      function: async () => {
        if (!useSettingStore().get('Comfy.Assets.UseAssetAPI')) {
          const confirmed = await dialogService.confirm({
            title: 'Enable Asset API',
            message:
              'The Asset API is currently disabled. Would you like to enable it?',
            type: 'default'
          })

          if (!confirmed) return

          const settingStore = useSettingStore()
          await settingStore.set('Comfy.Assets.UseAssetAPI', true)
          await workflowService.reloadCurrentWorkflow()
        }
        const assetBrowserDialog = useAssetBrowserDialog()
        await assetBrowserDialog.browse({
          assetType: 'models',
          title: t('sideToolbar.modelLibrary'),
          onAssetSelected: (asset) => {
            const result = createModelNodeFromAsset(asset)
            if (!result.success) {
              toastStore.add({
                severity: 'error',
                summary: t('g.error'),
                detail: t('assetBrowser.failedToCreateNode')
              })
              console.error('Node creation failed:', result.error)
            }
          }
        })
      }
    },
    {
      id: 'Comfy.ToggleAssetAPI',
      icon: 'pi pi-database',
      label: () =>
        `Experimental: ${
          useSettingStore().get('Comfy.Assets.UseAssetAPI')
            ? 'Disable'
            : 'Enable'
        } AssetAPI`,
      function: async () => {
        const settingStore = useSettingStore()
        const current = settingStore.get('Comfy.Assets.UseAssetAPI') ?? false
        await settingStore.set('Comfy.Assets.UseAssetAPI', !current)
        await useWorkflowService().reloadCurrentWorkflow() // ensure changes take effect immediately
      }
    },
    {
      id: 'Comfy.ToggleQPOV2',
      icon: 'pi pi-list',
      label: 'Toggle Queue Panel V2',
      function: toggleQueuePanelV2
    },
    {
      id: 'Comfy.ToggleLinear',
      icon: 'pi pi-database',
      label: 'Toggle App Mode',
      function: (metadata?: Record<string, unknown>) => {
        const source =
          typeof metadata?.source === 'string' ? metadata.source : 'keybind'
        const newMode = !canvasStore.linearMode
        if (newMode) useTelemetry()?.trackEnterLinear({ source })
        canvasStore.linearMode = newMode
      }
    }
  ]

  return commands.map((command) => ({ ...command, source: 'System' }))
}
