import { z } from 'zod'

import { LinkMarkerShape } from '@/lib/litegraph/src/litegraph'
import { zNodeId } from '@/platform/workflow/validation/schemas/workflowSchema'
import { colorPalettesSchema } from '@/schemas/colorPaletteSchema'
import { zKeybinding } from '@/platform/keybindings/types'
import { NodeBadgeMode } from '@/types/nodeSource'
import { LinkReleaseTriggerAction } from '@/types/searchBoxTypes'

const zNodeType = z.string()
const zJobId = z.string()
export type JobId = z.infer<typeof zJobId>
export const resultItemType = z.enum(['input', 'output', 'temp'])
export type ResultItemType = z.infer<typeof resultItemType>

const zCustomNodesI18n = z.record(z.string(), z.unknown())
export type CustomNodesI18n = z.infer<typeof zCustomNodesI18n>

const zResultItem = z.object({
  filename: z.string().optional(),
  subfolder: z.string().optional(),
  type: resultItemType.optional()
})
export type ResultItem = z.infer<typeof zResultItem>
const zOutputs = z
  .object({
    audio: z.array(zResultItem).optional(),
    images: z.array(zResultItem).optional(),
    video: z.array(zResultItem).optional(),
    animated: z.array(z.boolean()).optional(),
    text: z.union([z.string(), z.array(z.string())]).optional()
  })
  .passthrough()

export type NodeExecutionOutput = z.infer<typeof zOutputs>

export type NodeOutputWith<T extends Record<string, unknown>> =
  NodeExecutionOutput & T

// WS messages
const zStatusWsMessageStatus = z.object({
  exec_info: z.object({
    queue_remaining: z.number().int()
  })
})

const zStatusWsMessage = z.object({
  status: zStatusWsMessageStatus.nullish(),
  sid: z.string().nullish()
})

const zProgressWsMessage = z.object({
  value: z.number().int(),
  max: z.number().int(),
  prompt_id: zJobId,
  node: zNodeId
})

const zNodeProgressState = z.object({
  value: z.number(),
  max: z.number(),
  state: z.enum(['pending', 'running', 'finished', 'error']),
  node_id: zNodeId,
  prompt_id: zJobId,
  display_node_id: zNodeId.optional(),
  parent_node_id: zNodeId.optional(),
  real_node_id: zNodeId.optional()
})

const zProgressStateWsMessage = z.object({
  prompt_id: zJobId,
  nodes: z.record(zNodeId, zNodeProgressState)
})

const zExecutingWsMessage = z.object({
  node: zNodeId,
  display_node: zNodeId,
  prompt_id: zJobId
})

const zExecutedWsMessage = zExecutingWsMessage.extend({
  output: zOutputs,
  merge: z.boolean().optional()
})

const zExecutionWsMessageBase = z.object({
  prompt_id: zJobId,
  timestamp: z.number().int()
})

const zExecutionStartWsMessage = zExecutionWsMessageBase
const zExecutionSuccessWsMessage = zExecutionWsMessageBase
const zExecutionCachedWsMessage = zExecutionWsMessageBase.extend({
  nodes: z.array(zNodeId)
})
const zExecutionInterruptedWsMessage = zExecutionWsMessageBase.extend({
  node_id: zNodeId,
  node_type: zNodeType,
  executed: z.array(zNodeId)
})
const zExecutionErrorWsMessage = zExecutionWsMessageBase.extend({
  node_id: zNodeId,
  node_type: zNodeType,
  executed: z.array(zNodeId),
  exception_message: z.string(),
  exception_type: z.string(),
  traceback: z.array(z.string()),
  current_inputs: z.any(),
  current_outputs: z.any()
})

const zProgressTextWsMessage = z.object({
  nodeId: zNodeId,
  text: z.string()
})

const zNotificationWsMessage = z.object({
  value: z.string(),
  id: z.string().optional()
})
const zTerminalSize = z.object({
  cols: z.number(),
  row: z.number()
})
const zLogEntry = z.object({
  t: z.string(),
  m: z.string()
})
const zLogsWsMessage = z.object({
  size: zTerminalSize.optional(),
  entries: z.array(zLogEntry)
})
const zLogRawResponse = z.object({
  size: zTerminalSize,
  entries: z.array(zLogEntry)
})

const zFeatureFlagsWsMessage = z.record(z.string(), z.any())

const zAssetDownloadWsMessage = z.object({
  task_id: z.string(),
  asset_name: z.string(),
  bytes_total: z.number(),
  bytes_downloaded: z.number(),
  progress: z.number(),
  status: z.enum(['created', 'running', 'completed', 'failed']),
  asset_id: z.string().optional(),
  error: z.string().optional()
})

const zAssetExportWsMessage = z.object({
  task_id: z.string(),
  export_name: z.string().optional(),
  assets_total: z.number(),
  assets_attempted: z.number(),
  assets_failed: z.number(),
  bytes_total: z.number(),
  bytes_processed: z.number(),
  progress: z.number(),
  status: z.enum(['created', 'running', 'completed', 'failed']),
  error: z.string().optional()
})

export type StatusWsMessageStatus = z.infer<typeof zStatusWsMessageStatus>
export type StatusWsMessage = z.infer<typeof zStatusWsMessage>
export type ProgressWsMessage = z.infer<typeof zProgressWsMessage>
export type ExecutingWsMessage = z.infer<typeof zExecutingWsMessage>
export type ExecutedWsMessage = z.infer<typeof zExecutedWsMessage>
export type ExecutionStartWsMessage = z.infer<typeof zExecutionStartWsMessage>
export type ExecutionSuccessWsMessage = z.infer<
  typeof zExecutionSuccessWsMessage
>
export type ExecutionCachedWsMessage = z.infer<typeof zExecutionCachedWsMessage>
export type ExecutionInterruptedWsMessage = z.infer<
  typeof zExecutionInterruptedWsMessage
>
export type ExecutionErrorWsMessage = z.infer<typeof zExecutionErrorWsMessage>
export type LogsWsMessage = z.infer<typeof zLogsWsMessage>
export type ProgressTextWsMessage = z.infer<typeof zProgressTextWsMessage>
export type NodeProgressState = z.infer<typeof zNodeProgressState>
export type ProgressStateWsMessage = z.infer<typeof zProgressStateWsMessage>
export type FeatureFlagsWsMessage = z.infer<typeof zFeatureFlagsWsMessage>
export type AssetDownloadWsMessage = z.infer<typeof zAssetDownloadWsMessage>
export type AssetExportWsMessage = z.infer<typeof zAssetExportWsMessage>
// End of ws messages

export type NotificationWsMessage = z.infer<typeof zNotificationWsMessage>

export const zTaskOutput = z.record(zNodeId, zOutputs)
export type TaskOutput = z.infer<typeof zTaskOutput>

const zEmbeddingsResponse = z.array(z.string())
const zExtensionsResponse = z.array(z.string())
const zError = z.object({
  type: z.string(),
  message: z.string(),
  details: z.string(),
  extra_info: z
    .object({
      input_name: z.string().optional()
    })
    .passthrough()
    .optional()
})
const zNodeError = z.object({
  errors: z.array(zError),
  class_type: z.string(),
  dependent_outputs: z.array(z.any())
})
const zPromptResponse = z.object({
  node_errors: z.record(zNodeId, zNodeError).optional(),
  prompt_id: z.string().optional(),
  exec_info: z
    .object({
      queue_remaining: z.number().optional()
    })
    .optional(),
  error: z.union([z.string(), zError])
})

const zPromptError = z.object({
  type: z.string(),
  message: z.string(),
  details: z.string()
})

const zDeviceStats = z.object({
  name: z.string(),
  type: z.string(),
  index: z.number(),
  vram_total: z.number(),
  vram_free: z.number(),
  torch_vram_total: z.number(),
  torch_vram_free: z.number()
})

const zSystemStats = z.object({
  system: z.object({
    os: z.string(),
    python_version: z.string(),
    embedded_python: z.boolean(),
    comfyui_version: z.string(),
    pytorch_version: z.string(),
    required_frontend_version: z.string().optional(),
    argv: z.array(z.string()),
    ram_total: z.number(),
    ram_free: z.number(),
    // Cloud-specific fields
    cloud_version: z.string().optional(),
    comfyui_frontend_version: z.string().optional(),
    workflow_templates_version: z.string().optional(),
    installed_templates_version: z.string().optional(),
    required_templates_version: z.string().optional()
  }),
  devices: z.array(zDeviceStats)
})
const zUser = z.object({
  storage: z.enum(['server']),
  // `migrated` is only available in single-user mode.
  migrated: z.boolean().optional(),
  // `users` is only available in multi-user server mode.
  users: z.record(z.string(), z.string()).optional()
})
const zUserData = z.array(z.array(z.string(), z.string()))
const zUserDataFullInfo = z.object({
  path: z.string(),
  size: z.number(),
  modified: z.number()
})
const zBookmarkCustomization = z.object({
  icon: z.string().optional(),
  color: z.string().optional()
})
export type BookmarkCustomization = z.infer<typeof zBookmarkCustomization>

const zLinkReleaseTriggerAction = z.enum(
  Object.values(LinkReleaseTriggerAction) as [string, ...string[]]
)

const zNodeBadgeMode = z.enum(
  Object.values(NodeBadgeMode) as [string, ...string[]]
)

const zPreviewMethod = z.enum([
  'default',
  'none',
  'auto',
  'latent2rgb',
  'taesd'
])
export type PreviewMethod = z.infer<typeof zPreviewMethod>

const zSettings = z.object({
  'Comfy.ColorPalette': z.string(),
  'Comfy.CustomColorPalettes': colorPalettesSchema,
  'Comfy.Canvas.BackgroundImage': z.string().optional(),
  'Comfy.ConfirmClear': z.boolean(),
  'Comfy.DevMode': z.boolean(),
  'Comfy.UI.TabBarLayout': z.enum(['Default', 'Integrated']),
  'Comfy.Workflow.ShowMissingNodesWarning': z.boolean(),
  'Comfy.Workflow.ShowMissingModelsWarning': z.boolean(),
  'Comfy.Workflow.WarnBlueprintOverwrite': z.boolean(),
  'Comfy.DisableFloatRounding': z.boolean(),
  'Comfy.DisableSliders': z.boolean(),
  'Comfy.DOMClippingEnabled': z.boolean(),
  'Comfy.EditAttention.Delta': z.number(),
  'Comfy.EnableTooltips': z.boolean(),
  'Comfy.EnableWorkflowViewRestore': z.boolean(),
  'Comfy.FloatRoundingPrecision': z.number(),
  'Comfy.Graph.CanvasInfo': z.boolean(),
  'Comfy.Graph.CanvasMenu': z.boolean(),
  'Comfy.Graph.CtrlShiftZoom': z.boolean(),
  'Comfy.Graph.DeduplicateSubgraphNodeIds': z.boolean(),
  'Comfy.Graph.LiveSelection': z.boolean(),
  'Comfy.Graph.LinkMarkers': z.nativeEnum(LinkMarkerShape),
  'Comfy.Graph.ZoomSpeed': z.number(),
  'Comfy.Group.DoubleClickTitleToEdit': z.boolean(),
  'Comfy.GroupSelectedNodes.Padding': z.number(),
  'Comfy.Locale': z.string(),
  'Comfy.NodeLibrary.NewDesign': z.boolean(),
  'Comfy.NodeLibrary.Bookmarks': z.array(z.string()),
  'Comfy.NodeLibrary.Bookmarks.V2': z.array(z.string()),
  'Comfy.NodeLibrary.BookmarksCustomization': z.record(
    z.string(),
    zBookmarkCustomization
  ),
  'Comfy.LinkRelease.Action': zLinkReleaseTriggerAction,
  'Comfy.LinkRelease.ActionShift': zLinkReleaseTriggerAction,
  'Comfy.ModelLibrary.AutoLoadAll': z.boolean(),
  'Comfy.ModelLibrary.NameFormat': z.enum(['filename', 'title']),
  'Comfy.NodeSearchBoxImpl.NodePreview': z.boolean(),
  'Comfy.NodeSearchBoxImpl': z.enum([
    'default',
    'v1 (legacy)',
    'litegraph (legacy)'
  ]),
  'Comfy.NodeSearchBoxImpl.ShowCategory': z.boolean(),
  'Comfy.NodeSearchBoxImpl.ShowIdName': z.boolean(),
  'Comfy.NodeSearchBoxImpl.ShowNodeFrequency': z.boolean(),
  'Comfy.NodeSuggestions.number': z.number(),
  'Comfy.Node.BypassAllLinksOnDelete': z.boolean(),
  'Comfy.Node.Opacity': z.number(),
  'Comfy.Node.MiddleClickRerouteNode': z.boolean(),
  'Comfy.Node.ShowDeprecated': z.boolean(),
  'Comfy.Node.ShowExperimental': z.boolean(),
  'Comfy.NodeReplacement.Enabled': z.boolean(),
  'Comfy.Pointer.ClickBufferTime': z.number(),
  'Comfy.Pointer.ClickDrift': z.number(),
  'Comfy.Pointer.DoubleClickTime': z.number(),
  'Comfy.PreviewFormat': z.string(),
  'Comfy.PromptFilename': z.boolean(),
  'Comfy.Sidebar.Location': z.enum(['left', 'right']),
  'Comfy.Sidebar.Size': z.enum(['small', 'normal']),
  'Comfy.Sidebar.UnifiedWidth': z.boolean(),
  'Comfy.Sidebar.Style': z.enum(['floating', 'connected']),
  'Comfy.SnapToGrid.GridSize': z.number(),
  'Comfy.TextareaWidget.FontSize': z.number(),
  'Comfy.TextareaWidget.Spellcheck': z.boolean(),
  'Comfy.UseNewMenu': z.enum(['Disabled', 'Top']),
  'Comfy.TreeExplorer.ItemPadding': z.number(),
  'Comfy.Validation.Workflows': z.boolean(),
  'Comfy.Workflow.SortNodeIdOnSave': z.boolean(),
  'Comfy.Execution.PreviewMethod': zPreviewMethod,
  'Comfy.Workflow.WorkflowTabsPosition': z.enum(['Sidebar', 'Topbar']),
  'Comfy.Node.DoubleClickTitleToEdit': z.boolean(),
  'Comfy.WidgetControlMode': z.enum(['before', 'after']),
  'Comfy.Window.UnloadConfirmation': z.boolean(),
  'Comfy.NodeBadge.NodeSourceBadgeMode': zNodeBadgeMode,
  'Comfy.NodeBadge.NodeIdBadgeMode': zNodeBadgeMode,
  'Comfy.NodeBadge.NodeLifeCycleBadgeMode': zNodeBadgeMode,
  'Comfy.NodeBadge.ShowApiPricing': z.boolean(),
  'Comfy.Notification.ShowVersionUpdates': z.boolean(),
  'Comfy.QueueButton.BatchCountLimit': z.number(),
  'Comfy.Queue.MaxHistoryItems': z.number(),
  'Comfy.Queue.History.Expanded': z.boolean(),
  'Comfy.Keybinding.UnsetBindings': z.array(zKeybinding),
  'Comfy.Keybinding.NewBindings': z.array(zKeybinding),
  'Comfy.Extension.Disabled': z.array(z.string()),
  'Comfy.LinkRenderMode': z.number(),
  'Comfy.Node.AutoSnapLinkToSlot': z.boolean(),
  'Comfy.Node.SnapHighlightsNode': z.boolean(),
  'Comfy.Server.ServerConfigValues': z.record(z.string(), z.any()),
  'Comfy.Server.LaunchArgs': z.record(z.string(), z.string()),
  'LiteGraph.Canvas.MaximumFps': z.number(),
  'Comfy.Workflow.ConfirmDelete': z.boolean(),
  'Comfy.Workflow.AutoSaveDelay': z.number(),
  'Comfy.Workflow.AutoSave': z.enum(['off', 'after delay']),
  'Comfy.RerouteBeta': z.boolean(),
  'LiteGraph.Canvas.MinFontSizeForLOD': z.number(),
  'Comfy.Canvas.SelectionToolbox': z.boolean(),
  'LiteGraph.Node.TooltipDelay': z.number(),
  'LiteGraph.ContextMenu.Scaling': z.boolean(),
  'LiteGraph.Reroute.SplineOffset': z.number(),
  'LiteGraph.Canvas.LowQualityRenderingZoomThreshold': z.number(),
  'Comfy.Toast.DisableReconnectingToast': z.boolean(),
  'Comfy.Workflow.Persist': z.boolean(),
  'Comfy.TutorialCompleted': z.boolean(),
  'Comfy.InstalledVersion': z.string().nullable(),
  'Comfy.Node.AllowImageSizeDraw': z.boolean(),
  'Comfy.Minimap.Visible': z.boolean(),
  'Comfy.Minimap.NodeColors': z.boolean(),
  'Comfy.Minimap.ShowLinks': z.boolean(),
  'Comfy.Minimap.ShowGroups': z.boolean(),
  'Comfy.Minimap.RenderBypassState': z.boolean(),
  'Comfy.Minimap.RenderErrorState': z.boolean(),
  'Comfy.Canvas.NavigationMode': z.string(),
  'Comfy.Canvas.LeftMouseClickBehavior': z.string(),
  'Comfy.Canvas.MouseWheelScroll': z.string(),
  'Comfy.VueNodes.Enabled': z.boolean(),
  'Comfy.VueNodes.AutoScaleLayout': z.boolean(),
  'Comfy.Assets.UseAssetAPI': z.boolean(),
  'Comfy.Queue.QPOV2': z.boolean(),
  'Comfy-Desktop.AutoUpdate': z.boolean(),
  'Comfy-Desktop.SendStatistics': z.boolean(),
  'Comfy-Desktop.WindowStyle': z.string(),
  'Comfy-Desktop.UV.PythonInstallMirror': z.string(),
  'Comfy-Desktop.UV.PypiInstallMirror': z.string(),
  'Comfy-Desktop.UV.TorchInstallMirror': z.string(),
  'Comfy.MaskEditor.BrushAdjustmentSpeed': z.number(),
  'Comfy.MaskEditor.UseDominantAxis': z.boolean(),
  'Comfy.Load3D.ShowGrid': z.boolean(),
  'Comfy.Load3D.BackgroundColor': z.string(),
  'Comfy.Load3D.LightIntensity': z.number(),
  'Comfy.Load3D.LightIntensityMaximum': z.number(),
  'Comfy.Load3D.LightIntensityMinimum': z.number(),
  'Comfy.Load3D.LightAdjustmentIncrement': z.number(),
  'Comfy.Load3D.CameraType': z.enum(['perspective', 'orthographic']),
  'Comfy.Load3D.3DViewerEnable': z.boolean(),
  'Comfy.Load3D.PLYEngine': z.enum(['threejs', 'fastply', 'sparkjs']),
  'Comfy.Memory.AllowManualUnload': z.boolean(),
  'pysssss.SnapToGrid': z.boolean(),
  /** VHS setting is used for queue video preview support. */
  'VHS.AdvancedPreviews': z.string(),
  /** Release data settings */
  'Comfy.Release.Version': z.string(),
  'Comfy.Release.Status': z.enum([
    'skipped',
    'changelog seen',
    "what's new seen"
  ]),
  'Comfy.Release.Timestamp': z.number(),
  /** Template library filter settings */
  'Comfy.Templates.SelectedModels': z.array(z.string()),
  'Comfy.Templates.SelectedUseCases': z.array(z.string()),
  'Comfy.Templates.SelectedRunsOn': z.array(z.string()),
  'Comfy.Templates.SortBy': z.enum([
    'default',
    'recommended',
    'popular',
    'alphabetical',
    'newest',
    'vram-low-to-high',
    'model-size-low-to-high'
  ]),
  /** Settings used for testing */
  'test.setting': z.any(),
  'main.sub.setting.name': z.any(),
  'single.setting': z.any(),
  'LiteGraph.Node.DefaultPadding': z.boolean(),
  'LiteGraph.Pointer.TrackpadGestures': z.boolean(),
  'Comfy.VersionCompatibility.DisableWarnings': z.boolean(),
  'Comfy.RightSidePanel.IsOpen': z.boolean(),
  'Comfy.RightSidePanel.ShowErrorsTab': z.boolean(),
  'Comfy.Node.AlwaysShowAdvancedWidgets': z.boolean()
})

export type EmbeddingsResponse = z.infer<typeof zEmbeddingsResponse>
export type ExtensionsResponse = z.infer<typeof zExtensionsResponse>
export type PromptResponse = z.infer<typeof zPromptResponse>
export type PromptError = z.infer<typeof zPromptError>
export type NodeError = z.infer<typeof zNodeError>
export type Settings = z.infer<typeof zSettings>
export type DeviceStats = z.infer<typeof zDeviceStats>
export type SystemStats = z.infer<typeof zSystemStats>
export type User = z.infer<typeof zUser>
export type UserData = z.infer<typeof zUserData>
export type UserDataFullInfo = z.infer<typeof zUserDataFullInfo>
export type TerminalSize = z.infer<typeof zTerminalSize>
export type LogEntry = z.infer<typeof zLogEntry>
export type LogsRawResponse = z.infer<typeof zLogRawResponse>
