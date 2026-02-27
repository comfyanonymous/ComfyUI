import { LinkMarkerShape, LiteGraph } from '@/lib/litegraph/src/litegraph'
import { isCloud } from '@/platform/distribution/types'
import { useSettingStore } from '@/platform/settings/settingStore'
import type { SettingParams } from '@/platform/settings/types'
import type { ColorPalettes } from '@/schemas/colorPaletteSchema'
import type { Keybinding } from '@/platform/keybindings/types'
import { NodeBadgeMode } from '@/types/nodeSource'
import { LinkReleaseTriggerAction } from '@/types/searchBoxTypes'
import { breakpointsTailwind } from '@vueuse/core'

/**
 * Core settings are essential configuration parameters required for ComfyUI's basic functionality.
 * These settings must be present in the settings store and cannot be omitted.
 *
 * IMPORTANT: To prevent ID conflicts, settings should be marked as deprecated rather than removed
 * when they are no longer needed.
 */
export const CORE_SETTINGS: SettingParams[] = [
  {
    id: 'Comfy.Memory.AllowManualUnload',
    name: 'Allow manual unload of models and execution cache via user command',
    type: 'hidden',
    defaultValue: isCloud ? false : true,
    versionAdded: '1.18.0'
  },
  {
    id: 'Comfy.Validation.Workflows',
    name: 'Validate workflows',
    type: 'boolean',
    defaultValue: false
  },
  {
    id: 'Comfy.NodeSearchBoxImpl',
    category: ['Comfy', 'Node Search Box', 'Implementation'],
    experimental: true,
    name: 'Node search box implementation',
    type: 'combo',
    options: ['default', 'v1 (legacy)', 'litegraph (legacy)'],
    defaultValue: 'default'
  },
  {
    id: 'Comfy.LinkRelease.Action',
    category: ['LiteGraph', 'LinkRelease', 'Action'],
    name: 'Action on link release (No modifier)',
    type: 'combo',
    options: Object.values(LinkReleaseTriggerAction),
    defaultValue: LinkReleaseTriggerAction.CONTEXT_MENU,
    defaultsByInstallVersion: {
      '1.24.1': LinkReleaseTriggerAction.SEARCH_BOX
    }
  },
  {
    id: 'Comfy.LinkRelease.ActionShift',
    category: ['LiteGraph', 'LinkRelease', 'ActionShift'],
    name: 'Action on link release (Shift)',
    type: 'combo',
    options: Object.values(LinkReleaseTriggerAction),
    defaultValue: LinkReleaseTriggerAction.SEARCH_BOX,
    defaultsByInstallVersion: {
      '1.24.1': LinkReleaseTriggerAction.CONTEXT_MENU
    }
  },
  {
    id: 'Comfy.NodeSearchBoxImpl.NodePreview',
    category: ['Comfy', 'Node Search Box', 'NodePreview'],
    name: 'Node preview',
    tooltip: 'Only applies to the default implementation',
    type: 'boolean',
    defaultValue: true
  },
  {
    id: 'Comfy.NodeSearchBoxImpl.ShowCategory',
    category: ['Comfy', 'Node Search Box', 'ShowCategory'],
    name: 'Show node category in search results',
    tooltip: 'Only applies to v1 (legacy)',
    type: 'boolean',
    defaultValue: true
  },
  {
    id: 'Comfy.NodeSearchBoxImpl.ShowIdName',
    category: ['Comfy', 'Node Search Box', 'ShowIdName'],
    name: 'Show node id name in search results',
    tooltip: 'Does not apply to litegraph (legacy)',
    type: 'boolean',
    defaultValue: false
  },
  {
    id: 'Comfy.NodeSearchBoxImpl.ShowNodeFrequency',
    category: ['Comfy', 'Node Search Box', 'ShowNodeFrequency'],
    name: 'Show node frequency in search results',
    tooltip: 'Only applies to v1 (legacy)',
    type: 'boolean',
    defaultValue: false
  },
  {
    id: 'Comfy.Sidebar.Location',
    category: ['Appearance', 'Sidebar', 'Location'],
    name: 'Sidebar location',
    type: 'combo',
    options: ['left', 'right'],
    defaultValue: 'left'
  },
  {
    id: 'Comfy.Sidebar.Size',
    category: ['Appearance', 'Sidebar', 'Size'],
    name: 'Sidebar size',
    type: 'combo',
    options: ['normal', 'small'],
    // Default to small if the window is less than 1536px(2xl) wide.
    defaultValue: () => (window.innerWidth < 1536 ? 'small' : 'normal')
  },
  {
    id: 'Comfy.Sidebar.UnifiedWidth',
    category: ['Appearance', 'Sidebar', 'UnifiedWidth'],
    name: 'Unified sidebar width',
    type: 'boolean',
    defaultValue: true,
    versionAdded: '1.18.1'
  },
  {
    id: 'Comfy.Sidebar.Style',
    category: ['Appearance', 'Sidebar', 'Style'],
    name: 'Sidebar style',
    type: 'combo',
    options: ['floating', 'connected'],
    defaultValue: 'connected'
  },
  {
    id: 'Comfy.TextareaWidget.FontSize',
    category: ['Appearance', 'Node Widget', 'TextareaWidget', 'FontSize'],
    name: 'Textarea widget font size',
    type: 'slider',
    defaultValue: 10,
    attrs: {
      min: 8,
      max: 24
    }
  },
  {
    id: 'Comfy.TextareaWidget.Spellcheck',
    category: ['Comfy', 'Node Widget', 'TextareaWidget', 'Spellcheck'],
    name: 'Textarea widget spellcheck',
    type: 'boolean',
    defaultValue: false
  },
  {
    id: 'Comfy.Workflow.SortNodeIdOnSave',
    name: 'Sort node IDs when saving workflow',
    type: 'boolean',
    defaultValue: false
  },
  {
    id: 'Comfy.Canvas.NavigationMode',
    category: ['LiteGraph', 'Canvas Navigation', 'NavigationMode'],
    name: 'Navigation Mode',
    defaultValue: 'legacy',
    type: 'combo',
    sortOrder: 100,
    options: [
      { value: 'standard', text: 'Standard (New)' },
      { value: 'legacy', text: 'Drag Navigation' },
      { value: 'custom', text: 'Custom' }
    ],
    versionAdded: '1.25.0',
    defaultsByInstallVersion: {
      '1.25.0': 'legacy'
    },
    onChange: async (val: unknown, old?: unknown) => {
      const newValue = val as string
      const oldValue = old as string | undefined
      if (!oldValue) return
      const settingStore = useSettingStore()

      if (newValue === 'standard') {
        await settingStore.setMany({
          'Comfy.Canvas.LeftMouseClickBehavior': 'select',
          'Comfy.Canvas.MouseWheelScroll': 'panning'
        })
      } else if (newValue === 'legacy') {
        await settingStore.setMany({
          'Comfy.Canvas.LeftMouseClickBehavior': 'panning',
          'Comfy.Canvas.MouseWheelScroll': 'zoom'
        })
      }
    }
  },
  {
    id: 'Comfy.Canvas.LeftMouseClickBehavior',
    category: ['LiteGraph', 'Canvas Navigation', 'LeftMouseClickBehavior'],
    name: 'Left Mouse Click Behavior',
    defaultValue: 'panning',
    type: 'radio',
    sortOrder: 50,
    options: [
      { value: 'panning', text: 'Panning' },
      { value: 'select', text: 'Select' }
    ],
    versionAdded: '1.27.4',
    onChange: async (val: unknown) => {
      const newValue = val as string
      const settingStore = useSettingStore()

      const navigationMode = settingStore.get('Comfy.Canvas.NavigationMode')

      if (navigationMode !== 'custom') {
        if (
          (newValue === 'select' && navigationMode === 'standard') ||
          (newValue === 'panning' && navigationMode === 'legacy')
        ) {
          return
        }

        // only set to custom if it doesn't match the preset modes
        await settingStore.set('Comfy.Canvas.NavigationMode', 'custom')
      }
    }
  },
  {
    id: 'Comfy.Canvas.MouseWheelScroll',
    category: ['LiteGraph', 'Canvas Navigation', 'MouseWheelScroll'],
    name: 'Mouse Wheel Scroll',
    defaultValue: 'zoom',
    type: 'radio',
    options: [
      { value: 'panning', text: 'Panning' },
      { value: 'zoom', text: 'Zoom in/out' }
    ],
    versionAdded: '1.27.4',
    onChange: async (val: unknown) => {
      const newValue = val as string
      const settingStore = useSettingStore()

      const navigationMode = settingStore.get('Comfy.Canvas.NavigationMode')

      if (navigationMode !== 'custom') {
        if (
          (newValue === 'panning' && navigationMode === 'standard') ||
          (newValue === 'zoom' && navigationMode === 'legacy')
        ) {
          return
        }

        // only set to custom if it doesn't match the preset modes
        await settingStore.set('Comfy.Canvas.NavigationMode', 'custom')
      }
    }
  },
  {
    id: 'Comfy.Graph.CanvasInfo',
    category: ['LiteGraph', 'Canvas', 'CanvasInfo'],
    name: 'Show canvas info on bottom left corner (fps, etc.)',
    type: 'boolean',
    defaultValue: true
  },
  {
    id: 'Comfy.Node.ShowDeprecated',
    name: 'Show deprecated nodes in search',
    tooltip:
      'Deprecated nodes are hidden by default in the UI, but remain functional in existing workflows that use them.',
    type: 'boolean',
    defaultValue: false
  },
  {
    id: 'Comfy.Node.ShowExperimental',
    name: 'Show experimental nodes in search',
    tooltip:
      'Experimental nodes are marked as such in the UI and may be subject to significant changes or removal in future versions. Use with caution in production workflows',
    type: 'boolean',
    defaultValue: true
  },
  {
    id: 'Comfy.Node.Opacity',
    category: ['Appearance', 'Node', 'Opacity'],
    name: 'Node opacity',
    type: 'slider',
    defaultValue: 1,
    attrs: {
      min: 0.01,
      max: 1,
      step: 0.01
    }
  },
  {
    id: 'Comfy.Workflow.ShowMissingNodesWarning',
    name: 'Show missing nodes warning',
    type: 'boolean',
    defaultValue: true
  },
  {
    id: 'Comfy.Workflow.ShowMissingModelsWarning',
    name: 'Show missing models warning',
    type: isCloud ? 'hidden' : 'boolean',
    defaultValue: isCloud ? false : true,
    experimental: true
  },
  {
    id: 'Comfy.Workflow.WarnBlueprintOverwrite',
    name: 'Require confirmation to overwrite an existing subgraph blueprint',
    type: 'boolean',
    defaultValue: true
  },
  {
    id: 'Comfy.Graph.ZoomSpeed',
    category: ['LiteGraph', 'Canvas', 'ZoomSpeed'],
    name: 'Canvas zoom speed',
    type: 'slider',
    defaultValue: 1.1,
    attrs: {
      min: 1.01,
      max: 2.5,
      step: 0.01
    }
  },
  // Bookmarks are stored in the settings store.
  {
    id: 'Comfy.NodeLibrary.NewDesign',
    category: ['Comfy', 'Node Library', 'NewDesign'],
    name: 'New Node Library Design',
    type: 'boolean',
    tooltip:
      'Enable the redesigned node library sidebar with tabs (Essential, All, Custom), improved search, and hover previews.',
    defaultValue: true,
    experimental: true
  },
  // Bookmarks are in format of category/display_name. e.g. "conditioning/CLIPTextEncode"
  {
    id: 'Comfy.NodeLibrary.Bookmarks',
    name: 'Node library bookmarks with display name (deprecated)',
    type: 'hidden',
    defaultValue: [],
    deprecated: true
  },
  {
    id: 'Comfy.NodeLibrary.Bookmarks.V2',
    name: 'Node library bookmarks v2 with unique name',
    type: 'hidden',
    defaultValue: []
  },
  // Stores mapping from bookmark folder name to its customization.
  {
    id: 'Comfy.NodeLibrary.BookmarksCustomization',
    name: 'Node library bookmarks customization',
    type: 'hidden',
    defaultValue: {}
  },
  {
    id: 'Comfy.GroupSelectedNodes.Padding',
    category: ['LiteGraph', 'Group', 'Padding'],
    name: 'Group selected nodes padding',
    type: 'slider',
    defaultValue: 10,
    attrs: {
      min: 0,
      max: 100
    }
  },
  {
    id: 'Comfy.Node.DoubleClickTitleToEdit',
    category: ['LiteGraph', 'Node', 'DoubleClickTitleToEdit'],
    name: 'Double click node title to edit',
    type: 'boolean',
    defaultValue: true
  },
  {
    id: 'Comfy.Node.AllowImageSizeDraw',
    category: ['LiteGraph', 'Node Widget', 'AllowImageSizeDraw'],
    name: 'Show width × height below the image preview',
    type: 'boolean',
    defaultValue: true
  },
  {
    id: 'Comfy.Group.DoubleClickTitleToEdit',
    category: ['LiteGraph', 'Group', 'DoubleClickTitleToEdit'],
    name: 'Double click group title to edit',
    type: 'boolean',
    defaultValue: true
  },
  {
    id: 'Comfy.Window.UnloadConfirmation',
    name: 'Show confirmation when closing window',
    type: 'boolean',
    defaultValue: true,
    versionModified: '1.7.12'
  },
  {
    id: 'Comfy.TreeExplorer.ItemPadding',
    category: ['Appearance', 'Tree Explorer', 'ItemPadding'],
    name: 'Tree explorer item padding',
    type: 'slider',
    defaultValue: 2,
    attrs: {
      min: 0,
      max: 8,
      step: 1
    }
  },
  {
    id: 'Comfy.ModelLibrary.AutoLoadAll',
    name: 'Automatically load all model folders',
    tooltip:
      'If true, all folders will load as soon as you open the model library (this may cause delays while it loads). If false, root level model folders will only load once you click on them.',
    type: isCloud ? 'hidden' : 'boolean',
    defaultValue: false
  },
  {
    id: 'Comfy.ModelLibrary.NameFormat',
    name: 'What name to display in the model library tree view',
    tooltip:
      'Select "filename" to render a simplified view of the raw filename (without directory or ".safetensors" extension) in the model list. Select "title" to display the configurable model metadata title.',
    type: 'combo',
    options: ['filename', 'title'],
    defaultValue: 'title'
  },
  {
    id: 'Comfy.Locale',
    name: 'Language',
    type: 'combo',
    options: [
      { value: 'en', text: 'English' },
      { value: 'zh', text: '中文' },
      { value: 'zh-TW', text: '繁體中文' },
      { value: 'ru', text: 'Русский' },
      { value: 'ja', text: '日本語' },
      { value: 'ko', text: '한국어' },
      { value: 'fr', text: 'Français' },
      { value: 'es', text: 'Español' },
      { value: 'ar', text: 'عربي' },
      { value: 'tr', text: 'Türkçe' },
      { value: 'pt-BR', text: 'Português (BR)' },
      { value: 'fa', text: 'فارسی' }
    ],
    defaultValue: () => navigator.language.split('-')[0] || 'en'
  },
  {
    id: 'Comfy.NodeBadge.NodeSourceBadgeMode',
    category: ['LiteGraph', 'Node', 'NodeSourceBadgeMode'],
    name: 'Node source badge mode',
    type: 'combo',
    options: Object.values(NodeBadgeMode),
    defaultValue: NodeBadgeMode.HideBuiltIn
  },
  {
    id: 'Comfy.NodeBadge.NodeIdBadgeMode',
    category: ['LiteGraph', 'Node', 'NodeIdBadgeMode'],
    name: 'Node ID badge mode',
    type: 'combo',
    options: [NodeBadgeMode.None, NodeBadgeMode.ShowAll],
    defaultValue: NodeBadgeMode.None
  },
  {
    id: 'Comfy.NodeBadge.NodeLifeCycleBadgeMode',
    category: ['LiteGraph', 'Node', 'NodeLifeCycleBadgeMode'],
    name: 'Node life cycle badge mode',
    type: 'combo',
    options: [NodeBadgeMode.None, NodeBadgeMode.ShowAll],
    defaultValue: NodeBadgeMode.ShowAll
  },
  {
    id: 'Comfy.NodeBadge.ShowApiPricing',
    category: ['Comfy', 'API Nodes'],
    name: 'Show API node pricing badge',
    type: 'boolean',
    defaultValue: true,
    versionAdded: '1.20.3'
  },
  {
    id: 'Comfy.Notification.ShowVersionUpdates',
    category: ['Comfy', 'Notification Preferences'],
    name: 'Show version updates',
    tooltip: 'Show updates for new models, and major new features.',
    type: 'boolean',
    defaultValue: true
  },
  {
    id: 'Comfy.ConfirmClear',
    category: ['Comfy', 'Workflow', 'ConfirmClear'],
    name: 'Require confirmation when clearing workflow',
    type: 'boolean',
    defaultValue: true
  },
  {
    id: 'Comfy.PromptFilename',
    category: ['Comfy', 'Workflow', 'PromptFilename'],
    name: 'Prompt for filename when saving workflow',
    type: 'boolean',
    defaultValue: true
  },
  /**
   * file format for preview
   *
   * format;quality
   *
   * ex)
   * webp;50 -> webp, quality 50
   * jpeg;80 -> rgb, jpeg, quality 80
   *
   * @type {string}
   */
  {
    id: 'Comfy.PreviewFormat',
    category: ['LiteGraph', 'Node Widget', 'PreviewFormat'],
    name: 'Preview image format',
    tooltip:
      'When displaying a preview in the image widget, convert it to a lightweight image, e.g. webp, jpeg, webp;50, etc.',
    type: 'text',
    defaultValue: ''
  },
  {
    id: 'Comfy.DisableSliders',
    category: ['LiteGraph', 'Node Widget', 'DisableSliders'],
    name: 'Disable node widget sliders',
    type: 'boolean',
    defaultValue: false
  },
  {
    id: 'Comfy.DisableFloatRounding',
    category: ['LiteGraph', 'Node Widget', 'DisableFloatRounding'],
    name: 'Disable default float widget rounding.',
    tooltip:
      '(requires page reload) Cannot disable round when round is set by the node in the backend.',
    type: 'boolean',
    defaultValue: false
  },
  {
    id: 'Comfy.FloatRoundingPrecision',
    category: ['LiteGraph', 'Node Widget', 'FloatRoundingPrecision'],
    name: 'Float widget rounding decimal places [0 = auto].',
    tooltip: '(requires page reload)',
    type: 'slider',
    attrs: {
      min: 0,
      max: 6,
      step: 1
    },
    defaultValue: 0
  },
  {
    id: 'LiteGraph.Node.TooltipDelay',
    name: 'Tooltip Delay',
    type: 'number',
    attrs: {
      min: 100,
      max: 3000,
      step: 50
    },
    defaultValue: 500,
    versionAdded: '1.9.0'
  },
  {
    id: 'Comfy.EnableTooltips',
    category: ['LiteGraph', 'Node', 'EnableTooltips'],
    name: 'Enable Tooltips',
    type: 'boolean',
    defaultValue: true
  },
  {
    id: 'Comfy.DevMode',
    name: 'Enable dev mode options (API save, etc.)',
    type: 'boolean',
    defaultValue: false,
    onChange: (value) => {
      const element = document.getElementById('comfy-dev-save-api-button')
      if (element) {
        element.style.display = value ? 'flex' : 'none'
      }
    }
  },
  {
    id: 'Comfy.UI.TabBarLayout',
    category: ['Appearance', 'General'],
    name: 'Tab Bar Layout',
    type: 'combo',
    options: ['Default', 'Integrated'],
    tooltip:
      'Controls the layout of the tab bar. "Integrated" moves Help and User controls into the tab bar area.',
    defaultValue: 'Default'
  },
  {
    id: 'Comfy.UseNewMenu',
    category: ['Comfy', 'Menu', 'UseNewMenu'],
    defaultValue: 'Top',
    name: 'Use new menu',
    type: 'combo',
    options: ['Disabled', 'Top'],
    tooltip: 'Enable the redesigned top menu bar.',
    migrateDeprecatedValue: (val: unknown) => {
      const value = val as string
      // Floating is now supported by dragging the docked actionbar off.
      if (value === 'Floating') {
        return 'Top'
      } else if (value === 'Bottom') {
        return 'Top'
      }
      return value
    }
  },
  {
    id: 'Comfy.Workflow.WorkflowTabsPosition',
    name: 'Opened workflows position',
    type: 'combo',
    options: ['Sidebar', 'Topbar'],
    defaultValue: 'Topbar',
    migrateDeprecatedValue: (val: unknown) => {
      const value = val as string
      if (value === 'Topbar (2nd-row)') {
        return 'Topbar'
      }
      return value
    }
  },
  {
    id: 'Comfy.Graph.CanvasMenu',
    category: ['LiteGraph', 'Canvas', 'CanvasMenu'],
    name: 'Show graph canvas menu',
    type: 'boolean',
    defaultValue: true
  },
  {
    id: 'Comfy.QueueButton.BatchCountLimit',
    name: 'Batch count limit',
    tooltip:
      'The maximum number of tasks added to the queue at one button click',
    type: 'number',
    defaultValue: isCloud ? 32 : 100,
    versionAdded: '1.3.5'
  },
  {
    id: 'Comfy.Keybinding.UnsetBindings',
    name: 'Keybindings unset by the user',
    type: 'hidden',
    defaultValue: [] as Keybinding[],
    versionAdded: '1.3.7',
    versionModified: '1.7.3',
    migrateDeprecatedValue: (val: unknown) => {
      const value = val as (Keybinding & { targetSelector?: string })[]
      return value.map((keybinding) => {
        if (keybinding.targetSelector === '#graph-canvas') {
          keybinding.targetElementId = 'graph-canvas-container'
        }
        return keybinding
      })
    }
  },
  {
    id: 'Comfy.Keybinding.NewBindings',
    name: 'Keybindings set by the user',
    type: 'hidden',
    defaultValue: [] as Keybinding[],
    versionAdded: '1.3.7'
  },
  {
    id: 'Comfy.Extension.Disabled',
    name: 'Disabled extension names',
    type: 'hidden',
    defaultValue: [] as string[],
    versionAdded: '1.3.11'
  },
  {
    id: 'Comfy.LinkRenderMode',
    category: ['LiteGraph', 'Graph', 'LinkRenderMode'],
    name: 'Link Render Mode',
    tooltip:
      'Controls the appearance and visibility of connection links between nodes on the canvas.',
    defaultValue: 2,
    type: 'combo',
    options: [
      { value: LiteGraph.STRAIGHT_LINK, text: 'Straight' },
      { value: LiteGraph.LINEAR_LINK, text: 'Linear' },
      { value: LiteGraph.SPLINE_LINK, text: 'Spline' },
      { value: LiteGraph.HIDDEN_LINK, text: 'Hidden' }
    ]
  },
  {
    id: 'Comfy.Node.AutoSnapLinkToSlot',
    category: ['LiteGraph', 'Node', 'AutoSnapLinkToSlot'],
    name: 'Auto snap link to node slot',
    tooltip:
      'When dragging a link over a node, the link automatically snap to a viable input slot on the node',
    type: 'boolean',
    defaultValue: true,
    versionAdded: '1.3.29'
  },
  {
    id: 'Comfy.Node.SnapHighlightsNode',
    category: ['LiteGraph', 'Node', 'SnapHighlightsNode'],
    name: 'Snap highlights node',
    tooltip:
      'When dragging a link over a node with viable input slot, highlight the node',
    type: 'boolean',
    defaultValue: true,
    versionAdded: '1.3.29'
  },
  {
    id: 'Comfy.Node.BypassAllLinksOnDelete',
    category: ['LiteGraph', 'Node', 'BypassAllLinksOnDelete'],
    name: 'Keep all links when deleting nodes',
    tooltip:
      'When deleting a node, attempt to reconnect all of its input and output links (bypassing the deleted node)',
    type: 'boolean',
    defaultValue: true,
    versionAdded: '1.3.40'
  },
  {
    id: 'Comfy.Node.MiddleClickRerouteNode',
    category: ['LiteGraph', 'Node', 'MiddleClickRerouteNode'],
    name: 'Middle-click creates a new Reroute node',
    type: 'boolean',
    defaultValue: true,
    versionAdded: '1.3.42'
  },
  {
    id: 'Comfy.Graph.LinkMarkers',
    category: ['LiteGraph', 'Link', 'LinkMarkers'],
    name: 'Link midpoint markers',
    defaultValue: LinkMarkerShape.Circle,
    type: 'combo',
    options: [
      { value: LinkMarkerShape.None, text: 'None' },
      { value: LinkMarkerShape.Circle, text: 'Circle' },
      { value: LinkMarkerShape.Arrow, text: 'Arrow' }
    ],
    versionAdded: '1.3.42'
  },
  {
    id: 'Comfy.DOMClippingEnabled',
    category: ['LiteGraph', 'Node', 'DOMClippingEnabled'],
    name: 'Enable DOM element clipping (enabling may reduce performance)',
    type: 'boolean',
    defaultValue: true
  },
  {
    id: 'Comfy.Graph.CtrlShiftZoom',
    category: ['LiteGraph', 'Canvas', 'CtrlShiftZoom'],
    name: 'Enable fast-zoom shortcut (Ctrl + Shift + Drag)',
    type: 'boolean',
    defaultValue: true,
    versionAdded: '1.4.0'
  },
  {
    id: 'Comfy.Graph.LiveSelection',
    category: ['LiteGraph', 'Canvas', 'LiveSelection'],
    name: 'Live selection',
    tooltip:
      'When enabled, nodes are selected/deselected in real-time as you drag the selection rectangle, similar to other design tools.',
    type: 'boolean',
    defaultValue: false,
    versionAdded: '1.36.1'
  },
  {
    id: 'Comfy.Pointer.ClickDrift',
    category: ['LiteGraph', 'Pointer', 'ClickDrift'],
    name: 'Pointer click drift (maximum distance)',
    tooltip:
      'If the pointer moves more than this distance while holding a button down, it is considered dragging (rather than clicking).\n\nHelps prevent objects from being unintentionally nudged if the pointer is moved whilst clicking.',
    experimental: true,
    type: 'slider',
    attrs: {
      min: 0,
      max: 20,
      step: 1
    },
    defaultValue: 6,
    versionAdded: '1.4.3'
  },
  {
    id: 'Comfy.Pointer.ClickBufferTime',
    category: ['LiteGraph', 'Pointer', 'ClickBufferTime'],
    name: 'Pointer click drift delay',
    tooltip:
      'After pressing a pointer button down, this is the maximum time (in milliseconds) that pointer movement can be ignored for.\n\nHelps prevent objects from being unintentionally nudged if the pointer is moved whilst clicking.',
    experimental: true,
    type: 'slider',
    attrs: {
      min: 0,
      max: 1000,
      step: 25
    },
    defaultValue: 150,
    versionAdded: '1.4.3'
  },
  {
    id: 'Comfy.Pointer.DoubleClickTime',
    category: ['LiteGraph', 'Pointer', 'DoubleClickTime'],
    name: 'Double click interval (maximum)',
    tooltip:
      'The maximum time in milliseconds between the two clicks of a double-click.  Increasing this value may assist if double-clicks are sometimes not registered.',
    type: 'slider',
    attrs: {
      min: 100,
      max: 1000,
      step: 50
    },
    defaultValue: 300,
    versionAdded: '1.4.3'
  },
  {
    id: 'Comfy.SnapToGrid.GridSize',
    category: ['LiteGraph', 'Canvas', 'GridSize'],
    name: 'Snap to grid size',
    type: 'slider',
    attrs: {
      min: 1,
      max: 500
    },
    tooltip:
      'When dragging and resizing nodes while holding shift they will be aligned to the grid, this controls the size of that grid.',
    defaultValue: LiteGraph.CANVAS_GRID_SIZE
  },
  // Keep the 'pysssss.SnapToGrid' setting id so we don't need to migrate setting values.
  // Using a new setting id can cause existing users to lose their existing settings.
  {
    id: 'pysssss.SnapToGrid',
    category: ['LiteGraph', 'Canvas', 'AlwaysSnapToGrid'],
    name: 'Always snap to grid',
    tooltip:
      'When enabled, nodes will automatically align to the grid when moved or resized.',
    type: 'boolean',
    defaultValue: false,
    versionAdded: '1.3.13'
  },
  {
    id: 'Comfy.Server.ServerConfigValues',
    name: 'Server config values for frontend display',
    tooltip: 'Server config values used for frontend display only',
    type: 'hidden',
    // Mapping from server config id to value.
    defaultValue: {} as Record<string, unknown>,
    versionAdded: '1.4.8'
  },
  {
    id: 'Comfy.Server.LaunchArgs',
    name: 'Server launch arguments',
    tooltip:
      'These are the actual arguments that are passed to the server when it is launched.',
    type: 'hidden',
    defaultValue: {} as Record<string, string>,
    versionAdded: '1.4.8'
  },
  {
    id: 'Comfy.Queue.MaxHistoryItems',
    name: 'Queue history size',
    tooltip: 'The maximum number of tasks that show in the queue history.',
    type: 'slider',
    attrs: {
      min: 2,
      max: 256,
      step: 2
    },
    defaultValue: 64,
    versionAdded: '1.4.12'
  },
  {
    id: 'Comfy.Queue.History.Expanded',
    name: 'Queue history expanded',
    type: 'hidden',
    defaultValue: false,
    versionAdded: '1.37.0'
  },
  {
    id: 'Comfy.Execution.PreviewMethod',
    category: ['Comfy', 'Execution', 'PreviewMethod'],
    name: 'Live preview method',
    tooltip:
      'Live preview method during image generation. "default" uses the server CLI setting.',
    type: 'combo',
    options: ['default', 'none', 'auto', 'latent2rgb', 'taesd'],
    defaultValue: 'default',
    versionAdded: '1.36.0'
  },
  {
    id: 'LiteGraph.Canvas.MaximumFps',
    name: 'Maximum FPS',
    tooltip:
      'The maximum frames per second that the canvas is allowed to render. Caps GPU usage at the cost of smoothness. If 0, the screen refresh rate is used. Default: 0',
    type: 'slider',
    attrs: {
      min: 0,
      max: 120
    },
    defaultValue: 0,
    versionAdded: '1.5.1'
  },
  {
    id: 'Comfy.EnableWorkflowViewRestore',
    category: ['Comfy', 'Workflow', 'EnableWorkflowViewRestore'],
    name: 'Save and restore canvas position and zoom level in workflows',
    type: 'boolean',
    defaultValue: true,
    versionModified: '1.5.4'
  },
  {
    id: 'Comfy.Workflow.ConfirmDelete',
    name: 'Show confirmation when deleting workflows',
    type: 'boolean',
    defaultValue: true,
    versionAdded: '1.5.6'
  },
  {
    id: 'Comfy.ColorPalette',
    name: 'The active color palette id',
    type: 'hidden',
    defaultValue: 'dark',
    versionModified: '1.6.7',
    migrateDeprecatedValue(val: unknown) {
      const value = val as string
      // Legacy custom palettes were prefixed with 'custom_'
      return value.startsWith('custom_') ? value.replace('custom_', '') : value
    }
  },
  {
    id: 'Comfy.CustomColorPalettes',
    name: 'Custom color palettes',
    type: 'hidden',
    defaultValue: {} as ColorPalettes,
    versionModified: '1.6.7'
  },
  {
    id: 'Comfy.WidgetControlMode',
    category: ['Comfy', 'Node Widget', 'WidgetControlMode'],
    name: 'Widget control mode',
    tooltip:
      'Controls when widget values are updated (randomize/increment/decrement), either before the prompt is queued or after.',
    type: 'combo',
    defaultValue: 'after',
    options: ['before', 'after'],
    versionModified: '1.6.10'
  },
  {
    id: 'Comfy.TutorialCompleted',
    name: 'Tutorial completed',
    type: 'hidden',
    defaultValue: false,
    versionAdded: '1.8.7'
  },
  {
    id: 'Comfy.InstalledVersion',
    name: 'The frontend version that was running when the user first installed ComfyUI',
    type: 'hidden',
    defaultValue: null,
    versionAdded: '1.24.0'
  },
  {
    id: 'LiteGraph.ContextMenu.Scaling',
    name: 'Scale node combo widget menus (lists) when zoomed in',
    defaultValue: false,
    type: 'boolean',
    versionAdded: '1.8.8'
  },

  {
    id: 'LiteGraph.Canvas.LowQualityRenderingZoomThreshold',
    type: 'hidden',
    deprecated: true,
    name: 'Low quality rendering zoom threshold (deprecated)',
    tooltip:
      'Zoom level threshold for performance mode. Lower values (0.1) = quality at all zoom levels. Higher values (1.0) = performance mode even when zoomed in. Performance mode simplifies rendering by hiding text labels, shadows, and details.',
    attrs: {
      min: 0.1,
      max: 1,
      step: 0.01
    },
    defaultValue: 0.6,
    versionAdded: '1.9.1',
    versionModified: '1.26.7'
  },
  {
    id: 'LiteGraph.Canvas.MinFontSizeForLOD',
    name: 'Zoom Node Level of Detail - font size threshold',
    tooltip:
      'Controls when the nodes switch to low quality LOD rendering. Uses font size in pixels to determine when to switch. Set to 0 to disable. Values 1-24 set the minimum font size threshold for LOD - higher values (24px) = switch nodes to simplified rendering sooner when zooming out, lower values (1px) = maintain full node quality longer.',
    type: 'slider',
    attrs: {
      min: 0,
      max: 24,
      step: 1
    },
    defaultValue: 8,
    versionAdded: '1.26.7',
    hideInVueNodes: true
  },
  {
    id: 'Comfy.Canvas.SelectionToolbox',
    category: ['LiteGraph', 'Canvas', 'SelectionToolbox'],
    name: 'Show selection toolbox',
    tooltip:
      'Display a floating toolbar when nodes are selected, providing quick access to common actions.',
    type: 'boolean',
    defaultValue: true,
    versionAdded: '1.10.5'
  },
  {
    id: 'LiteGraph.Reroute.SplineOffset',
    name: 'Reroute spline offset',
    tooltip: 'The bezier control point offset from the reroute centre point',
    type: 'slider',
    defaultValue: 20,
    attrs: {
      min: 0,
      max: 400
    },
    versionAdded: '1.15.7'
  },
  {
    id: 'Comfy.Toast.DisableReconnectingToast',
    name: 'Disable toasts when reconnecting or reconnected',
    type: 'hidden',
    defaultValue: false,
    versionAdded: '1.15.12'
  },
  {
    id: 'Comfy.Minimap.Visible',
    name: 'Display minimap on canvas',
    type: 'hidden',
    defaultValue: window.innerWidth >= breakpointsTailwind.lg,
    versionAdded: '1.25.0'
  },
  {
    id: 'Comfy.Minimap.NodeColors',
    name: 'Display node with its original color on minimap',
    type: 'hidden',
    defaultValue: false,
    versionAdded: '1.26.0'
  },
  {
    id: 'Comfy.Minimap.ShowLinks',
    name: 'Display links on minimap',
    type: 'hidden',
    defaultValue: true,
    versionAdded: '1.26.0'
  },
  {
    id: 'Comfy.Minimap.ShowGroups',
    name: 'Display node groups on minimap',
    type: 'hidden',
    defaultValue: true,
    versionAdded: '1.26.0'
  },
  {
    id: 'Comfy.Minimap.RenderBypassState',
    name: 'Render bypass state on minimap',
    type: 'hidden',
    defaultValue: true,
    versionAdded: '1.26.0'
  },
  {
    id: 'Comfy.Minimap.RenderErrorState',
    name: 'Render error state on minimap',
    type: 'hidden',
    defaultValue: true,
    versionAdded: '1.26.0'
  },
  {
    id: 'Comfy.Workflow.AutoSaveDelay',
    name: 'Auto Save Delay (ms)',
    defaultValue: 1000,
    type: 'number',
    tooltip: 'Only applies if Auto Save is set to "after delay".',
    versionAdded: '1.16.0'
  },
  {
    id: 'Comfy.Workflow.AutoSave',
    name: 'Auto Save',
    type: 'combo',
    options: ['off', 'after delay'], // Room for other options like on focus change, tab change, window change
    defaultValue: 'off', // Popular request by users (https://github.com/Comfy-Org/ComfyUI_frontend/issues/1584#issuecomment-2536610154)
    versionAdded: '1.16.0'
  },
  {
    id: 'Comfy.Workflow.Persist',
    name: 'Persist workflow state and restore on page (re)load',
    type: 'boolean',
    defaultValue: true,
    versionAdded: '1.16.1'
  },
  {
    id: 'LiteGraph.Node.DefaultPadding',
    name: 'Always shrink new nodes',
    tooltip:
      'Resize nodes to the smallest possible size when created. When disabled, a newly added node will be widened slightly to show widget values.',
    type: 'boolean',
    defaultValue: false,
    versionAdded: '1.18.0'
  },
  {
    id: 'Comfy.Canvas.BackgroundImage',
    category: ['Appearance', 'Canvas', 'Background'],
    name: 'Canvas background image',
    type: 'backgroundImage',
    tooltip:
      'Image URL for the canvas background. You can right-click an image in the outputs panel and select "Set as Background" to use it, or upload your own image using the upload button.',
    defaultValue: '',
    versionAdded: '1.20.4',
    versionModified: '1.20.5'
  },
  // Release data stored in settings
  {
    id: 'Comfy.Release.Version',
    name: 'Last seen release version',
    type: 'hidden',
    defaultValue: ''
  },
  {
    id: 'Comfy.Release.Status',
    name: 'Release status',
    type: 'hidden',
    defaultValue: 'skipped'
  },
  {
    id: 'Comfy.Release.Timestamp',
    name: 'Release seen timestamp',
    type: 'hidden',
    defaultValue: 0
  },

  /**
   * Template Library Filter Settings
   */
  {
    id: 'Comfy.Templates.SelectedModels',
    name: 'Template library - Selected model filters',
    type: 'hidden',
    defaultValue: []
  },
  {
    id: 'Comfy.Templates.SelectedUseCases',
    name: 'Template library - Selected use case filters',
    type: 'hidden',
    defaultValue: []
  },
  {
    id: 'Comfy.Templates.SelectedRunsOn',
    name: 'Template library - Selected runs on filters',
    type: 'hidden',
    defaultValue: []
  },
  {
    id: 'Comfy.Templates.SortBy',
    name: 'Template library - Sort preference',
    type: 'hidden',
    defaultValue: 'default'
  },

  /**
   * Nodes 2.0 Settings
   */
  {
    id: 'Comfy.VueNodes.Enabled',
    category: ['Comfy', 'Nodes 2.0', 'VueNodesEnabled'],
    name: 'Modern Node Design (Nodes 2.0)',
    type: 'boolean',
    tooltip:
      'Modern: DOM-based rendering with enhanced interactivity, native browser features, and updated visual design. Classic: Traditional canvas rendering.',
    defaultValue: false,
    defaultsByInstallVersion: { '1.41.0': isCloud },
    sortOrder: 100,
    experimental: true,
    versionAdded: '1.27.1'
  },
  {
    id: 'Comfy.VueNodes.AutoScaleLayout',
    category: ['Comfy', 'Nodes 2.0', 'AutoScaleLayout'],
    name: 'Auto-scale layout (Nodes 2.0)',
    tooltip:
      'Automatically scale node positions when switching to Nodes 2.0 rendering to prevent overlap',
    type: 'boolean',
    sortOrder: 50,
    experimental: true,
    defaultValue: true,
    versionAdded: '1.30.3'
  },
  {
    id: 'Comfy.Assets.UseAssetAPI',
    name: 'Use Asset API for model library',
    type: 'hidden',
    tooltip: 'Use new Asset API for model browsing',
    defaultValue: isCloud ? true : false,
    experimental: true
  },
  {
    id: 'Comfy.VersionCompatibility.DisableWarnings',
    name: 'Disable version compatibility warnings',
    type: 'hidden',
    defaultValue: false,
    versionAdded: '1.34.1'
  },
  {
    id: 'Comfy.RightSidePanel.IsOpen',
    name: 'Right side panel open state',
    type: 'hidden',
    defaultValue: false,
    versionAdded: '1.37.0'
  },
  {
    id: 'Comfy.Queue.QPOV2',
    category: ['Comfy', 'Queue', 'Layout'],
    name: 'Use the unified job queue in the Assets side panel',
    type: 'boolean',
    tooltip:
      'Replaces the floating job queue panel with an equivalent job queue embedded in the Assets side panel. You can disable this to return to the floating panel layout.',
    defaultValue: false,
    experimental: true
  },
  {
    id: 'Comfy.Node.AlwaysShowAdvancedWidgets',
    category: ['LiteGraph', 'Node Widget', 'AlwaysShowAdvancedWidgets'],
    name: 'Always show advanced widgets on all nodes',
    tooltip:
      'When enabled, advanced widgets are always visible on all nodes without needing to expand them individually.',
    type: 'boolean',
    defaultValue: false,
    versionAdded: '1.39.0'
  },
  {
    id: 'Comfy.NodeReplacement.Enabled',
    category: ['Comfy', 'Workflow', 'NodeReplacement'],
    name: 'Enable node replacement suggestions',
    tooltip:
      'When enabled, missing nodes with known replacements will be shown as replaceable in the missing nodes dialog, allowing you to review and apply replacements.',
    type: 'boolean',
    defaultValue: false,
    experimental: true,
    versionAdded: '1.40.0'
  },
  {
    id: 'Comfy.Graph.DeduplicateSubgraphNodeIds',
    category: ['Comfy', 'Graph', 'Subgraph'],
    name: 'Deduplicate subgraph node IDs',
    tooltip:
      'Automatically reassign duplicate node IDs in subgraphs when loading a workflow.',
    type: 'boolean',
    deprecated: true,
    defaultValue: true,
    experimental: true,
    versionAdded: '1.40.0'
  },
  {
    id: 'Comfy.RightSidePanel.ShowErrorsTab',
    category: ['Comfy', 'Error System'],
    name: 'Show errors tab in side panel',
    tooltip:
      'When enabled, an errors tab is displayed in the right side panel to show workflow execution errors at a glance.',
    type: 'boolean',
    defaultValue: true,
    experimental: true,
    versionAdded: '1.40.0'
  }
]
