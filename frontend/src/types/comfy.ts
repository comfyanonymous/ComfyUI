import type {
  IContextMenuValue,
  Positionable
} from '@/lib/litegraph/src/interfaces'
import type { LGraphCanvas, LGraphNode } from '@/lib/litegraph/src/litegraph'
import type { NodeReplacement } from '@/platform/nodeReplacement/types'
import type { SettingParams } from '@/platform/settings/types'
import type { ComfyWorkflowJSON } from '@/platform/workflow/validation/schemas/workflowSchema'
import type { Keybinding } from '@/platform/keybindings/types'
import type { ComfyNodeDef } from '@/schemas/nodeDefSchema'
import type { ComfyApp } from '@/scripts/app'
import type { ComfyWidgetConstructor } from '@/scripts/widgets'
import type { ComfyCommand } from '@/stores/commandStore'
import type { AuthUserInfo } from '@/types/authTypes'
import type { BottomPanelExtension } from '@/types/extensionTypes'

type Widgets = Record<string, ComfyWidgetConstructor>

export interface AboutPageBadge {
  label: string
  url: string
  icon: string
  severity?: 'danger' | 'warn'
}

type MenuCommandGroup = {
  /**
   * The path to the menu group.
   */
  path: string[]
  /**
   * Command ids.
   * Note: Commands must be defined in `commands` array in the extension.
   */
  commands: string[]
}

export interface TopbarBadge {
  text: string
  /**
   * Optional badge label (e.g., "BETA", "ALPHA", "NEW")
   */
  label?: string
  /**
   * Visual variant for the badge
   * - info: Default informational badge (white label, gray background)
   * - warning: Warning badge (orange theme, higher emphasis)
   * - error: Error/alert badge (red theme, highest emphasis)
   */
  variant?: 'info' | 'warning' | 'error'
  /**
   * Optional icon class (e.g., "pi-exclamation-triangle")
   * If not provided, variant will determine the default icon
   */
  icon?: string
  /**
   * Optional tooltip text to show on hover
   */
  tooltip?: string
}

/*
 * Action bar button definition: add buttons to the action bar
 */
export interface ActionBarButton {
  /**
   * Icon class to display (e.g., "icon-[lucide--message-circle-question-mark]")
   */
  icon: string
  /**
   * Optional label text to display next to the icon
   */
  label?: string
  /**
   * Optional tooltip text to show on hover
   */
  tooltip?: string
  /**
   * Optional CSS classes to apply to the button
   */
  class?: string
  /**
   * Click handler for the button
   */
  onClick: () => void
}

export type MissingNodeType =
  | string
  // Primarily used by group nodes.
  | {
      type: string
      nodeId?: string | number
      cnrId?: string
      hint?: string
      action?: {
        text: string
        callback: () => void
      }
      isReplaceable?: boolean
      replacement?: NodeReplacement
    }

export interface ComfyExtension {
  /**
   * The name of the extension
   */
  name: string
  /**
   * The commands defined by the extension
   */
  commands?: ComfyCommand[]
  /**
   * The keybindings defined by the extension
   */
  keybindings?: Keybinding[]
  /**
   * Menu commands to add to the menu bar
   */
  menuCommands?: MenuCommandGroup[]
  /**
   * Settings to add to the settings menu
   */
  settings?: SettingParams[]
  /**
   * Bottom panel tabs to add to the bottom panel
   */
  bottomPanelTabs?: BottomPanelExtension[]
  /**
   * Badges to add to the about page
   */
  aboutPageBadges?: AboutPageBadge[]
  /**
   * Badges to add to the top bar
   */
  topbarBadges?: TopbarBadge[]
  /**
   * Buttons to add to the action bar
   */
  actionBarButtons?: ActionBarButton[]
  /**
   * Allows any initialisation, e.g. loading resources. Called after the canvas is created but before nodes are added
   */
  init?(app: ComfyApp): Promise<void> | void
  /**
   * Allows any additional setup, called after the application is fully set up and running
   */
  setup?(app: ComfyApp): Promise<void> | void
  /**
   * Called before nodes are registered with the graph
   * @param defs The collection of node definitions, add custom ones or edit existing ones
   */
  addCustomNodeDefs?(
    defs: Record<string, ComfyNodeDef>,
    app: ComfyApp
  ): Promise<void> | void
  // TODO(huchenlei): We should deprecate the async return value of
  // getCustomWidgets.
  /**
   * Allows the extension to add custom widgets
   * @returns An array of {[widget name]: widget data}
   */
  getCustomWidgets?(app: ComfyApp): Promise<Widgets> | Widgets

  /**
   * Allows the extension to add additional commands to the selection toolbox
   * @param selectedItem The selected item on the canvas
   * @returns An array of command ids to add to the selection toolbox
   */
  getSelectionToolboxCommands?(selectedItem: Positionable): string[]

  /**
   * Allows the extension to add context menu items to canvas right-click menus
   * @param canvas The canvas instance
   * @returns An array of context menu items to add (null values represent separators)
   */
  getCanvasMenuItems?(canvas: LGraphCanvas): (IContextMenuValue | null)[]

  /**
   * Allows the extension to add context menu items to node right-click menus
   * @param node The node being right-clicked
   * @returns An array of context menu items to add (null values represent separators)
   */
  getNodeMenuItems?(node: LGraphNode): (IContextMenuValue | null)[]

  /**
   * Allows the extension to add additional handling to the node before it is registered with **LGraph**
   * @param nodeType The node class (not an instance)
   * @param nodeData The original node object info config object
   * @param app The app instance
   */
  beforeRegisterNodeDef?(
    nodeType: typeof LGraphNode,
    nodeData: ComfyNodeDef,
    app: ComfyApp
  ): Promise<void> | void

  /**
   * Allows the extension to modify the node definitions before they are used in the Vue app
   * Modifications is expected to be made in place.
   *
   * @param defs The node definitions
   * @param app The app instance
   */
  beforeRegisterVueAppNodeDefs?(defs: ComfyNodeDef[], app: ComfyApp): void

  /**
   * Allows the extension to register additional nodes with LGraph after standard nodes are added.
   * Custom node classes should extend **LGraphNode**.
   */
  registerCustomNodes?(app: ComfyApp): Promise<void> | void
  /**
   * Allows the extension to modify a node that has been reloaded onto the graph.
   * If you break something in the backend and want to patch workflows in the frontend
   * This is the place to do this
   * @param node The node that has been loaded
   * @param app The app instance
   */
  loadedGraphNode?(node: LGraphNode, app: ComfyApp): void
  /**
   * Allows the extension to run code after the constructor of the node
   * @param node The node that has been created
   * @param app The app instance
   */
  nodeCreated?(node: LGraphNode, app: ComfyApp): void

  /**
   * Allows the extension to modify the graph data before it is configured.
   * @param graphData The graph data
   * @param missingNodeTypes The missing node types
   * @param app The app instance
   */
  beforeConfigureGraph?(
    graphData: ComfyWorkflowJSON,
    missingNodeTypes: MissingNodeType[],
    app: ComfyApp
  ): Promise<void> | void

  /**
   * Allows the extension to run code after the graph is configured.
   * @param missingNodeTypes The missing node types
   * @param app The app instance
   */
  afterConfigureGraph?(
    missingNodeTypes: MissingNodeType[],
    app: ComfyApp
  ): Promise<void> | void

  /**
   * Fired whenever authentication resolves, providing the anonymized user id..
   * Extensions can register at any time and will receive the latest value immediately.
   * This is an experimental API and may be changed or removed in the future.
   */
  onAuthUserResolved?(user: AuthUserInfo, app: ComfyApp): Promise<void> | void

  /**
   * Fired whenever the auth token is refreshed.
   * This is an experimental API and may be changed or removed in the future.
   */
  onAuthTokenRefreshed?(): Promise<void> | void

  /**
   * Fired when user logs out.
   * This is an experimental API and may be changed or removed in the future.
   */
  onAuthUserLogout?(): Promise<void> | void

  [key: string]: unknown
}
