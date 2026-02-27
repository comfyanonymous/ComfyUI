import type { Component } from 'vue'

import type { NodeId } from '@/platform/workflow/validation/schemas/workflowSchema'
import type { ExecutionErrorWsMessage, NodeError } from '@/schemas/apiSchema'
import type { useDialogService } from '@/services/dialogService'
import type { ComfyCommand } from '@/stores/commandStore'

interface BaseSidebarTabExtension {
  id: string
  title: string
  icon?: string | Component
  iconBadge?: string | (() => string | null)
  tooltip?: string
  label?: string
}

interface BaseBottomPanelExtension {
  id: string
  title?: string // For extensions that provide static titles
  titleKey?: string // For core tabs with i18n keys
  targetPanel?: 'terminal' | 'shortcuts'
}

export interface VueExtension {
  id: string
  type: 'vue'
  component: Component
}

export interface CustomExtension {
  id: string
  type: 'custom'
  render: (container: HTMLElement) => void
  destroy?: () => void
}

type VueSidebarTabExtension = BaseSidebarTabExtension & VueExtension
type CustomSidebarTabExtension = BaseSidebarTabExtension & CustomExtension
export type SidebarTabExtension =
  | VueSidebarTabExtension
  | CustomSidebarTabExtension

type VueBottomPanelExtension = BaseBottomPanelExtension & VueExtension
type CustomBottomPanelExtension = BaseBottomPanelExtension & CustomExtension
export type BottomPanelExtension =
  | VueBottomPanelExtension
  | CustomBottomPanelExtension

/**
 * Defines message options in Toast component.
 */
export interface ToastMessageOptions {
  /**
   * Severity level of the message.
   * @defaultValue info
   */
  severity?:
    | 'success'
    | 'info'
    | 'warn'
    | 'error'
    | 'secondary'
    | 'contrast'
    | undefined
  /**
   * Summary content of the message.
   */
  summary?: string | undefined
  /**
   * Detail content of the message.
   */
  detail?: string
  /**
   * Whether the message can be closed manually using the close icon.
   * @defaultValue true
   */
  closable?: boolean | undefined
  /**
   * Delay in milliseconds to close the message automatically.
   */
  life?: number | undefined
  /**
   * Key of the Toast to display the message.
   */
  group?: string | undefined
  /**
   * Style class of the message.
   */
  styleClass?: string | string[] | Record<string, boolean>
  /**
   * Style class of the content.
   * Matches PrimeVue Toast API which accepts Vue class bindings.
   */
  contentStyleClass?: string | string[] | Record<string, boolean>
}

export type ToastManager = {
  add(message: ToastMessageOptions): void
  remove(message: ToastMessageOptions): void
  removeAll(): void
}

export interface ExtensionManager {
  // Sidebar tabs
  registerSidebarTab(tab: SidebarTabExtension): void
  unregisterSidebarTab(id: string): void
  getSidebarTabs(): SidebarTabExtension[]

  toast: ToastManager
  dialog: ReturnType<typeof useDialogService>
  command: CommandManager
  setting: {
    get: <T = unknown>(id: string) => T | undefined
    set: <T = unknown>(id: string, value: T) => void
  }

  // Execution error state (read-only)
  lastNodeErrors: Record<NodeId, NodeError> | null
  lastExecutionError: ExecutionErrorWsMessage | null
}

export interface CommandManager {
  commands: ComfyCommand[]
  execute(
    command: string,
    options?: {
      errorHandler?: (error: unknown) => void
      metadata?: Record<string, unknown>
    }
  ): void
}
